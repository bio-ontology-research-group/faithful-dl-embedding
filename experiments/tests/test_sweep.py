"""Tests for sweep robustness: OOM retry with batch halving and per-cell failure handling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from experiments import aggregate, sweep


class _DummyCM:
    names = ["A", "B"]


class _DummyBundle:
    name = "tk3"
    task = "satisfaction"


def _fake_payload(dataset: str, method: str, dim: int, seed: int, batch_size: int) -> Dict[str, Any]:
    return {
        "config": {
            "dataset": dataset,
            "task": "satisfaction",
            "method": method,
            "model": "ELBE",
            "n_components": 1,
            "init": None,
            "anchor_weight": 0.0,
            "dim": dim,
            "effective_dim": dim,
            "seed": seed,
            "epochs": 2,
            "batch_size": batch_size,
            "effective_batch_size": batch_size,
            "oom_retries": 0,
            "device": "cpu",
        },
        "metrics": {
            "modelness_all": 1.0,
            "helly_violation_rate": 0.0,
            "rank_f_mrr": None,
            "peak_gpu_memory_mb": 10.0,
        },
        "timings": {
            "total_train_time_s": 0.1,
            "mean_epoch_time_s": 0.05,
            "wall_clock_s": 0.2,
            "eval_time_s": 0.05,
        },
        "env": {
            "git_hash_experiments": "abcdef1",
            "git_hash_mowl": "1234567",
            "git_dirty_experiments": False,
            "git_dirty_mowl": False,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": None,
        },
    }


def test_sweep_oom_retries_twice_with_halved_batch_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When execute_run raises OOM twice then succeeds, the cell completes, two retries are logged, and batch_size is halved twice."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sweep, "load_dataset", lambda ds: _DummyBundle())
    monkeypatch.setattr(sweep, "get_canonical_model", lambda b: _DummyCM())

    empty_cache_calls = {"n": 0}
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.__setitem__("n", empty_cache_calls["n"] + 1),
    )

    seen_batch_sizes: List[int] = []

    def _flaky_execute_run(**kwargs: Any) -> Dict[str, Any]:
        bs = int(kwargs.get("batch_size", 32768))
        seen_batch_sizes.append(bs)
        if len(seen_batch_sizes) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory on attempt 1\ndetails line 2")
        if len(seen_batch_sizes) == 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.88 GiB.")
        payload = _fake_payload(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"], bs
        )
        out_file = Path(kwargs["out_dir"]) / sweep.run_filename(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"], lr=kwargs.get("lr")
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(sweep, "execute_run", _flaky_execute_run)

    rc = sweep.main(
        [
            "--suite",
            "synthetic",
            "--datasets",
            "tk3",
            "--methods",
            "elbe",
            "--dims",
            "2",
            "--seeds",
            "0",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--runs-dir",
            str(runs_dir),
        ]
    )

    assert rc == 0
    assert seen_batch_sizes == [32768, 16384, 8192]
    assert empty_cache_calls["n"] >= 2

    out_json = out_dir / "tk3__elbe__d2__s0.json"
    failed_json = out_dir / "tk3__elbe__d2__s0.json.failed.json"
    assert out_json.exists()
    assert not failed_json.exists()

    log_text = (log_dir / "sweep_synthetic.log").read_text(encoding="utf-8")
    retry_lines = [line for line in log_text.splitlines() if "RETRY" in line]
    assert len(retry_lines) == 2
    assert "eff_bs=16384" in retry_lines[0]
    assert "eff_bs=8192" in retry_lines[1]
    assert "DONE tk3__elbe__d2__s0.json" in log_text
    assert "ran=1 skipped=0 failed=0 total=1" in log_text


def test_sweep_generic_exception_writes_failed_json_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When a cell raises a generic exception, sweep logs FAILED with traceback, writes <file>.failed.json, continues to the next cell, and aggregate ignores .failed.json."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"
    tables_dir = tmp_path / "tables"
    csv_path = tmp_path / "all_runs.csv"

    monkeypatch.setattr(sweep, "load_dataset", lambda ds: _DummyBundle())
    monkeypatch.setattr(sweep, "get_canonical_model", lambda b: _DummyCM())

    empty_cache_calls = {"n": 0}
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.__setitem__("n", empty_cache_calls["n"] + 1),
    )

    def _selective_execute_run(**kwargs: Any) -> Dict[str, Any]:
        if kwargs["method"] == "elbe":
            raise ValueError("synthetic boom in elbe\nextra detail line")
        bs = int(kwargs.get("batch_size", 32768))
        payload = _fake_payload(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"], bs
        )
        out_file = Path(kwargs["out_dir"]) / sweep.run_filename(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"], lr=kwargs.get("lr")
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(sweep, "execute_run", _selective_execute_run)

    rc = sweep.main(
        [
            "--suite",
            "synthetic",
            "--datasets",
            "tk3",
            "--methods",
            "elbe",
            "box2el",
            "--dims",
            "2",
            "--seeds",
            "0",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--runs-dir",
            str(runs_dir),
        ]
    )

    assert rc == 0
    assert empty_cache_calls["n"] >= 1

    failed_file = out_dir / "tk3__elbe__d2__s0.json.failed.json"
    ok_file = out_dir / "tk3__box2el__d2__s0.json"
    assert failed_file.exists()
    assert not (out_dir / "tk3__elbe__d2__s0.json").exists()
    assert ok_file.exists()

    failed_data = json.loads(failed_file.read_text(encoding="utf-8"))
    assert "synthetic boom in elbe" in failed_data["exception"]
    assert "Traceback" in failed_data["traceback"]

    log_text = (log_dir / "sweep_synthetic.log").read_text(encoding="utf-8")
    assert "[  1/  2] FAILED tk3__elbe__d2__s0.json | ValueError: synthetic boom in elbe" in log_text
    assert "Traceback (most recent call last):" in log_text
    assert "[  2/  2] DONE tk3__box2el__d2__s0.json" in log_text
    assert "ran=1 skipped=0 failed=1 total=2" in log_text

    # Verify aggregate.py ignores .failed.json files
    runs = aggregate.load_raw_runs(out_dir)
    assert len(runs) == 1
    assert runs[0]["_source_file"] == "tk3__box2el__d2__s0.json"
    rc_agg = aggregate.main(["--raw", str(out_dir), "--out", str(tables_dir), "--csv", str(csv_path)])
    assert rc_agg == 0


def test_sweep_returns_nonzero_when_all_pending_cells_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exit code is 1 if all pending cells fail (and 0 if any cell ran)."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sweep, "load_dataset", lambda ds: _DummyBundle())
    monkeypatch.setattr(sweep, "get_canonical_model", lambda b: _DummyCM())

    calls = {"n": 0}

    def _always_oom(**kwargs: Any) -> Dict[str, Any]:
        calls["n"] += 1
        raise torch.OutOfMemoryError("persistent OOM")

    monkeypatch.setattr(sweep, "execute_run", _always_oom)

    rc = sweep.main(
        [
            "--suite",
            "synthetic",
            "--datasets",
            "tk3",
            "--methods",
            "elbe",
            "--dims",
            "2",
            "--seeds",
            "0",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--runs-dir",
            str(runs_dir),
        ]
    )

    assert rc == 1
    assert calls["n"] == 4  # initial + 3 OOM retries
    assert (out_dir / "tk3__elbe__d2__s0.json.failed.json").exists()
    log_text = (log_dir / "sweep_synthetic.log").read_text(encoding="utf-8")
    assert "ran=0 skipped=0 failed=1 total=1" in log_text
