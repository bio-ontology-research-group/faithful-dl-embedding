"""Tests for sweep robustness: per-cell failure handling, sidecar cleanup, dataset reload, and posthoc/aggregate sidecar filtering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from experiments import aggregate, posthoc, sweep


class _DummyCM:
    names = ["A", "B"]


class _DummyBundle:
    def __init__(self, name: str = "tk3") -> None:
        self.name = name
        self.task = "satisfaction"


def _fake_payload(
    dataset: str,
    method: str,
    dim: int,
    seed: int,
    batch_size: int = 32768,
) -> Dict[str, Any]:
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


def test_sweep_generic_exception_writes_failed_json_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When a cell raises an exception, sweep logs FAILED with traceback once, writes <file>.failed.json, continues to the next cell, and aggregate ignores .failed.json."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"
    tables_dir = tmp_path / "tables"
    csv_path = tmp_path / "all_runs.csv"

    monkeypatch.setattr(sweep, "load_dataset", lambda ds: _DummyBundle(ds))
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
        payload = _fake_payload(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"]
        )
        out_file = Path(kwargs["out_dir"]) / sweep.run_filename(
            kwargs["dataset"],
            kwargs["method"],
            kwargs["dim"],
            kwargs["seed"],
            lr=kwargs.get("lr"),
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
    assert (
        "[  1/  2] FAILED tk3__elbe__d2__s0.json | ValueError: synthetic boom in elbe"
        in log_text
    )
    assert log_text.count("Traceback (most recent call last):") == 1
    assert "[  2/  2] DONE tk3__box2el__d2__s0.json" in log_text
    assert "ran=1 skipped=0 failed=1 total=2" in log_text

    # Verify aggregate.py ignores .failed.json files
    runs = aggregate.load_raw_runs(out_dir)
    assert len(runs) == 1
    assert runs[0]["_source_file"] == "tk3__box2el__d2__s0.json"
    rc_agg = aggregate.main(
        ["--raw", str(out_dir), "--out", str(tables_dir), "--csv", str(csv_path)]
    )
    assert rc_agg == 0


def test_posthoc_ignores_failed_json_sidecar(tmp_path: Path) -> None:
    """posthoc.process_raw_directory must skip *.failed.json sidecars without crashing or counting them."""
    raw_dir = tmp_path / "raw"
    ckpt_dir = tmp_path / "ckpt"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Write a .failed.json sidecar (no 'config' or 'metrics' keys) and a matching .pt
    # so that if posthoc did not filter '*.failed.json' it would load it and raise KeyError.
    sidecar = raw_dir / "tk3__elbe__d2__s0.json.failed.json"
    sidecar.write_text(
        json.dumps({"dataset": "tk3", "exception": "RuntimeError: boom"}),
        encoding="utf-8",
    )
    (ckpt_dir / f"{sidecar.stem}.pt").write_bytes(b"dummy")

    # Also write one valid run JSON without a checkpoint
    valid_json = raw_dir / "tk3__box2el__d2__s0.json"
    valid_json.write_text(
        json.dumps(_fake_payload("tk3", "box2el", 2, 0)),
        encoding="utf-8",
    )

    summary = posthoc.process_raw_directory(raw_dir=raw_dir, ckpt_dir=ckpt_dir)
    assert summary["total_json"] == 1
    assert summary["skipped_missing_ckpt"] == 1
    assert summary["updated"] == 0


def test_sweep_removes_stale_failed_json_on_later_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When a previously failed cell succeeds on a later sweep run, its stale .failed.json sidecar is removed."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    stale_sidecar = out_dir / "tk3__elbe__d2__s0.json.failed.json"
    stale_sidecar.write_text(
        json.dumps({"exception": "OutOfMemoryError: old failure"}),
        encoding="utf-8",
    )
    assert stale_sidecar.exists()

    monkeypatch.setattr(sweep, "load_dataset", lambda ds: _DummyBundle(ds))
    monkeypatch.setattr(sweep, "get_canonical_model", lambda b: _DummyCM())

    def _succeed_execute_run(**kwargs: Any) -> Dict[str, Any]:
        payload = _fake_payload(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"]
        )
        out_file = Path(kwargs["out_dir"]) / sweep.run_filename(
            kwargs["dataset"],
            kwargs["method"],
            kwargs["dim"],
            kwargs["seed"],
            lr=kwargs.get("lr"),
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(sweep, "execute_run", _succeed_execute_run)

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
    assert (out_dir / "tk3__elbe__d2__s0.json").exists()
    assert not stale_sidecar.exists()


def test_sweep_reloads_dataset_after_dataset_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If dataset load/CM build fails on a cell, current_ds_name is reset so a subsequent cell of the same dataset reloads it."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"

    load_calls: List[str] = []

    def _flaky_load_dataset(ds: str) -> _DummyBundle:
        load_calls.append(ds)
        if ds == "tk4" and load_calls.count("tk4") == 1:
            raise RuntimeError("transient dataset load failure")
        return _DummyBundle(ds)

    monkeypatch.setattr(sweep, "load_dataset", _flaky_load_dataset)
    monkeypatch.setattr(sweep, "get_canonical_model", lambda b: _DummyCM())

    seen_bundles: List[Any] = []

    def _ok_execute_run(**kwargs: Any) -> Dict[str, Any]:
        seen_bundles.append((kwargs["dataset"], kwargs["bundle"], kwargs["cm"]))
        payload = _fake_payload(
            kwargs["dataset"], kwargs["method"], kwargs["dim"], kwargs["seed"]
        )
        out_file = Path(kwargs["out_dir"]) / sweep.run_filename(
            kwargs["dataset"],
            kwargs["method"],
            kwargs["dim"],
            kwargs["seed"],
            lr=kwargs.get("lr"),
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(sweep, "execute_run", _ok_execute_run)

    rc = sweep.main(
        [
            "--suite",
            "synthetic",
            "--datasets",
            "tk3",
            "tk4",
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
    assert load_calls == ["tk3", "tk4", "tk4"]
    assert len(seen_bundles) == 3
    for ds_name, bundle, cm in seen_bundles:
        assert bundle is not None and bundle.name == ds_name
        assert cm is not None


def test_sweep_returns_nonzero_when_all_pending_cells_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exit code is 1 if all pending cells fail, and sweep does not do outer OOM retries."""
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sweep, "load_dataset", lambda ds: _DummyBundle(ds))
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
    assert calls["n"] == 1
    assert (out_dir / "tk3__elbe__d2__s0.json.failed.json").exists()
    log_text = (log_dir / "sweep_synthetic.log").read_text(encoding="utf-8")
    assert "ran=0 skipped=0 failed=1 total=1" in log_text
