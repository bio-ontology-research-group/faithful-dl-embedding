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


def test_sweep_structural_suite_cells() -> None:
    """Verify structural suite enumerates 384 cells at default grid and 8 cells (one per method at 5 epochs) under --smoke."""
    from experiments.config import LR_GRID, STRUCTURAL_BASE_METHODS, SYNTHETIC_DATASETS

    cells_full = sweep.enumerate_cells("structural", smoke=False)
    assert len(cells_full) == 384
    assert {c.dataset for c in cells_full} == set(SYNTHETIC_DATASETS)
    assert {c.method for c in cells_full} == set(STRUCTURAL_BASE_METHODS)
    assert {c.dim for c in cells_full} == {2, 10}
    assert {c.seed for c in cells_full} == {0, 1, 2}
    assert {c.lr for c in cells_full} == set(LR_GRID)
    assert all(c.epochs == 2000 for c in cells_full)

    cells_smoke = sweep.enumerate_cells("structural", smoke=True)
    assert len(cells_smoke) == len(STRUCTURAL_BASE_METHODS) == 8
    assert [c.method for c in cells_smoke] == list(STRUCTURAL_BASE_METHODS)
    assert all(c.epochs == 5 for c in cells_smoke)
    assert all(c.seed == 0 for c in cells_smoke)


def test_sweep_structural_extra_suite_cells() -> None:
    """Verify structural-extra suite enumerates 144 cells at default lr=[0.01], supports --lr override, and enumerates 6 cells under --smoke."""
    from experiments.config import (
        STRUCTURAL_BASE_METHODS,
        STRUCTURAL_EXTRA_METHODS,
        SYNTHETIC_DATASETS,
    )

    cells_full = sweep.enumerate_cells("structural-extra", smoke=False)
    assert len(cells_full) == 144
    assert {c.dataset for c in cells_full} == set(SYNTHETIC_DATASETS)
    assert {c.method for c in cells_full} == set(STRUCTURAL_EXTRA_METHODS)
    assert {c.method for c in cells_full}.isdisjoint(set(STRUCTURAL_BASE_METHODS))
    assert {c.dim for c in cells_full} == {2, 10}
    assert {c.seed for c in cells_full} == {0, 1, 2}
    assert {c.lr for c in cells_full} == {0.01}
    assert all(c.epochs == 2000 for c in cells_full)

    cells_override_lr = sweep.enumerate_cells(
        "structural-extra", smoke=False, lrs=[0.001, 0.01]
    )
    assert len(cells_override_lr) == 288
    assert {c.lr for c in cells_override_lr} == {0.001, 0.01}

    cells_smoke = sweep.enumerate_cells("structural-extra", smoke=True)
    assert len(cells_smoke) == len(STRUCTURAL_EXTRA_METHODS) == 6
    assert [c.method for c in cells_smoke] == list(STRUCTURAL_EXTRA_METHODS)
    assert all(c.epochs == 5 for c in cells_smoke)
    assert all(c.seed == 0 for c in cells_smoke)
    assert all(c.dim == 2 for c in cells_smoke)
    assert all(c.dataset == "tk3" for c in cells_smoke)
    assert all(c.lr == 0.01 for c in cells_smoke)


def test_aggregate_structural_table(tmp_path: Path) -> None:
    """Verify aggregate.structural_table writes STRUCTURAL.md with Deg+ExistViol in pass criterion, Eff-k (mat.), ExistViol, Best lr parsing, synthetic filtering, and deduplication."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_md = tmp_path / "STRUCTURAL.md"

    # Synthetic runs for multibox-k4-struct on tk3 at d=2:
    # lr=0.001: 1/3 seeds pass; lr=0.01: 3/3 seeds pass (HVR=0, Deg=0, ExistViol=0, Mod=1.0) -> PASS,
    # success fraction 4/6, best_lr=0.01.
    # Also write an older duplicate JSON for (seed=1, lr=0.01) that failed, and a newer one that passed,
    # to test deduplication by (seed, lr) keeping the newest.
    import os

    for lr, pass_seeds in ((0.001, {0}), (0.01, {0, 1, 2})):
        for s in (0, 1, 2):
            passed = s in pass_seeds
            rec = {
                "config": {
                    "dataset": "tk3",
                    "task": "satisfaction",
                    "method": "multibox-k4-struct",
                    "model": "MultiBoxEL",
                    "n_components": 4,
                    "dim": 2,
                    "effective_dim": 2,
                    "seed": s,
                    "lr": lr,
                    "learning_rate": lr,
                    "objective": "structural",
                },
                "metrics": {
                    "modelness_all": 1.0 if passed else 0.85,
                    "helly_violation_rate": 0.0 if passed else 1.0,
                    "degeneracy_rate": 0.0,
                    "existence_violation_rate": 0.0 if passed else 0.25,
                    "mean_effective_components": 4.0,
                    "mean_effective_components_materialised": 2.5,
                    "mean_epoch_time_s": 0.012,
                    "n_defined": 3,
                    "reseed_moves_total": 6,
                },
                "timings": {
                    "mean_epoch_time_s": 0.012,
                },
            }
            fname = f"tk3__multibox-k4-struct__d2__s{s}__lr{lr:g}.json"
            fpath = raw_dir / fname
            fpath.write_text(json.dumps(rec), encoding="utf-8")
            os.utime(fpath, (2000.0, 2000.0))

    # Write a stale duplicate for seed=1, lr=0.01 with older mtime that had modelness_all=0.10 (should be ignored by dedupe)
    stale_rec = json.loads((raw_dir / "tk3__multibox-k4-struct__d2__s1__lr0.01.json").read_text(encoding="utf-8"))
    stale_rec["metrics"]["modelness_all"] = 0.10
    stale_rec["metrics"]["helly_violation_rate"] = 1.0
    stale_path = raw_dir / "tk3__multibox-k4-struct__d2__s1__lr0.01__old.json"
    stale_path.write_text(json.dumps(stale_rec), encoding="utf-8")
    os.utime(stale_path, (1000.0, 1000.0))

    # Synthetic runs for prism-struct on tk3 at d=2:
    # All 3 seeds have Mod=1.0, HVR=0.0, BUT Deg=1.0 and ExistViol=1.0 (degenerate solution) -> must FAIL!
    for s in (0, 1, 2):
        rec = {
            "config": {
                "dataset": "tk3",
                "task": "satisfaction",
                "method": "prism-struct",
                "model": "PrismEL",
                "n_components": 1,
                "dim": 2,
                "effective_dim": 3,
                "seed": s,
                "lr": 0.01,
                "learning_rate": 0.01,
                "objective": "structural",
            },
            "metrics": {
                "modelness_all": 1.0,
                "helly_violation_rate": 0.0,
                "degeneracy_rate": 1.0,
                "existence_violation_rate": 1.0,
                "mean_effective_components": 1.0,
                "mean_effective_components_materialised": 0.0,
                "mean_epoch_time_s": 0.008,
                "n_defined": 3,
                "reseed_moves_total": 0,
            },
            "timings": {
                "mean_epoch_time_s": 0.008,
            },
        }
        fname = f"tk3__prism-struct__d2__s{s}__lr0.01.json"
        (raw_dir / fname).write_text(json.dumps(rec), encoding="utf-8")

    # Non-synthetic dataset (e.g. 'go') with structural objective must be filtered out of Synthetic Benchmarks
    go_rec = {
        "config": {
            "dataset": "go",
            "task": "subsumption",
            "method": "multibox-k4-struct",
            "model": "MultiBoxEL",
            "n_components": 4,
            "dim": 50,
            "effective_dim": 50,
            "seed": 0,
            "lr": 0.001,
            "learning_rate": 0.001,
            "objective": "structural",
        },
        "metrics": {
            "modelness_all": 1.0,
            "helly_violation_rate": 0.0,
            "degeneracy_rate": 0.0,
            "existence_violation_rate": 0.0,
        },
    }
    (raw_dir / "go__multibox-k4-struct__d50__s0.json").write_text(
        json.dumps(go_rec), encoding="utf-8"
    )

    # Pizza sanity run for multibox-k4-struct
    pizza_rec = {
        "config": {
            "dataset": "pizza",
            "task": "subsumption",
            "method": "multibox-k4-struct",
            "model": "MultiBoxEL",
            "n_components": 4,
            "dim": 50,
            "effective_dim": 50,
            "seed": 0,
            "lr": 0.001,
            "learning_rate": 0.001,
            "objective": "structural",
        },
        "metrics": {
            "rank_f_mrr": 0.42,
            "auc_pos_neg": 0.91,
            "fvr": 0.07,
            "modelness_all": 0.995,
            "mean_epoch_time_s": 0.055,
            "n_defined": 12,
            "reseed_moves_total": 0,
        },
        "timings": {
            "mean_epoch_time_s": 0.055,
        },
    }
    (raw_dir / "pizza__multibox-k4-struct__d50__s0.json").write_text(
        json.dumps(pizza_rec), encoding="utf-8"
    )

    md = aggregate.structural_table(raw_dir, out_md)
    assert out_md.exists()
    assert out_md.read_text(encoding="utf-8") == md

    assert "`go`" not in md
    assert "Eff-k (mat.)" in md
    assert "ExistViol" in md

    header_line = next(
        line for line in md.splitlines() if line.startswith("| Dataset |")
    )
    headers = [c.strip() for c in header_line.strip("|").split("|")]
    best_lr_idx = headers.index("Best lr")
    pass_idx = headers.index("Pass")
    succ_idx = headers.index("Success (all lr)")
    eff_k_idx = headers.index("Eff-k (mat.)")
    exist_idx = headers.index("ExistViol")

    mb_lines = [line for line in md.splitlines() if "`multibox-k4-struct`" in line and "`tk3`" in line]
    assert len(mb_lines) == 1
    mb_cells = [c.strip() for c in mb_lines[0].strip("|").split("|")]
    assert mb_cells[pass_idx] == "PASS"
    assert mb_cells[succ_idx].startswith("4/6")
    assert float(mb_cells[best_lr_idx]) == pytest.approx(0.01)
    assert mb_cells[eff_k_idx].startswith("2.50")
    assert mb_cells[exist_idx].startswith("0.0000")

    pr_lines = [line for line in md.splitlines() if "`prism-struct`" in line and "`tk3`" in line]
    assert len(pr_lines) == 1
    pr_cells = [c.strip() for c in pr_lines[0].strip("|").split("|")]
    assert pr_cells[pass_idx] == "FAIL"
    assert pr_cells[succ_idx].startswith("0/3")

    assert "Pizza" in md
    assert "0.4200" in md
    assert "0.9100" in md
    assert "0.0700" in md


