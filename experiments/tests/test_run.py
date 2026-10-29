"""Tests for the experiment runner, sweep, and aggregation (Task 8)."""
from __future__ import annotations

from pathlib import Path
from typing import List
import json

import mowl
mowl.init_jvm("8g")

import pytest
import torch

from experiments.config import (
    DATASETS,
    DATASET_DIMS,
    DATASET_SEEDS,
    EXPERIMENTS_DIR,
    METHODS,
    SUBSUMPTION_DATASETS,
    SYNTHETIC_DATASETS,
)
from experiments import aggregate, metrics, run, sweep

SCRATCH_DIR = Path("~/tmp/gfe-runs/test-scratch").expanduser().resolve()


@pytest.fixture
def scratch_dir(request: pytest.FixtureRequest) -> Path:
    """Return a clean per-test scratch directory under ~/tmp/gfe-runs/test-scratch/."""
    p = SCRATCH_DIR / request.node.name
    if p.exists():
        for child in sorted(p.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_run_fails_without_json_on_metric_error(
    monkeypatch: pytest.MonkeyPatch,
    scratch_dir: Path,
) -> None:
    """If a metric function raises during run.main, it must exit non-zero and write no JSON."""

    def _boom(*args, **kwargs):
        raise RuntimeError("injected modelness failure")

    monkeypatch.setattr(metrics, "modelness", _boom)
    monkeypatch.setattr(run, "modelness", _boom, raising=False)

    out_dir = scratch_dir / "raw"
    runs_dir = scratch_dir / "checkpoints"
    expected_json = out_dir / "tk3__elbe__d2__s0.json"

    with pytest.raises((RuntimeError, SystemExit)) as exc_info:
        run.main(
            [
                "--dataset",
                "tk3",
                "--method",
                "elbe",
                "--dim",
                "2",
                "--seed",
                "0",
                "--epochs",
                "2",
                "--device",
                "cpu",
                "--out",
                str(out_dir),
                "--runs-dir",
                str(runs_dir),
            ]
        )

    if isinstance(exc_info.value, SystemExit):
        assert exc_info.value.code not in (0, None)

    assert not expected_json.exists()
    assert list(out_dir.glob("*.json")) == []


def test_run_smoke_tk3_multibox_k2_cpu(scratch_dir: Path) -> None:
    """Smoke run (3 epochs) of multibox-k2 on tk3 on CPU writes JSON with all required keys."""
    out_dir = scratch_dir / "raw"
    runs_dir = scratch_dir / "checkpoints"

    rc = run.main(
        [
            "--dataset",
            "tk3",
            "--method",
            "multibox-k2",
            "--dim",
            "2",
            "--seed",
            "0",
            "--epochs",
            "3",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--runs-dir",
            str(runs_dir),
        ]
    )
    assert rc in (0, None)

    json_path = out_dir / "tk3__multibox-k2__d2__s0.json"
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for top_key in ("config", "metrics", "timings", "env"):
        assert top_key in payload, f"Missing top-level key {top_key!r}"

    cfg = payload["config"]
    assert cfg["dataset"] == "tk3"
    assert cfg["method"] == "multibox-k2"
    assert cfg["dim"] == 2
    assert cfg["seed"] == 0
    assert cfg["epochs"] == 3
    assert "effective_batch_size" in cfg
    assert cfg["effective_batch_size"] == cfg["batch_size"]

    m = payload["metrics"]
    for req_metric_key in (
        "modelness_all",
        "modelness_gci0",
        "modelness_gci1",
        "modelness_gci1_bot",
        "helly_violation_rate",
        "degeneracy_rate",
        "n_certificates",
        "auc_pos_neg",
        "pair_fvr",
        "fvr",
        "n_pos",
        "n_und",
        "n_neg",
        "param_count",
        "mean_epoch_time_s",
        "total_train_time_s",
        "n_components",
        "mean_effective_components",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_mb",
    ):
        assert req_metric_key in m, f"Missing metric key {req_metric_key!r}"

    assert m["n_certificates"] == 1
    assert m["n_components"] == 2

    env = payload["env"]
    assert "git_hash_experiments" in env and len(env["git_hash_experiments"]) >= 7
    assert "git_hash_mowl" in env and len(env["git_hash_mowl"]) >= 7
    assert "torch_version" in env
    assert "cuda_version" in env
    assert "gpu_name" in env


def test_sweep_dry_run_lists_all_cells(scratch_dir: Path) -> None:
    """Verify sweep dry-run enumerates 11 methods x datasets x dims x seeds for every suite."""
    n_methods = len(METHODS)
    assert n_methods == 11

    def _expected_count(ds_list: List[str], smoke: bool = False) -> int:
        total = 0
        for d in ds_list:
            n_dims = len(DATASET_DIMS[d])
            n_seeds = 1 if smoke else len(DATASET_SEEDS[d])
            total += n_methods * n_dims * n_seeds
        return total

    cells_syn = sweep.enumerate_cells("synthetic", smoke=False)
    assert len(cells_syn) == _expected_count(SYNTHETIC_DATASETS, smoke=False) == 11 * 4 * 2 * 3

    cells_pizza = sweep.enumerate_cells("pizza", smoke=False)
    assert len(cells_pizza) == _expected_count(["pizza"], smoke=False) == 11 * 1 * 1 * 3

    cells_real = sweep.enumerate_cells("real", smoke=False)
    assert len(cells_real) == _expected_count(["go", "foodon"], smoke=False) == 11 * 2 * 1 * 1

    cells_all = sweep.enumerate_cells("all", smoke=False)
    assert len(cells_all) == _expected_count(DATASETS, smoke=False) == 264 + 33 + 22

    cells_smoke = sweep.enumerate_cells("all", smoke=True)
    assert len(cells_smoke) == _expected_count(DATASETS, smoke=True) == 11 * (4 * 2 + 3 * 1) == 121

    # Also test CLI --dry-run does not execute training or write files
    out_dir = scratch_dir / "dry_out"
    rc = sweep.main(["--suite", "all", "--dry-run", "--out", str(out_dir)])
    assert rc in (0, None)
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []


def test_run_oom_halves_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    scratch_dir: Path,
) -> None:
    """Verify CUDA OOM during training halves effective_batch_size up to 3 times and records it."""
    out_dir = scratch_dir / "oom_raw"
    runs_dir = scratch_dir / "oom_ckpts"

    call_count = {"n": 0}
    orig_train = run._train_model_with_chunking

    def _flaky_train(model, epochs, validate_every, effective_batch_size, epoch_times):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM on first attempt")
        return orig_train(model, epochs, validate_every, effective_batch_size, epoch_times)

    monkeypatch.setattr(run, "_train_model_with_chunking", _flaky_train)

    rc = run.main(
        [
            "--dataset",
            "tk3",
            "--method",
            "box2el",
            "--dim",
            "2",
            "--seed",
            "0",
            "--epochs",
            "2",
            "--batch-size",
            "1024",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--runs-dir",
            str(runs_dir),
        ]
    )
    assert rc in (0, None)
    json_path = out_dir / "tk3__box2el__d2__s0.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"]["batch_size"] == 1024
    assert payload["config"]["effective_batch_size"] == 512
    assert payload["config"]["oom_retries"] == 1


def test_aggregate_generates_tables_and_csv(scratch_dir: Path) -> None:
    """Verify aggregate.py reads raw JSONs and produces dataset tables, multibox_curve.md, and all_runs.csv."""
    raw_dir = scratch_dir / "raw"
    tables_dir = scratch_dir / "tables"
    csv_path = scratch_dir / "all_runs.csv"

    for method in ("box2el", "multibox-k1", "multibox-k2"):
        run.main(
            [
                "--dataset",
                "tk3",
                "--method",
                method,
                "--dim",
                "2",
                "--seed",
                "0",
                "--epochs",
                "2",
                "--device",
                "cpu",
                "--out",
                str(raw_dir),
                "--runs-dir",
                str(scratch_dir / "ckpts"),
            ]
        )

    rc = aggregate.main(
        [
            "--raw",
            str(raw_dir),
            "--out",
            str(tables_dir),
            "--csv",
            str(csv_path),
        ]
    )
    assert rc in (0, None)
    assert (tables_dir / "tk3.md").exists()
    assert (tables_dir / "multibox_curve.md").exists()
    assert csv_path.exists()
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 4  # header + 3 runs
