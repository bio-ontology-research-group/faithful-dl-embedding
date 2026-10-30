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
    OBJECTIVE_METHODS,
    PRISM_METHODS,
    SUBSUMPTION_DATASETS,
    SYNTHETIC_DATASETS,
    methods_for_dataset,
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
    assert isinstance(env.get("git_dirty_experiments"), bool)
    assert isinstance(env.get("git_dirty_mowl"), bool)
    assert "torch_version" in env
    assert "cuda_version" in env
    assert "gpu_name" in env


def test_sweep_dry_run_lists_all_cells(scratch_dir: Path) -> None:
    """Verify sweep dry-run enumerates methods_for_dataset(d) x datasets x dims x seeds for every suite."""
    assert len(METHODS) >= 52
    assert len(methods_for_dataset("tk3")) == 49
    assert len(methods_for_dataset("pizza")) == 51
    assert len(methods_for_dataset("go")) == 12
    assert "box2el-milp" in methods_for_dataset("pizza")
    assert "box2el-milp" not in methods_for_dataset("tk3")
    assert "box2el-milp" not in methods_for_dataset("go")
    assert "canonical-mixture" in methods_for_dataset("yeast-function")
    assert "canonical-mixture" in methods_for_dataset("helis")
    assert "canonical-mixture" not in methods_for_dataset("pizza")
    assert "canonical-mixture" not in methods_for_dataset("tk3")
    assert "elem" in methods_for_dataset("yeast-function")
    assert "elem" in methods_for_dataset("helis")
    assert "elem" in methods_for_dataset("pizza")
    assert "elem" in methods_for_dataset("go")
    assert "elem" in methods_for_dataset("foodon")
    assert "elem" not in methods_for_dataset("tk3")

    obj_prism_set = set(OBJECTIVE_METHODS) | set(PRISM_METHODS)

    def _expected_count(ds_list: List[str], smoke: bool = False) -> int:
        total = 0
        for d in ds_list:
            n_m = len([m for m in methods_for_dataset(d) if m not in obj_prism_set])
            n_dims = len(DATASET_DIMS[d])
            n_seeds = 1 if smoke else len(DATASET_SEEDS[d])
            total += n_m * n_dims * n_seeds
        return total

    cells_syn = sweep.enumerate_cells("synthetic", smoke=False)
    assert len(cells_syn) == _expected_count(SYNTHETIC_DATASETS, smoke=False) == 11 * 4 * 2 * 3
    assert all(c.method not in obj_prism_set for c in cells_syn)

    cells_pizza = sweep.enumerate_cells("pizza", smoke=False)
    assert len(cells_pizza) == _expected_count(["pizza"], smoke=False) == 13 * 1 * 1 * 3
    assert all(c.method not in obj_prism_set for c in cells_pizza)

    cells_real = sweep.enumerate_cells("real", smoke=False)
    assert len(cells_real) == _expected_count(["go", "foodon"], smoke=False) == 12 * 2 * 1 * 3

    cells_all = sweep.enumerate_cells("all", smoke=False)
    assert len(cells_all) == _expected_count(DATASETS, smoke=False)
    assert all(c.method not in obj_prism_set for c in cells_all)

    cells_smoke = sweep.enumerate_cells("all", smoke=True)
    assert len(cells_smoke) == _expected_count(DATASETS, smoke=True)

    cells_obj = sweep.enumerate_cells("objectives", smoke=False)
    assert len(cells_obj) == 38 * 4 * 2 * 3 * 2
    assert {c.method for c in cells_obj} == obj_prism_set

    # Disallowed --methods override for a dataset raises a clear ValueError
    with pytest.raises(ValueError, match="not allowed on dataset 'go'"):
        sweep.enumerate_cells("real", datasets=["go"], methods=["prism"])
    with pytest.raises(ValueError, match="not allowed on dataset 'tk3'"):
        sweep.enumerate_cells("synthetic", datasets=["tk3"], methods=["box2el-milp"])

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

    # Verify retry limit is 3 (4th consecutive OOM raises OutOfMemoryError)
    fail_count = {"n": 0}

    def _always_oom(model, epochs, validate_every, effective_batch_size, epoch_times):
        fail_count["n"] += 1
        raise torch.cuda.OutOfMemoryError("Persistent CUDA OOM")

    monkeypatch.setattr(run, "_train_model_with_chunking", _always_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run.main(
            [
                "--dataset",
                "tk3",
                "--method",
                "box2el",
                "--dim",
                "2",
                "--seed",
                "1",
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
    assert fail_count["n"] == 4  # initial attempt + 3 retries


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


def test_aggregate_null_nan_handling_and_certs_column() -> None:
    """Fix 4 & 5: Verify aggregate.py reports k/n, renders all-null as '—', uses ddof=1, and includes Certs in subsumption tables."""
    # All null / NaN -> "—"
    assert aggregate._format_mean_sd([None, float("nan"), None], decimals=4) == "—"

    # Partial non-null (2 of 3) with ddof=1: mean([0.2, 0.4]) = 0.3000, std([0.2, 0.4], ddof=1) = sqrt(0.02) = 0.1414
    assert aggregate._format_mean_sd([0.2, 0.4, None], decimals=4) == "0.3000 ± 0.1414 (2/3)"

    # Single non-null out of 3 -> "0.5000 (1/3)"
    assert aggregate._format_mean_sd([0.5, None, float("nan")], decimals=4) == "0.5000 (1/3)"

    # Full non-null (3 of 3) with ddof=1: mean([1.0, 2.0, 3.0]) = 2.0000, std([1.0, 2.0, 3.0], ddof=1) = 1.0000
    assert aggregate._format_mean_sd([1.0, 2.0, 3.0], decimals=4) == "2.0000 ± 1.0000"

    # Subsumption table rendering with n_certificates == 0 (null HVR/DegRate) and partial NaN
    pizza_runs = [
        {
            "config": {
                "dataset": "pizza",
                "method": "box2el",
                "dim": 50,
                "seed": s,
                "n_components": 1,
                "init": None,
                "anchor_weight": 0.0,
            },
            "metrics": {
                "rank_f_mrr": 0.20 if s == 0 else (0.40 if s == 1 else float("nan")),
                "rank_f_mr": 10.0,
                "rank_f_hits@1": 0.1,
                "rank_f_hits@3": 0.2,
                "rank_f_hits@10": 0.5,
                "rank_f_auc": 0.9,
                "rank_mrr": 0.25,
                "rank_mr": 12.0,
                "auc_pos_neg": 0.8,
                "auc_pos_und": 0.7,
                "fvr": 0.05,
                "pair_fvr": 0.02,
                "modelness_all": 0.01,
                "helly_violation_rate": None,
                "degeneracy_rate": None,
                "n_certificates": 0,
                "mean_effective_components": 1.0,
                "mean_epoch_time_s": 0.05,
                "total_train_time_s": 1.0,
                "peak_gpu_memory_mb": 7.0,
                "param_count": 1000,
            },
        }
        for s in (0, 1, 2)
    ]
    md = aggregate.render_dataset_markdown("pizza", pizza_runs)
    assert "| Certs |" in md
    assert "| `box2el` | 1 | random | 0 | 2/3 |" in md
    assert "0.3000 ± 0.1414 (2/3)" in md
    # HVR and DegRate are both "—" followed by Certs "0 ± 0"
    assert "| — | — | 0 ± 0 |" in md


def test_run_abox_reloads_best_checkpoint_and_stores_single_curve(
    monkeypatch: pytest.MonkeyPatch,
    scratch_dir: Path,
) -> None:
    """Verify execute_run on an ABox bundle reloads the intermediate peak validation-MRR checkpoint and stores val_mrr_curve once."""
    import numpy as np
    import experiments.abox_eval as abox_eval_mod
    from experiments.datasets import DatasetBundle

    good_scores = torch.tensor([[0.1, 0.4, 0.7, 0.9]], dtype=torch.float32)
    bad_scores = torch.tensor([[0.9, 0.1, 0.2, 0.3]], dtype=torch.float32)

    class _ScriptedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scores_param = torch.nn.Parameter(bad_scores.clone())
            self.margin = 0.02
            self.gamma = 0.02
            self.n_components = 1

        def effective_components(self) -> torch.Tensor:
            return torch.ones(4, dtype=torch.float32)

        def forward(self, data: torch.Tensor, gci_name: str, neg: bool = False) -> torch.Tensor:
            if gci_name == "class_assertion":
                return self.scores_param[data[:, 0].long(), data[:, 1].long()]
            raise NotImplementedError(gci_name)

    class _ScriptedModel:
        def __init__(self) -> None:
            self.module = _ScriptedModule()
            self.individual_index_dict = {"a0": 0}
            self.class_index_dict = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
            self.training_datasets = {
                "class_assertion": torch.tensor([[0, 0]], dtype=torch.long)
            }
            self.validation_datasets = {}
            self.testing_datasets = {}
            self.device = "cpu"
            self.margin = 0.02

        def train(self, epochs: int, validate_every: int, epoch_callback=None) -> None:
            for ep in range(epochs):
                ep_num = ep + 1
                with torch.no_grad():
                    if ep_num == 50:
                        self.module.scores_param.copy_(good_scores)
                    elif ep_num == 51:
                        self.module.scores_param.copy_(bad_scores)
                if epoch_callback is not None:
                    epoch_callback(ep, self)

    fake_split = {
        "individuals": ["a0"],
        "candidates": ["C0", "C1", "C2", "C3"],
        "cand_to_idx": {"C0": 0, "C1": 1, "C2": 2, "C3": 3},
        "positives": [(0, 0)],
        "true_sets": [["C0"]],
        "skip_mask": np.zeros((1, 4), dtype=bool),
        "label_codes": np.zeros((1, 4), dtype=np.int8),
        "label_gold_codes": None,
    }
    fake_bundle = DatasetBundle(
        name="helis",
        task="abox",
        dataset=object(),
        deductive_closure_ontology=None,
        paths={"ontology": str(scratch_dir / "dummy.owl")},
    )
    setattr(fake_bundle, "_abox_split_cache_valid", fake_split)
    setattr(fake_bundle, "_abox_split_cache_test", fake_split)

    monkeypatch.setattr(
        abox_eval_mod,
        "get_candidate_hierarchy",
        lambda *_a, **_kw: {"C0": ["C0"], "C1": ["C1"], "C2": ["C2"], "C3": ["C3"]},
    )
    monkeypatch.setattr(
        run,
        "_build_model",
        lambda **_kw: (_ScriptedModel(), None),
    )

    payload = run.execute_run(
        dataset="helis",
        method="elbe",
        dim=8,
        seed=0,
        epochs=100,
        device="cpu",
        out_dir=scratch_dir / "out",
        runs_dir=scratch_dir / "ckpts",
        bundle=fake_bundle,
    )

    assert set(payload.keys()) == {"config", "metrics", "timings", "env"}
    assert "validation_curve" not in payload
    assert "validation_curve" not in payload["config"]
    assert payload["metrics"]["best_val_epoch"] == 50
    assert payload["metrics"]["best_val_mrr"] == pytest.approx(1.0)
    assert payload["metrics"]["rank_mrr"] == pytest.approx(1.0)


def test_run_rejects_point_objective_for_plain_backbone(
    monkeypatch: pytest.MonkeyPatch,
    scratch_dir: Path,
) -> None:
    """Verify run.py rejects objective != 'axiom' for plain backbones without point losses."""
    for plain_method, model_cls in (
        ("elbe", "ELBE"),
        ("box2el", "BoxSquaredEL"),
        ("elem", "ELEm"),
        ("prism", "PrismEL"),
        ("canonical-mixture", "CanonicalMixture"),
    ):
        with pytest.raises(ValueError, match="objective"):
            run.execute_run(
                dataset="tk3" if plain_method != "canonical-mixture" else "yeast-function",
                method=plain_method,
                dim=2,
                seed=0,
                epochs=1,
                device="cpu",
                out_dir=scratch_dir / "out",
                runs_dir=scratch_dir / "ckpts",
                objective="point",
            )

    # Also verify ELEm rejects *-init / *-anchor variants
    for bad_spec in (
        {"model": "ELEm", "n_components": 1, "init": "projected", "anchor_weight": 0.0},
        {"model": "ELEm", "n_components": 1, "init": None, "anchor_weight": 0.1},
    ):
        monkeypatch.setitem(METHODS, "elem-init", bad_spec)
        with pytest.raises(ValueError, match="ELEm"):
            run.execute_run(
                dataset="tk3",
                method="elem-init",
                dim=2,
                seed=0,
                epochs=1,
                device="cpu",
                out_dir=scratch_dir / "out",
                runs_dir=scratch_dir / "ckpts",
            )

    # Also test when a method spec in METHODS has a plain backbone with objective != "axiom"
    for plain_cls in ("ELBE", "BoxSquaredEL", "ELEm", "PrismEL", "CanonicalMixture"):
        fake_id = f"bad-{plain_cls.lower()}-point"
        monkeypatch.setitem(
            METHODS,
            fake_id,
            {
                "model": plain_cls,
                "n_components": 1,
                "init": None,
                "anchor_weight": 0.0,
                "objective": "point",
                "objective_kwargs": {"objective": "point"},
            },
        )
        with pytest.raises(ValueError, match="objective"):
            run.execute_run(
                dataset="tk3",
                method=fake_id,
                dim=2,
                seed=0,
                epochs=1,
                device="cpu",
                out_dir=scratch_dir / "out",
                runs_dir=scratch_dir / "ckpts",
            )


def test_run_lr_filename_and_objectives_table(scratch_dir: Path) -> None:
    """Verify --lr records lr in JSON and filename when lr != 0.001, and aggregate writes objectives_<dataset>.md."""
    raw_dir = scratch_dir / "raw"
    ckpts_dir = scratch_dir / "ckpts"
    tables_dir = scratch_dir / "tables"
    csv_path = scratch_dir / "all_runs.csv"

    assert run.run_filename("tk3", "multibox-k2-point", 2, 0) == "tk3__multibox-k2-point__d2__s0.json"
    assert run.run_filename("tk3", "multibox-k2-point", 2, 0, lr=0.001) == "tk3__multibox-k2-point__d2__s0.json"
    assert run.run_filename("tk3", "multibox-k2-point", 2, 0, lr=0.01) == "tk3__multibox-k2-point__d2__s0__lr0.01.json"
    assert (
        run.run_filename("tk3", "multibox-k2", 2, 0, objective="point")
        == "tk3__multibox-k2__d2__s0__obj-point.json"
    )
    assert (
        run.run_filename("tk3", "multibox-k2", 2, 0, lr=0.01, objective="mixed")
        == "tk3__multibox-k2__d2__s0__lr0.01__obj-mixed.json"
    )

    for method, lr, obj_override in (
        ("multibox-k2-point", "0.001", None),
        ("multibox-k2-point", "0.01", None),
        ("multibox-k2-hybrid", "0.001", None),
        ("multibox-k2-softdiv", "0.001", None),
        ("prism", "0.001", None),
        ("multibox-k2", "0.001", "point"),
    ):
        argv = [
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
            "--lr",
            lr,
            "--device",
            "cpu",
            "--out",
            str(raw_dir),
            "--runs-dir",
            str(ckpts_dir),
        ]
        if obj_override is not None:
            argv.extend(["--objective", obj_override])
        rc = run.main(argv)
        assert rc in (0, None)

    p_default = raw_dir / "tk3__multibox-k2-point__d2__s0.json"
    p_lr01 = raw_dir / "tk3__multibox-k2-point__d2__s0__lr0.01.json"
    p_hybrid = raw_dir / "tk3__multibox-k2-hybrid__d2__s0.json"
    p_softdiv = raw_dir / "tk3__multibox-k2-softdiv__d2__s0.json"
    p_prism = raw_dir / "tk3__prism__d2__s0.json"
    p_obj_override = raw_dir / "tk3__multibox-k2__d2__s0__obj-point.json"
    assert p_default.exists()
    assert p_lr01.exists()
    assert p_hybrid.exists()
    assert p_softdiv.exists()
    assert p_prism.exists()
    assert p_obj_override.exists()

    j_hybrid = json.loads(p_hybrid.read_text(encoding="utf-8"))
    assert j_hybrid["config"]["objective"] == "hybrid"
    pm_hyb = j_hybrid["metrics"]["point_modelness"]
    assert isinstance(pm_hyb, dict)
    assert "pos_frac" in pm_hyb and "hardneg_frac" in pm_hyb

    j_point = json.loads(p_lr01.read_text(encoding="utf-8"))
    assert j_point["config"]["lr"] == pytest.approx(0.01)
    assert j_point["config"]["dim"] == 2
    assert j_point["config"]["effective_dim"] == 2
    assert j_point["config"]["objective"] == "point"
    assert j_point["config"]["tau_start"] == pytest.approx(1.0)
    assert j_point["config"]["tau_end"] == pytest.approx(0.05)
    assert j_point["config"]["tau_anneal"] is False
    assert j_point["config"]["diversity_reg"] == pytest.approx(0.0)
    pm = j_point["metrics"]["point_modelness"]
    assert isinstance(pm, dict)
    assert "pos_frac" in pm and "hardneg_frac" in pm

    j_softdiv = json.loads(p_softdiv.read_text(encoding="utf-8"))
    assert j_softdiv["config"]["component_aggregation"] == "soft"
    assert j_softdiv["config"]["tau_anneal"] is True
    assert j_softdiv["config"]["tau_schedule"] == {
        "tau_start": 1.0,
        "tau_end": 0.05,
        "tau_anneal": True,
    }
    assert j_softdiv["config"]["diversity_reg"] == pytest.approx(0.1)

    j_prism = json.loads(p_prism.read_text(encoding="utf-8"))
    assert j_prism["config"]["dim"] == 2
    assert j_prism["config"]["effective_dim"] == 3
    assert j_prism["config"]["tau_start"] is None
    assert j_prism["config"]["tau_end"] is None
    assert j_prism["config"]["tau_anneal"] is None
    assert j_prism["config"]["tau_schedule"] is None
    assert j_prism["metrics"]["point_modelness"] is None

    # Also test older prism JSON without effective_dim key -> aggregate infers effective_dim=3
    legacy_prism = json.loads(p_prism.read_text(encoding="utf-8"))
    del legacy_prism["config"]["effective_dim"]
    assert aggregate._infer_effective_dim(legacy_prism["config"]) == 3

    rc_agg = aggregate.main(
        [
            "--raw",
            str(raw_dir),
            "--out",
            str(tables_dir),
            "--csv",
            str(csv_path),
        ]
    )
    assert rc_agg in (0, None)
    obj_table = tables_dir / "objectives_tk3.md"
    assert obj_table.exists()
    text = obj_table.read_text(encoding="utf-8")
    for col in (
        "method",
        "suite",
        "backbone",
        "k",
        "objective",
        "lr",
        "d",
        "Mod strict",
        "Mod@γ",
        "point-Mod (pos/hardneg)",
        "HVR",
        "Deg",
        "eff-k",
        "seeds n/k",
    ):
        assert col in text

    # Verify lr 0.001 and 0.01 produce separate rows for multibox-k2-point at d=2
    mb_point_rows = [
        line for line in text.splitlines() if line.startswith("| `multibox-k2-point` |")
    ]
    assert len(mb_point_rows) == 2
    assert any("| `multibox-k2-point` | objectives | multibox | 2 | point | 0.001 | 2 |" in r for r in mb_point_rows)
    assert any("| `multibox-k2-point` | objectives | multibox | 2 | point | 0.01 | 2 |" in r for r in mb_point_rows)

    # Verify prism trained at nominal d=2 appears under d=3 with the footnote
    prism_rows = [line for line in text.splitlines() if line.startswith("| `prism` |")]
    assert len(prism_rows) == 1
    assert "| `prism` | objectives | prism | 1 | axiom | 0.001 | 3 |" in prism_rows[0]
    assert "prism needs ≥ 3 dims; nominal d=2 cells trained at d=3" in text

    # Verify baseline method (multibox-k2) has suite 'baseline'
    baseline_rows = [line for line in text.splitlines() if line.startswith("| `multibox-k2` |")]
    assert len(baseline_rows) == 1
    assert "| `multibox-k2` | baseline |" in baseline_rows[0]

    # Verify hybrid row shows point-Mod (pos/hardneg) and not '—'
    hybrid_rows = [
        line for line in text.splitlines() if line.startswith("| `multibox-k2-hybrid` |")
    ]
    assert len(hybrid_rows) == 1
    hybrid_cells = [c.strip() for c in hybrid_rows[0].strip("|").split("|")]
    # Column index 9 is 'point-Mod (pos/hardneg)'
    assert hybrid_cells[9] != "—" and "/" in hybrid_cells[9]


def test_run_smoke_tk3_elem_cpu(scratch_dir: Path) -> None:
    """Verify elem method builds ELEmbeddings and runs cleanly through execute_run."""
    out_dir = scratch_dir / "raw"
    runs_dir = scratch_dir / "checkpoints"

    rc = run.main(
        [
            "--dataset",
            "tk3",
            "--method",
            "elem",
            "--dim",
            "4",
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
    assert rc in (0, None)
    json_path = out_dir / "tk3__elem__d4__s0.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"]["method"] == "elem"
    assert payload["config"]["model"] == "ELEm"
    assert payload["config"]["n_components"] == 1
    assert payload["config"]["init"] is None
    assert payload["metrics"]["point_modelness"] is None
    assert payload["metrics"]["helly_violation_rate"] is None
    assert 0.0 <= payload["metrics"]["modelness_all"] <= 1.0



