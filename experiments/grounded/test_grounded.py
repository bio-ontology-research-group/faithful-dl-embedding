"""Fast unit and smoke tests for ``experiments.grounded`` (design §9 E1 & E2)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
import torch as th

os.environ.setdefault(
    "_JAVA_OPTIONS",
    "-Xms64m -XX:CompressedClassSpaceSize=64m -XX:ReservedCodeCacheSize=48m -XX:ParallelGCThreads=4 -XX:ConcGCThreads=2",
)
import mowl

mowl.init_jvm(os.environ.get("MOWL_JVM_MEMORY", "1536m"))

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.models import GroundedEL
from mowl.nn.el.grounded.losses import crisp_membership

from experiments.grounded.aggregate import (
    load_grounded_runs,
    render_grounded_markdown,
    summarize_cell_group,
    write_grounded_markdown,
)
from experiments.grounded.run_go import (
    _batched_crisp_elem_membership,
    compute_crisp_and_eval_metrics,
    evaluate_subsumption_ranking,
)
from experiments.grounded.run_synthetic import (
    parse_capacity_arg,
    run_synthetic_cell,
)
from experiments.grounded.theories import (
    SYNTHETIC_NS,
    THEORIES,
    build_theory,
    generate_all_theories,
)


def test_all_theories_build_and_satisfiability(tmp_path: Path) -> None:
    """All 11 synthetic theories build valid OWL files with expected satisfiability."""
    expected = {
        "tk3",
        "tk4",
        "tk5",
        "tk6",
        "tk7",
        "box2el_t1",
        "box2el_c4",
        "role_functional",
        "role_cycle",
        "role_chain",
        "unsat_mix",
    }
    assert set(THEORIES) == expected

    paths = generate_all_theories(out_dir=tmp_path)
    assert set(paths.keys()) == expected

    for name in ("box2el_t1", "box2el_c4", "role_functional", "role_cycle", "role_chain"):
        ds = PathDataset(str(paths[name]))
        cm = CanonicalModel.from_ontology(ds.ontology)
        named_iris = [
            c
            for c in ds.classes.as_str
            if not c.endswith("Thing") and not c.endswith("Nothing")
        ]
        assert len(named_iris) > 0
        for iri in named_iris:
            assert cm.is_satisfiable(iri), f"{name}: {iri} should be satisfiable"

    # Check role_chain specifically fires both r o s <= t (A <= D) and s o r <= u (F <= E)
    cm_rc = CanonicalModel.from_ontology(PathDataset(str(paths["role_chain"])).ontology)
    assert cm_rc.extension(f"{SYNTHETIC_NS}A").issubset(cm_rc.extension(f"{SYNTHETIC_NS}D"))
    assert cm_rc.extension(f"{SYNTHETIC_NS}F").issubset(cm_rc.extension(f"{SYNTHETIC_NS}E"))
    assert not cm_rc.extension(f"{SYNTHETIC_NS}A").issubset(cm_rc.extension(f"{SYNTHETIC_NS}E"))
    assert not cm_rc.extension(f"{SYNTHETIC_NS}F").issubset(cm_rc.extension(f"{SYNTHETIC_NS}D"))

    # Check unsat_mix has Q unsatisfiable and all other named classes satisfiable
    ds_um = PathDataset(str(paths["unsat_mix"]))
    cm_um = CanonicalModel.from_ontology(ds_um.ontology)
    assert not cm_um.is_satisfiable(f"{SYNTHETIC_NS}Q")
    for iri in ds_um.classes.as_str:
        if iri.endswith("Q") or iri.endswith("Nothing") or iri.endswith("Thing"):
            continue
        assert cm_um.is_satisfiable(iri), f"unsat_mix: {iri} should be satisfiable"


def test_parse_k_and_run_synthetic_smoke(tmp_path: Path) -> None:
    """run_synthetic handles numeric K and K='full' and writes valid JSON."""
    assert parse_capacity_arg("4") == 4
    assert parse_capacity_arg("full") == "full"
    with pytest.raises(ValueError):
        parse_capacity_arg("0")
    with pytest.raises(ValueError):
        parse_capacity_arg("invalid")

    res = run_synthetic_cell(
        theory="role_functional",
        K="full",
        Kr=None,
        dim=2,
        seed=0,
        epochs=5,
        lr=0.02,
        out=tmp_path / "runs",
        theory_dir=tmp_path / "owl",
        n_eval=128,
    )
    assert res["config"]["n_components"] == res["config"]["nb_elements"]
    assert res["config"]["n_role_components"] == res["config"]["nb_elements"]
    assert "elem_viol" in res["crisp_report"]
    assert "region_viol" in res["crisp_report"]
    assert "exact" in res["metrics"]
    assert len(res["loss_curve"]) >= 1

    out_json = tmp_path / "runs" / "role_functional__Kfull__d2__s0.json"
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["config"]["theory"] == "role_functional"


def test_aggregate_report_generation(tmp_path: Path) -> None:
    """aggregate.py aggregates JSON cells and writes markdown tables."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    sample = {
        "config": {
            "theory": "tk3",
            "K": "4",
            "Kr": "4",
            "n_components": 4,
            "n_role_components": 4,
            "dim": 2,
            "seed": 0,
            "epochs": 100,
            "lr": 0.01,
            "nb_classes": 12,
            "nb_elements": 12,
            "nb_rels": 1,
        },
        "crisp_report": {"elem_viol": 0.0, "region_viol": 0.0},
        "metrics": {
            "elem_viol": 0.0,
            "region_viol": 0.0,
            "exist": 1.0,
            "dead_ok": True,
            "faithful_named_precision": 1.0,
            "faithful_named_recall": 1.0,
            "faithful_named": [1.0, 1.0],
            "exact": True,
        },
        "final_losses": {"total": 0.001},
        "loss_curve": [{"epoch": 1, "total": 1.0}, {"epoch": 100, "total": 0.001}],
        "wall_time_s": 1.25,
    }
    (runs_dir / "tk3__K4__d2__s0.json").write_text(
        json.dumps(sample), encoding="utf-8"
    )

    out_md = tmp_path / "GROUNDED.md"
    written = write_grounded_markdown(in_dir=runs_dir, out_path=out_md)
    assert written == out_md.resolve()
    runs = load_grounded_runs(in_dir=runs_dir)
    assert len(runs) == 1
    summary = summarize_cell_group(runs)
    assert summary["exact_rate"] == 1.0
    text = out_md.read_text(encoding="utf-8")
    assert "`tk3`" in text
    assert "1.00 (1/1)" in text


def test_go_batched_helpers_and_sbatch(tmp_path: Path) -> None:
    """Batched GO crisp membership and ranking helpers match GroundedELModule."""
    owl_path = build_theory("box2el_t1", out_dir=tmp_path / "owl")
    ds = PathDataset(str(owl_path), validation_path=str(owl_path), testing_path=str(owl_path))
    model = GroundedEL(
        ds,
        embed_dim=4,
        n_components=2,
        n_role_components=2,
        epochs=3,
        learning_rate=0.02,
        device="cpu",
        seed=0,
    )
    model.train(epochs=3)

    mod = model.module
    with th.no_grad():
        pts = mod.positions()
        batched_m = _batched_crisp_elem_membership(mod, pts, class_chunk_size=2)
        ref_m = th.stack(
            [
                crisp_membership(mod.depth(pts, c_id, crisp=True))
                for c_id in range(model.nb_classes)
            ],
            dim=1,
        )
        assert th.equal(batched_m, ref_m)

    crisp_direct, eval_direct = compute_crisp_and_eval_metrics(model, n_eval=64)
    crisp_batched, eval_batched = compute_crisp_and_eval_metrics(
        model, n_eval=64, force_batched=True
    )
    assert "elem_viol" in crisp_direct
    assert "exist" in eval_direct
    assert pytest.approx(crisp_batched["elem_viol"]) == crisp_direct["elem_viol"]
    assert pytest.approx(eval_batched["exist"]) == eval_direct["exist"]

    ranking = evaluate_subsumption_ranking(model, ds, eval_batch_size=4)
    assert "mrr" in ranking
    assert "hits@1" in ranking
    assert "hits@10" in ranking
    assert "hits@100" in ranking
    assert "auc" in ranking

    sbatch_path = Path(__file__).resolve().parent / "go.sbatch"
    sbatch_text = sbatch_path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=pi-hohndor" in sbatch_text
    assert "#SBATCH --gres=gpu:1" in sbatch_text
    assert "#SBATCH --cpus-per-task=8" in sbatch_text
    assert "#SBATCH --mem=64G" in sbatch_text
    assert "#SBATCH --time=12:00:00" in sbatch_text
    assert "/mnt/data/u/hohndor/gfe/venv/bin/python" in sbatch_text
