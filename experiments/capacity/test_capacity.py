"""Fast tests for the capacity experiments (``experiments/capacity``).

Covers: B1 exact agreement on tk3/tk4/box2el_t1/unsat_mix; B2 on a hand-built DAG whose
exact model needs 3 intervals for one class (K=3 lossless, K=1 lossy but sound with
recall 1); crisp box/ball containment precision/recall on hand-set regions; the
cells-file parser; the DFS order reimplementation.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import sparse

os.environ.setdefault(
    "_JAVA_OPTIONS",
    "-Xms64m -XX:CompressedClassSpaceSize=64m -XX:ReservedCodeCacheSize=48m "
    "-XX:ParallelGCThreads=4 -XX:ConcGCThreads=2",
)

from experiments.capacity.common import (  # noqa: E402
    Reference,
    bool_csr,
    pair_metrics,
    runs_summary,
    tables_from_arrays,
)


# ----------------------------------------------------------------------------- B2 DAG


def _dag_tables():
    """R1=0, R2=1, R3=2, X=3, A1=4, A2=5, A3=6; A_i ⊑ R_i, A_i ⊑ X.

    DFS (parent of A_i is R_i, the smaller superclass) visits R1 A1 R2 A2 R3 A3 X, so X
    occupies positions {1, 3, 5, 6}: three runs.
    """
    gci0 = [(4, 0), (5, 1), (6, 2), (4, 3), (5, 3), (6, 3)]
    return tables_from_arrays(7, gci0=gci0)


def _exact_reference(kb, lm):
    named = kb.named_ids
    m = lm.subsumption_matrix(named, empty_subsumed_by_all=False).tocoo()
    keep = m.row != m.col
    mat = bool_csr(m.row[keep], m.col[keep], (len(named), len(named)))
    sat = ~np.isin(named, lm.unsatisfiable_classes())
    return Reference([kb.class_iris[i] for i in named], sat, mat, "test", 0.0)


def test_dag_exact_needs_three_intervals():
    from mowl.geometric.least_model import interval_realisation

    kb = _dag_tables()
    lm = kb.least_model()
    assert all(v == 0 for v in kb.check(lm).values())
    real = interval_realisation(lm, "dfs")
    assert real.runs_per_class[3] == 3
    assert real.runs_per_class.max() == 3


def test_budget_lossless_and_lossy():
    from experiments.capacity.budget import run_budget

    kb = _dag_tables()
    ref = _exact_reference(kb, kb.least_model())
    assert ref.n_entailed == 6
    res3 = run_budget("dag", budgets=[3], kb=kb, ref=ref, write=False)[0]
    assert res3["lossless"] and res3["iterations"] == 0
    assert res3["check_model_all_zero"]
    assert res3["precision"] == 1.0 and res3["recall"] == 1.0
    assert res3["label_inflation"] == 1.0

    res1 = run_budget("dag", budgets=[1], kb=kb, ref=ref, write=False)[0]
    assert res1["status"] == "ok"
    assert res1["iterations"] >= 1
    assert res1["max_runs"] <= 1
    assert res1["check_model_all_zero"]
    assert res1["recall"] == 1.0
    assert res1["precision"] < 1.0
    # filling X's two one-block gaps adds R2 ⊑ X and R3 ⊑ X: 6 / 8
    assert res1["n_held_pairs"] == 8 and res1["tp"] == 6
    assert res1["precision"] == pytest.approx(0.75)
    assert res1["precision_live"] == pytest.approx(0.75) and res1["n_named_live"] == 7
    assert res1["label_inflation"] > 1.0
    assert res1["n_named_made_unsat"] == 0


def test_budget_stops_at_first_lossless_budget():
    from experiments.capacity.budget import run_budget

    kb = _dag_tables()
    ref = _exact_reference(kb, kb.least_model())
    out = run_budget("dag", budgets=[1, 2, 3, 4, 6], kb=kb, ref=ref, write=False)
    assert [r["K"] for r in out] == [1, 2, 3]
    assert all(r["check_model_all_zero"] and r["recall"] == 1.0 for r in out)
    desc = run_budget("dag", budgets=[6, 4, 3, 2, 1], kb=kb, ref=ref, write=False)
    assert [r["K"] for r in desc] == [3, 2, 1]
    assert [r["precision"] for r in desc] == [r["precision"] for r in out][::-1]


def test_budget_with_bottom_kills_blocks_soundly():
    """Filling a gap may derive bottom: the re-closure kills the block, still a model."""
    from experiments.capacity.budget import run_budget

    # same DAG plus classes 7 (bottom) and X ⊓ R2 ⊑ ⊥ is NOT present; instead make
    # R2 ⊓ X ⊑ ⊥ so that filling X into R2's block kills it.
    gci0 = [(4, 0), (5, 1), (6, 2), (4, 3), (5, 3), (6, 3)]
    gci1_bot = [(1, 3, 7)]
    kb = tables_from_arrays(8, gci0=gci0, gci1_bot=gci1_bot, bot_id=7)
    lm = kb.least_model()
    # A2 ⊑ R2 ⊓ X ⊑ ⊥: A2 is unsatisfiable already in the exact model
    assert 5 in set(lm.unsatisfiable_classes().tolist())
    ref = _exact_reference(kb, lm)
    out = run_budget("dagbot", budgets=[1, 2], kb=kb, ref=ref, write=False)
    for r in out:
        assert r["check_model_all_zero"]
        assert r["max_runs"] <= r["K"]
        assert r["recall"] == 1.0


def test_dfs_order_matches_mowl():
    from mowl.geometric.least_model import _dfs_order

    from experiments.capacity.budget import dfs_order

    kb = _dag_tables()
    lm = kb.least_model()
    assert np.array_equal(dfs_order(lm), _dfs_order(lm))
    o = dfs_order(lm, np.random.default_rng(0))
    assert np.array_equal(np.sort(o), np.flatnonzero(~lm.dead))


def test_best_of_n_order_records_choice():
    from experiments.capacity.budget import choose_order

    kb = _dag_tables()
    lm = kb.least_model()
    o, info = choose_order(lm, "best-of-4", seed=0)
    assert info["order"] == "best-of-4" and len(info["candidate_total_runs"]) == 4
    assert min(info["candidate_total_runs"]) == info["candidate_total_runs"][info["chosen_candidate"]]
    assert np.array_equal(np.sort(o), np.flatnonzero(~lm.dead))


# ----------------------------------------------------------------------------- metrics


def test_pair_metrics_with_unsat_rows():
    ref = bool_csr(np.array([0, 1]), np.array([1, 2]), (3, 3))
    pred = bool_csr(np.array([0, 2]), np.array([1, 1]), (3, 3))
    m = pair_metrics(pred, ref)
    assert (m["tp"], m["n_pred"], m["n_ref"]) == (1, 2, 2)
    # row 1 unsatisfiable: subsumed by 0 and 2 -> tp gains (1,2)
    m = pair_metrics(pred, ref, unsat_pred=np.array([False, True, False]))
    assert (m["tp"], m["n_pred"]) == (2, 4)
    assert m["recall"] == 1.0 and m["precision"] == 0.5


def test_runs_summary():
    s = runs_summary(np.array([0, 1, 1, 3, 20]))
    assert s["n"] == 4 and s["max"] == 20 and s["share_one"] == 0.5
    assert s["hist"] == {"1": 2, "3": 1, "17-32": 1}


def test_crisp_box_containment():
    import torch

    from experiments.capacity.baselines import crisp_subsumption_metrics

    lo = torch.tensor([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [3.0, 0.0]])
    hi = torch.tensor([[10.0, 10.0], [2.0, 2.0], [6.0, 6.0], [3.0, 1.0]])  # class 3 degenerate
    ref = bool_csr(np.array([1, 2, 3, 1]), np.array([0, 0, 0, 2]), (4, 4))
    m = crisp_subsumption_metrics("box", lo, hi, np.arange(4), ref, deg_eps=1e-3, chunk=2)
    # held: (1,0), (2,0) among regular boxes; degenerate 3 below all of 0, 1, 2
    assert m["n_pred"] == 5 and m["n_ref"] == 4 and m["tp"] == 3
    assert m["precision"] == pytest.approx(0.6) and m["recall"] == pytest.approx(0.75)
    assert m["n_degenerate"] == 1
    # live convention: only classes 0, 1, 2; reference pair (1, 2) is missed
    assert (m["n_live"], m["n_pred_live"], m["n_ref_live"], m["tp_live"]) == (3, 2, 3, 2)
    assert m["precision_live"] == 1.0 and m["recall_live"] == pytest.approx(2 / 3)
    # dense-to-sparse switch gives the same counts
    m2 = crisp_subsumption_metrics("box", lo, hi, np.arange(4), ref, dense_until=1.1)
    assert (m2["n_pred"], m2["tp"]) == (m["n_pred"], m["tp"])
    # a subset of classes is compared on its own
    sub = crisp_subsumption_metrics("box", lo, hi, np.array([0, 1]),
                                    bool_csr(np.array([1]), np.array([0]), (2, 2)))
    assert (sub["n_pred"], sub["tp"], sub["precision"], sub["recall"]) == (1, 1, 1.0, 1.0)


def test_crisp_ball_containment():
    import torch

    from experiments.capacity.baselines import crisp_subsumption_metrics

    c = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    r = torch.tensor([5.0, 1.0, 1.0])
    ref = bool_csr(np.array([1, 2]), np.array([0, 0]), (3, 3))
    m = crisp_subsumption_metrics("ball", c, r, np.arange(3), ref)
    assert (m["n_pred"], m["tp"]) == (1, 1)
    assert m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["precision_live"] == 1.0 and m["recall_live"] == 0.5


def test_crisp_ball_counts_match_brute_force_on_many_balls():
    """More than 25 balls (cdist would switch to its inexact shortcut), incl. duplicates."""
    import torch

    from experiments.capacity.baselines import crisp_subsumption_metrics

    g = torch.Generator().manual_seed(0)
    c = torch.randn(60, 3, generator=g) * 100.0
    r = torch.rand(60, generator=g) * 150.0
    c[40:] = c[:20]  # same centres: containment decided by the radius alone
    r[50:] = r[10:20]  # exact duplicates contain each other
    n = 60
    brute = [(i, j) for i in range(n) for j in range(n)
             if i != j and float(torch.linalg.norm(c[i] - c[j]) + r[i]) <= float(r[j])]
    ref = bool_csr(np.array([i for i, _ in brute]), np.array([j for _, j in brute]), (n, n))
    m = crisp_subsumption_metrics("ball", c, r, np.arange(n), ref, chunk=16)
    assert len(brute) > 20
    assert (m["n_pred"], m["tp"], m["n_pred_live"]) == (len(brute), len(brute), len(brute))


def test_parameter_counts_and_regions_on_hand_set_box_module():
    import torch

    from experiments.capacity.baselines import class_region_arrays, parameter_counts

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.class_embed = torch.nn.Embedding(3, 2)
            self.class_offset = torch.nn.Embedding(3, 2)
            self.rel_embed = torch.nn.Embedding(1, 2)

    class Model:
        def __init__(self):
            self.module = M()
            self.class_index_dict = {"a": 0, "b": 1, "c": 2}

    model = Model()
    with torch.no_grad():
        model.module.class_embed.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        model.module.class_offset.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.5], [0.1, 0.1]]))
    kind, lo, hi = class_region_arrays(model)
    assert kind == "box"
    assert torch.allclose(lo[0], torch.tensor([-1.0, -1.0])) and torch.allclose(hi[0], torch.tensor([1.0, 1.0]))
    pc = parameter_counts(model)
    assert pc == {"n_params_total": 14, "n_params_class": 12, "n_classes": 3}


# ----------------------------------------------------------------------------- cells


def test_total_budget_exact_and_one_interval():
    from experiments.capacity.budget_total import run_budget_total

    kb = _dag_tables()
    ref = _exact_reference(kb, kb.least_model())
    out = run_budget_total("dag", fractions=[0.0, 1.0, 0.5], kb=kb, ref=ref, write=False,
                           keep_facts=True)
    assert [r["fraction"] for r in out] == [1.0, 0.5, 0.0]
    full, half, zero = out
    assert full["iterations"] == 0 and full["n_facts"] == 0
    assert full["precision_live"] == 1.0 and full["recall_live"] == 1.0
    assert full["achieved_total"] == full["exact_total"] == 9
    assert half["target_total"] == 8 and half["achieved_total"] == 8
    assert zero["achieved_total"] == zero["n_live_classes"] == 7 and zero["max_runs"] == 1
    for r in out:
        assert r["check_model_all_zero"] and r["recall_live"] == 1.0
        assert r["n_named_made_unsat"] == 0
    assert zero["precision_live"] == pytest.approx(0.75)
    assert zero["n_false_pairs_live"] == 2
    # nested: facts only grow along the descending fractions
    f = [set(map(tuple, r["facts"].tolist())) for r in out]
    assert f[0] <= f[1] <= f[2] and len(f[1]) == 1 and len(f[2]) == 2


def _protect_tables():
    """DAG plus B=7 ⊑ R2 and B ⊓ X ⊑ ⊥ (bottom 8).

    DFS: R1 A1 R2 A2 B R3 A3 X; X sits at {1, 3, 6, 7}. Its gaps are {R2} (R2 has the
    subclass B) and {B, R3}: filling either puts B into X and kills it.
    """
    gci0 = [(4, 0), (5, 1), (6, 2), (4, 3), (5, 3), (6, 3), (7, 1)]
    return tables_from_arrays(9, gci0=gci0, gci1_bot=[(7, 3, 8)], bot_id=8)


def test_total_budget_protect_disjoint():
    from experiments.capacity.budget_total import run_budget_total

    kb = _protect_tables()
    lm = kb.least_model()
    assert len(lm.unsatisfiable_classes()) == 0
    ref = _exact_reference(kb, lm)
    prot = run_budget_total("prot", fractions=[0.0], protect="disjoint", kb=kb, ref=ref,
                            write=False)[0]
    assert prot["status"] == "target not reachable (protected)"
    assert prot["n_named_made_unsat"] == 0 and prot["n_gaps_ineligible"] == 2
    assert prot["precision_live"] == 1.0 and prot["check_model_all_zero"]
    none = run_budget_total("prot", fractions=[0.0], protect="none", kb=kb, ref=ref,
                            write=False)[0]
    assert none["status"] == "ok" and none["check_model_all_zero"]
    assert none["n_named_made_unsat"] >= 1


def test_defaults_exclude_corrupt_datasets():
    from pathlib import Path

    from experiments.capacity.baselines import parse_cells
    from experiments.capacity.common import CORRUPT_DATASETS, DEFAULT_DATASETS

    assert len(CORRUPT_DATASETS) == 3 and not set(CORRUPT_DATASETS) & set(DEFAULT_DATASETS)
    cells = parse_cells(Path(__file__).with_name("baselines_cells.txt").read_text().splitlines())
    ds = {c[0] for c in cells}
    assert not ds & set(CORRUPT_DATASETS) and {"go", "foodon", "pizza"} <= ds


def test_parse_cells():
    from experiments.capacity.baselines import parse_cells

    cells = parse_cells(["# header", "", "pizza box2el 16 0", "tk3 elbe 2 1  # comment"])
    assert cells == [("pizza", "box2el", 16, 0), ("tk3", "elbe", 2, 1)]
    with pytest.raises(ValueError):
        parse_cells(["pizza box2el 16"])
    with pytest.raises(ValueError):
        parse_cells(["pizza box2el sixteen 0"])


def test_committed_cells_file_parses():
    from pathlib import Path

    from experiments.capacity.baselines import (
        BASELINE_DATASETS,
        BASELINE_DIMS,
        BASELINE_METHODS,
        BASELINE_SEEDS,
        parse_cells,
    )

    path = Path(__file__).with_name("baselines_cells.txt")
    cells = parse_cells(path.read_text(encoding="utf-8").splitlines())
    assert len(cells) == len(BASELINE_DATASETS) * len(BASELINE_METHODS) * len(BASELINE_DIMS) * len(BASELINE_SEEDS)
    assert len(set(cells)) == len(cells)
    assert {c[1] for c in cells} == set(BASELINE_METHODS)


# ----------------------------------------------------------------------------- B1


@pytest.mark.parametrize("theory", ["tk3", "tk4", "box2el_t1", "unsat_mix"])
def test_exact_agreement_on_theories(theory, tmp_path):
    import mowl

    mowl.init_jvm(os.environ.get("MOWL_JVM_MEMORY", "1536m"))
    from experiments.capacity.exact import run_exact

    p = run_exact(theory, out_dir=tmp_path, n_eval=500)
    assert (tmp_path / f"{theory}.json").exists()
    assert p["check_model_all_zero"]
    assert p["least_model"]["consistent"]
    ref = p["reference"]
    assert ref["exact_agreement"], ref
    assert ref["precision"] == 1.0 and ref["recall"] == 1.0 and ref["unsat_named_agree"]
    assert p["lp"]["all_equal"] and p["lp"]["max_fractionality"] == 0.0
    for o in ("dfs", "identity"):
        assert p["intervals"][o]["membership_equals_labels"]
    assert p["geometric"]["exact"] is True
    assert p["geometric"]["elem_viol"] == 0.0 and p["geometric"]["region_viol"] == 0.0
    if theory == "unsat_mix":
        assert ref["n_unsat_named_ref"] == 1 and p["least_model"]["n_unsat_named"] == 1
