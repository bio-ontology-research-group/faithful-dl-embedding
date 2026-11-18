"""Unit tests for build_eomqa --tbox-json and hybrid.exact_ceiling (toy data)."""
from __future__ import annotations

import json
import pickle

import pytest
import torch

from experiments.omqa import build_eomqa
from experiments.omqa.hybrid import exact_ceiling as ec
from experiments.omqa.hybrid.tbox import Rules


def _dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _toy_eomqa(root):
    """Entities 0..5 individuals, 6 = class A, 7 = class B; relations 0 = r, 1 = inv(r), 2 = type, 3 = type_inv.
    Generalisations give the range axioms ∃r⁻ ⊑ A (0, 6) and ∃inv(r)⁻ ⊑ B (1, 7) and the class inclusion A ⊑ B."""
    q_r, q_a, q_i = (0, (0,)), (6, (3,)), (5, (1,))
    _dump({("e", ("r",)): {q_r, q_a, q_i}}, root / "train_plain" / "train-queries.pkl")
    _dump({q_r: {1}, q_a: {1}, q_i: {4}}, root / "train_plain" / "train-answers.pkl")
    _dump({q_r: {((6, (3,)), ("e", ("r",)))}, q_a: {((7, (3,)), ("e", ("r",)))}, q_i: {((7, (3,)), ("e", ("r",)))}},
          root / "train_plain" / "train-generalizations.pkl")


def _tbox_json(path, mined, drop_ci=(), drop_rng=(), t_inv=None):
    d = {"T_inverse_type": mined["type_inv"] if t_inv is None else t_inv, "inverse": {"0": 1, "1": 0, "2": 3, "3": 2},
         "role_inclusion": [list(x) for x in mined["role_inclusion"]],
         "range_axiom_exists_rinv_sub_A": [list(x) for x in mined["range_axiom"] if tuple(x) not in drop_rng],
         "class_inclusion": [list(x) for x in mined["class_inclusion"] if tuple(x) not in drop_ci], "resolved": True}
    path.write_text(json.dumps(d))
    return path


def _load(p):
    return torch.load(p, weights_only=False)


def _build(root, out, *extra):
    assert build_eomqa.main(["--data-root", str(root), "--out", str(out), "--even-odd-inverses", *extra]) == 0
    return _load(out / "tables.pt"), json.loads((out / "build_report.json").read_text())


def test_tbox_json_all_identical_and_filtered_shrinks(tmp_path):
    root = tmp_path / "eomqa"
    _toy_eomqa(root)
    mined = build_eomqa.mine_tbox(root)
    assert mined["type_inv"] == 3 and mined["classes"] == [6, 7]
    assert mined["range_axiom"] == [(0, 6), (1, 7)] and mined["class_inclusion"] == [(6, 7)]
    t0, rep0 = _build(root, tmp_path / "noopt")
    t_all, rep_all = _build(root, tmp_path / "all", "--tbox-json", str(_tbox_json(tmp_path / "all.json", mined)))
    for k in t0:
        assert (torch.equal(t0[k], t_all[k]) if isinstance(t0[k], torch.Tensor) else t0[k] == t_all[k]), k
    assert "tbox_json" not in rep0
    tj = rep_all["tbox_json"]
    assert {k: (tj[k]["used"], tj[k]["dropped"], tj[k]["added"]) for k in ("class_inclusion", "range_axiom", "role_inclusion")} == \
        {"class_inclusion": (1, 0, 0), "range_axiom": (2, 0, 0), "role_inclusion": (0, 0, 0)}
    assert tj["inverse_differs_from_build"] == []
    # verified-style TBox: drop A ⊑ B and ∃inv(r)⁻ ⊑ B; class B keeps its id although it occurs in no axiom any more
    t_v, rep_v = _build(root, tmp_path / "vt", "--tbox-json", str(_tbox_json(tmp_path / "v.json", mined, drop_ci={(6, 7)}, drop_rng={(1, 7)})))
    for k in t0:
        if k in ("gci0", "gci3"):
            continue
        assert (torch.equal(t0[k], t_v[k]) if isinstance(t0[k], torch.Tensor) else t0[k] == t_v[k]), k
    cls, rol = t0["class_index_dict"], t0["object_property_index_dict"]
    assert "C7" in t_v["class_index_dict"] and t_v["class_index_dict"] == cls
    gci0_0, gci0_v = {tuple(x) for x in t0["gci0"].tolist()}, {tuple(x) for x in t_v["gci0"].tolist()}
    gci3_0, gci3_v = {tuple(x) for x in t0["gci3"].tolist()}, {tuple(x) for x in t_v["gci3"].tolist()}
    assert gci0_0 - gci0_v == {(cls["C6"], cls["C7"])} and gci0_v <= gci0_0
    # range (1, 7) is stored as gci3 (inv(1) = 0 as role, ⊤, B)
    assert gci3_0 - gci3_v == {(rol["R0"], t0["top_id"], cls["C7"])} and gci3_v <= gci3_0
    tj = rep_v["tbox_json"]
    assert tj["class_inclusion"]["dropped_axioms"] == [[6, 7]] and tj["range_axiom"]["dropped_axioms"] == [[1, 7]]
    assert rep_v["n_gci0"] == 0 and rep_v["n_gci3_range_domain"] == 1 and rep_v["mined_counts"] == rep0["mined_counts"]


def test_tbox_json_type_mismatch(tmp_path):
    root = tmp_path / "eomqa"
    _toy_eomqa(root)
    mined = build_eomqa.mine_tbox(root)
    with pytest.raises(ValueError):
        _build(root, tmp_path / "x", "--tbox-json", str(_tbox_json(tmp_path / "bad.json", mined, t_inv=2)))


# ------------------------------------------------------------------------------------------------ exact_ceiling
RULES = Rules(type_rel=2, type_inv_rel=3, inverse={0: 1, 1: 0, 2: 3, 3: 2}, role_inclusion=[(0, 4)],
              range_axiom=[(0, 6)], class_inclusion=[(6, 7)])


def test_closure():
    clo = ec.closure({(0, 0, 1)}, RULES)
    assert clo == {(0, 0, 1), (1, 1, 0), (0, 4, 1), (6, 3, 1), (1, 2, 6), (7, 3, 1), (1, 2, 7)}
    # without the class inclusion, B is not derived
    clo2 = ec.closure({(0, 0, 1)}, Rules(type_rel=2, type_inv_rel=3, inverse=RULES.inverse, range_axiom=[(0, 6)]))
    assert (7, 3, 1) not in clo2 and (1, 2, 7) not in clo2


def test_answer_shapes():
    adj = ec.adjacency({(0, 0, 1), (0, 0, 2), (1, 5, 3), (2, 5, 4), (9, 5, 3)})
    assert ec.answer(("e", ("r",)), (0, (0,)), adj) == {1, 2}
    assert ec.answer(("e", ("r", "r")), (0, (0, 5)), adj) == {3, 4}
    s2i = (("e", ("r",)), ("e", ("r",)))
    assert ec.answer(s2i, ((0, (0,)), (9, (5,))), adj) == set()
    assert ec.answer(s2i, ((0, (0, )), (0, (0,))), adj) == {1, 2}
    s2u = (("e", ("r",)), ("e", ("r",)), ("u",))
    assert ec.answer(s2u, ((0, (0,)), (9, (5,))), adj) == {1, 2, 3}
    sup = ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",))
    assert ec.answer(sup, (((0, (0,)), (9, (5,)), ("u",)), (5,)), adj) == {3, 4}
    spi = (("e", ("r", "r")), ("e", ("r",)))
    assert ec.answer(spi, ((0, (0, 5)), (9, (5,))), adj) == {3}


def test_expected_hits3():
    assert ec.expected_hits3(0, 0, 2) == 0.0
    assert ec.expected_hits3(2, 2, 2) == 1.0
    # 1 hard + 5 non-answers in the set: rank uniform over 6 positions -> 3/6
    assert ec.expected_hits3(6, 1, 2) == pytest.approx(0.5 / 2)
    # 2 hard + 2 non-answers: filtered, each hard competes with 2 non-answers -> 3/3 = 1
    assert ec.expected_hits3(4, 2, 2) == 1.0


def test_evaluate_case(tmp_path):
    root = tmp_path / "d"
    q1, q2 = (0, (0,)), (1, (1,))
    _dump({("e", ("r",)): {q1, q2}}, root / "case" / "test-queries.pkl")
    _dump({q1: {1}, q2: set()}, root / "case" / "test-easy-answers.pkl")
    _dump({q1: {2, 5}, q2: {0}}, root / "case" / "test-hard-answers.pkl")
    adj = ec.adjacency({(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 9), (1, 1, 0)})
    r = ec.evaluate_case(root, "case", adj)
    v = r["per_shape"]["1p"]
    # q1: set {2,3,4,9} after removing easy 1; hard {2,5}: inter 1, n 4 -> recall 0.5, H@3 = min(1, 3/4)/2, FP 3
    # q2: set {0}, hard {0}: recall 1, H@3 1, FP 0
    assert v["n"] == 2 and v["recall"] == pytest.approx(0.75) and v["fp"] == pytest.approx(1.5)
    assert v["hits3"] == pytest.approx((0.375 + 1.0) / 2) and r["macro"]["hits3"] == pytest.approx(v["hits3"])
