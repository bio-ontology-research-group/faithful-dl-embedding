"""Unit tests for the rule closure of QTO adjacencies (toy KB, hand-computed expectations)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from experiments.omqa.hybrid.closure_adj import closure, coo_max
from experiments.omqa.hybrid.closure_adj import main as closure_main
from experiments.omqa.hybrid.qto_io import chunk_bounds, csr_to_chunks, load_qto_adj, save_qto_adj
from experiments.omqa.hybrid.tbox import Rules, rules_from_json

# toy KB: individuals 0..5, classes A = 6, B = 7
N = 8
A, B = 6, 7
R, S, RI, SI, TYPE, TINV, P, Q = range(8)  # relation ids
NREL = 8
RULES = Rules(
    type_rel=TYPE, type_inv_rel=TINV,
    inverse={R: RI, RI: R, S: SI, SI: S, TYPE: TINV, TINV: TYPE},
    role_inclusion=[(R, S), (P, Q), (Q, P)],  # chain r ⊑ s; cycle p ⊑ q ⊑ p
    range_axiom=[(S, A)],  # ∃s⁻ ⊑ A
    class_inclusion=[(A, B)],  # A ⊑ B
    classes=[A, B],
)
FACTS = {R: {(0, 1): 1.0, (2, 3): 0.6}, P: {(4, 5): 0.8}, Q: {(5, 4): 0.3}, TYPE: {(0, A): 0.9}}


def to_mats(spec):
    mats = []
    for r in range(NREL):
        d = spec.get(r, {})
        rows = [i for i, _ in d]
        cols = [j for _, j in d]
        mats.append(sp.csr_matrix((np.array(list(d.values()), dtype=np.float32), (rows, cols)), shape=(N, N)))
    return mats


def as_dict(m):
    c = m.tocoo()
    return {(int(i), int(j)): float(v) for i, j, v in zip(c.row, c.col, c.data)}


def check(mats, expected):
    for r in range(NREL):
        got = as_dict(mats[r])
        exp = expected.get(r, {})
        assert set(got) == set(exp), f"relation {r}: {sorted(got)} != {sorted(exp)}"
        for k, v in exp.items():
            assert got[k] == pytest.approx(v, abs=1e-6), f"relation {r} entry {k}: {got[k]} != {v}"


def transpose(d):
    return {(j, i): v for (i, j), v in d.items()}


EXP_D1 = {
    R: {(0, 1): 1.0, (2, 3): 0.6},
    S: {(0, 1): 1.0, (2, 3): 0.6},
    RI: {(1, 0): 1.0, (3, 2): 0.6},
    SI: {(1, 0): 1.0, (3, 2): 0.6},
    P: {(4, 5): 0.8, (5, 4): 0.3},
    Q: {(4, 5): 0.8, (5, 4): 0.3},
    TYPE: {(0, A): 0.9, (1, A): 1.0, (3, A): 0.6, (0, B): 0.9, (1, B): 1.0, (3, B): 0.6},
}
EXP_D1[TINV] = transpose(EXP_D1[TYPE])


def test_closure_decay1_hand_computed():
    mats, info = closure(to_mats(FACTS), RULES, decay=1.0, log=lambda s: None)
    assert info["converged"]
    check(mats, EXP_D1)


def test_closure_idempotent():
    mats, _ = closure(to_mats(FACTS), RULES, decay=1.0, log=lambda s: None)
    mats2, info2 = closure([m.copy() for m in mats], RULES, decay=1.0, log=lambda s: None)
    assert info2["iterations"] == 1 and info2["converged"]
    for a, b in zip(mats, mats2):
        assert (a != b).nnz == 0
    # also for decay < 1
    m1, _ = closure(to_mats(FACTS), RULES, decay=0.5, log=lambda s: None)
    m2, i2 = closure([m.copy() for m in m1], RULES, decay=0.5, log=lambda s: None)
    assert i2["iterations"] == 1
    for a, b in zip(m1, m2):
        assert (a != b).nnz == 0


def test_closure_decay():
    d = 0.5
    mats, info = closure(to_mats(FACTS), RULES, decay=d, log=lambda s: None)
    assert info["converged"]
    typ = {(0, A): 0.9, (1, A): 0.25, (3, A): 0.15, (0, B): 0.45, (1, B): 0.125, (3, B): 0.075}
    exp = {
        R: {(0, 1): 1.0, (2, 3): 0.6},
        S: {(0, 1): 0.5, (2, 3): 0.3},  # 1 step
        RI: {(1, 0): 0.5, (3, 2): 0.3},  # 1 step
        SI: {(1, 0): 0.25, (3, 2): 0.15},  # r -> s -> s_inv
        P: {(4, 5): 0.8, (5, 4): 0.15},
        Q: {(4, 5): 0.4, (5, 4): 0.3},
        TYPE: typ,  # r -> s -> range (2 steps); CI one more
        TINV: {k: v * d for k, v in transpose(typ).items()},
    }
    check(mats, exp)


def test_closure_min_score():
    mats, _ = closure(to_mats(FACTS), RULES, decay=1.0, min_score=0.7, log=lambda s: None)
    typ = {(0, A): 0.9, (1, A): 1.0, (0, B): 0.9, (1, B): 1.0}
    exp = {
        R: {(0, 1): 1.0, (2, 3): 0.6}, S: {(0, 1): 1.0}, RI: {(1, 0): 1.0}, SI: {(1, 0): 1.0},
        P: {(4, 5): 0.8}, Q: {(4, 5): 0.8, (5, 4): 0.3}, TYPE: typ, TINV: transpose(typ),
    }
    check(mats, exp)


def test_closure_max_iter_stops():
    _, info = closure(to_mats(FACTS), RULES, decay=1.0, max_iter=1, log=lambda s: None)
    assert info["iterations"] == 1 and not info["converged"]


def test_coo_max_dedups_by_max():
    m = coo_max(np.array([0, 0, 1]), np.array([1, 1, 2]), np.array([0.2, 0.7, 0.5], dtype=np.float32), (3, 3))
    assert as_dict(m) == {(0, 1): pytest.approx(0.7), (1, 2): pytest.approx(0.5)}


# ------------------------------------------------------------------------------------------- QTO file format
def qto_reference_chunks(dense: torch.Tensor, fraction: int):
    """The chunking code of QTO model.py (KGReasoning.__init__), verbatim apart from the device."""
    nentity = dense.shape[0]
    out = []
    dim = nentity // fraction
    rest = nentity - fraction * dim
    for i in range(fraction):
        s = i * dim
        t = (i + 1) * dim
        if i == fraction - 1:
            t += rest
        out.append(dense[s:t, :].to_sparse())
    return out


@pytest.mark.parametrize("n,fraction", [(8, 3), (23, 10), (10, 10)])
def test_qto_format_round_trip(tmp_path, n, fraction):
    rng = np.random.default_rng(n)
    dense = [torch.from_numpy((rng.random((n, n)) * (rng.random((n, n)) < 0.3)).astype(np.float32)) for _ in range(3)]
    dense[1][0, 0] = 1.0
    ref_path = tmp_path / "ref.pt"
    torch.save([qto_reference_chunks(d, fraction) for d in dense], ref_path)
    mats = load_qto_adj(ref_path, n)
    for m, d in zip(mats, dense):
        assert np.array_equal(m.toarray(), d.numpy())
    out = tmp_path / "neural_adj" / f"toyC_{fraction}_0.0002.pt"
    save_qto_adj(mats, out, fraction)
    assert not (out.parent / (out.name + ".partial")).exists()
    # read back the way QTO does (torch.load with map_location), with and without weights_only
    for kw in ({}, {"weights_only": True}, {"weights_only": False}):
        obj = torch.load(out, map_location="cpu", **kw)
        ref = torch.load(ref_path, map_location="cpu", **kw)
        assert isinstance(obj, list) and len(obj) == len(dense)
        for chunks, rchunks, d in zip(obj, ref, dense):
            assert isinstance(chunks, list) and len(chunks) == fraction
            for (s, t), ch, rch in zip(chunk_bounds(n, fraction), chunks, rchunks):
                assert ch.layout == torch.sparse_coo and ch.dtype == torch.float32 and ch.device.type == "cpu"
                assert ch.is_coalesced() and tuple(ch.shape) == (t - s, n) == tuple(rch.shape)
                assert torch.equal(ch.indices(), rch.indices()) and torch.equal(ch.values(), rch.values())
                assert torch.equal(ch.to_dense(), d[s:t])  # what QTO relation_projection uses
    back = load_qto_adj(out)
    for a, b in zip(mats, back):
        assert (a != b).nnz == 0


def test_cli_end_to_end(tmp_path):
    """CLI on the toy KB: QTO-format input -> closed QTO-format output + report."""
    data = tmp_path / "data" / "toy-ded"
    data.mkdir(parents=True)
    (data / "stats.txt").write_text(f"numentity: {N}\nnumrelations: {NREL}\n")
    adj = tmp_path / "neural_adj" / "toy_10_0.0002.pt"
    adj.parent.mkdir()
    torch.save([qto_reference_chunks(torch.from_numpy(m.toarray()), 3) for m in to_mats(FACTS)], adj)
    tb = {"T_inverse_type": TINV, "type_rel": TYPE, "inverse": {str(k): v for k, v in RULES.inverse.items()},
          "role_inclusion": RULES.role_inclusion, "range_axiom_exists_rinv_sub_A": RULES.range_axiom,
          "class_inclusion": RULES.class_inclusion, "classes": RULES.classes, "resolved": True}
    assert rules_from_json(json.loads(json.dumps(tb))) == RULES
    (tmp_path / "tbox.json").write_text(json.dumps(tb))
    out = tmp_path / "neural_adj" / "toyC_10_0.0002.pt"
    rep = tmp_path / "report.json"
    assert closure_main(["--adj", str(adj), "--tbox", str(tmp_path / "tbox.json"), "--data-dir", str(data),
                         "--out", str(out), "--fraction", "3", "--report", str(rep)]) == 0
    check(load_qto_adj(out), EXP_D1)
    r = json.loads(rep.read_text())
    assert r["converged"] and len(r["per_relation"]) == NREL
    pr = {x["rel"]: x for x in r["per_relation"]}
    assert pr[S]["nnz_before"] == 0 and pr[S]["nnz_after"] == 2 and pr[S]["n_one_after"] == 1
    assert pr[TYPE]["nnz_after"] == 6 and pr[TYPE]["n_one_before"] == 0 and pr[TYPE]["n_one_after"] == 2
