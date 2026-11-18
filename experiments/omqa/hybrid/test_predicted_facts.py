"""Unit tests for predicted-fact extraction and build_eomqa --extra-triples (toy data)."""
from __future__ import annotations

import json
import pickle

import numpy as np
import scipy.sparse as sp
import torch

from experiments.omqa import build_eomqa
from experiments.omqa.hybrid.predicted_facts import main as pred_main
from experiments.omqa.hybrid.predicted_facts import read_tsv, topk_new_facts, write_tsv
from experiments.omqa.hybrid.qto_io import csr_to_chunks
from experiments.omqa.hybrid.tbox import Rules

N = 6
RULES = Rules(inverse={0: 1, 1: 0})
ADJ = {0: {(0, 1): 1.0, (0, 2): 0.95, (0, 3): 0.8, (0, 4): 0.75, (0, 5): 0.6, (2, 3): 0.35, (2, 4): 0.2},
       1: {(1, 0): 0.99, (3, 2): 0.5}}
TRAIN = np.array([[0, 0, 1]], dtype=np.int64)


def mats():
    out = []
    for r in range(2):
        d = ADJ[r]
        out.append(sp.csr_matrix((np.array(list(d.values()), np.float32), ([i for i, _ in d], [j for _, j in d])), shape=(N, N)))
    return out


def facts(res, thr):
    sel = res["score"] >= thr
    return sorted(zip(res["h"][sel].tolist(), res["r"][sel].tolist(), res["t"][sel].tolist()))


def test_threshold_exclusion_cap():
    res = topk_new_facts(mats(), TRAIN, RULES, 0.3, 3)
    assert facts(res, 0.9) == [(0, 0, 2)]
    assert facts(res, 0.7) == [(0, 0, 2), (0, 0, 3), (0, 0, 4)]
    assert facts(res, 0.5) == [(0, 0, 2), (0, 0, 3), (0, 0, 4), (3, 1, 2)]  # (0,0,5) removed by the cap of 3
    assert facts(res, 0.3) == [(0, 0, 2), (0, 0, 3), (0, 0, 4), (2, 0, 3), (3, 1, 2)]
    assert res["n_excluded_train"] == 1 and res["n_excluded_inverse_of_train"] == 1 and res["n_excluded_cap"] == 1
    # cap 1 keeps only the best new tail per (h, r)
    res1 = topk_new_facts(mats(), TRAIN, RULES, 0.3, 1)
    assert facts(res1, 0.3) == [(0, 0, 2), (2, 0, 3), (3, 1, 2)]
    # without the inverse exclusion (1, 1, 0) (inverse of the training triple) is a "new" fact
    res2 = topk_new_facts(mats(), TRAIN, RULES, 0.3, 3, exclude_inverse_of_train=False)
    assert (1, 1, 0) in facts(res2, 0.9)


def test_tsv_round_trip(tmp_path):
    res = topk_new_facts(mats(), TRAIN, RULES, 0.3, 3)
    p = tmp_path / "pred.tsv"
    write_tsv(p, res["h"], res["r"], res["t"], res["score"])
    back = read_tsv(p)
    assert sorted(map(tuple, back.tolist())) == facts(res, 0.3)
    lines = p.read_text().splitlines()
    assert all(len(line.split("\t")) == 4 for line in lines)
    assert float(lines[0].split("\t")[3]) == np.float32(0.95).round(6)
    p2 = tmp_path / "noscore.tsv"
    p2.write_text("1\t0\t2\n3\t1\t4\n")
    assert read_tsv(p2).tolist() == [[1, 0, 2], [3, 1, 4]]
    assert build_eomqa.read_extra_triples([str(p2)]) == {(1, 0, 2), (3, 1, 4)}


def _dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def test_cli_counts_and_precision(tmp_path):
    root = tmp_path / "data"
    dd = root / "toy-ded"
    dd.mkdir(parents=True)
    (dd / "stats.txt").write_text(f"numentity: {N}\nnumrelations: 2\n")
    (dd / "train.txt").write_text("0\t0\t1\n")
    q1, q2 = (0, (0,)), (2, (0,))
    _dump({("e", ("r",)): {q1, q2}}, dd / "test-queries.pkl")
    _dump({q1: {1}, q2: set()}, dd / "test-easy-answers.pkl")
    _dump({q1: {2, 5}, q2: {3}}, dd / "test-hard-answers.pkl")
    adj = tmp_path / "toy_10_0.0002.pt"
    torch.save([csr_to_chunks(m, 2) for m in mats()], adj)
    tb = tmp_path / "tbox.json"
    tb.write_text(json.dumps({"T_inverse_type": None, "inverse": {"0": 1, "1": 0}, "role_inclusion": [], "resolved": True}))
    out = tmp_path / "pred"
    assert pred_main(["--adj", str(adj), "--data-dir", str(dd), "--tbox", str(tb), "--thresholds", "0.9", "0.7", "0.5", "0.3",
                      "--max-per-head-rel", "3", "--out-dir", str(out)]) == 0
    c = json.loads((out / "counts.json").read_text())
    assert {k: v["n_new"] for k, v in c["thresholds"].items()} == {"0.9": 1, "0.7": 3, "0.5": 4, "0.3": 5}
    assert c["thresholds"]["0.3"]["precision_all"] == 0.6 and c["thresholds"]["0.3"]["precision_judgeable"] == 0.6
    assert c["thresholds"]["0.9"]["precision_all"] == 1.0
    assert c["thresholds"]["0.3"]["per_relation"]["0"] == {"n_new": 4, "n_train": 1, "ratio": 4.0}
    assert c["thresholds"]["0.3"]["ratio_to_train"] == 5.0
    assert sorted(map(tuple, read_tsv(out / "pred_t0.3.tsv").tolist())) == [(0, 0, 2), (0, 0, 3), (0, 0, 4), (2, 0, 3), (3, 1, 2)]


def _toy_eomqa(root):
    """E-OMQA-style toy release: entities 0..5 individuals, 6 = class A; relations 0 = r, 1 = inv(r), 2 = type,
    3 = type_inv. The generalisation (0, (0,)) -> (6, (3,)) gives the range axiom ∃r⁻ ⊑ A and type_inv = 3."""
    q_r, q_a, q_i = (0, (0,)), (6, (3,)), (5, (1,))
    _dump({("e", ("r",)): {q_r, q_a, q_i}}, root / "train_plain" / "train-queries.pkl")
    _dump({q_r: {1}, q_a: {1}, q_i: {4}}, root / "train_plain" / "train-answers.pkl")
    _dump({q_r: {((6, (3,)), ("e", ("r",)))}}, root / "train_plain" / "train-generalizations.pkl")


def _load_tables(path):
    return torch.load(path, weights_only=False)


def test_extra_triples_build(tmp_path):
    root = tmp_path / "eomqa"
    _toy_eomqa(root)
    base, ext = tmp_path / "base", tmp_path / "ext"
    assert build_eomqa.main(["--data-root", str(root), "--out", str(base), "--even-odd-inverses"]) == 0
    extra = tmp_path / "extra.tsv"
    extra.write_text("2\t0\t3\t0.9\n6\t3\t2\t0.8\n4\t2\t6\n0\t0\t1\t0.99\n")
    assert build_eomqa.main(["--data-root", str(root), "--out", str(ext), "--even-odd-inverses", "--extra-triples", str(extra)]) == 0
    tb, te = _load_tables(base / "tables.pt"), _load_tables(ext / "tables.pt")
    assert tb["type_inv_rel"] == 3 and tb["type_fwd_rel"] == 2
    ind, cls, rol = te["individual_index_dict"], te["class_index_dict"], te["object_property_index_dict"]
    A = cls["C6"]
    ca = {tuple(x) for x in te["class_assertion"].tolist()}
    opa = {tuple(x) for x in te["object_property_assertion"].tolist()}
    assert ca == {(ind["I1"], A), (ind["I2"], A), (ind["I4"], A)}
    base_opa = lambda i, r: {(i["I0"], r["R0"], i["I1"]), (i["I1"], r["R1"], i["I0"]), (i["I5"], r["R1"], i["I4"]), (i["I4"], r["R0"], i["I5"])}
    assert opa == base_opa(ind, rol) | {(ind["I2"], rol["R0"], ind["I3"]), (ind["I3"], rol["R1"], ind["I2"])}
    ib, cb, rb = tb["individual_index_dict"], tb["class_index_dict"], tb["object_property_index_dict"]
    assert {tuple(x) for x in tb["class_assertion"].tolist()} == {(ib["I1"], cb["C6"])}
    assert {tuple(x) for x in tb["object_property_assertion"].tolist()} == base_opa(ib, rb)
    # TBox, roles and inverse pairing are unchanged
    for k in ("gci0", "gci3", "role_inclusion"):
        assert torch.equal(tb[k], te[k])
    assert tb["object_property_index_dict"] == te["object_property_index_dict"] and tb["inverse_of"] == te["inverse_of"]
    rep = json.loads((ext / "build_report.json").read_text())
    assert rep["extra_triples"]["n_read"] == 4 and rep["extra_triples"]["n_new"] == 3
    assert rep["extra_triples"]["n_new_class_entities_from_type_inv_heads"] == 0
    assert "extra_triples" not in json.loads((base / "build_report.json").read_text())


def test_no_option_output_unchanged(tmp_path):
    """Without --extra-triples the build is byte-for-byte reproducible (the real-data identity check against the
    existing LUBM tables is run separately; see the task summary)."""
    root = tmp_path / "eomqa"
    _toy_eomqa(root)
    a, b = tmp_path / "a", tmp_path / "b"
    build_eomqa.main(["--data-root", str(root), "--out", str(a), "--even-odd-inverses"])
    build_eomqa.main(["--data-root", str(root), "--out", str(b), "--even-odd-inverses"])
    ta, tb = _load_tables(a / "tables.pt"), _load_tables(b / "tables.pt")
    assert ta.keys() == tb.keys()
    for k in ta:
        assert (torch.equal(ta[k], tb[k]) if isinstance(ta[k], torch.Tensor) else ta[k] == tb[k]), k
