"""End-to-end smoke test of the query-answering pipeline on a toy KB (CPU, seconds).

Toy KB: entities 0..9, class entities 100 (A) and 101 (B), relations 0 (r), 1 (r⁻), 2 = type, 3 = type⁻.
Facts: r(i, i+1) for i in 0..8 (+ inverses); A(0..4), B(5..9); TBox: ∃r⁻ ⊑ B is *false* here, so we only add
A ⊑ ⊤-style harmless axioms: class inclusion none, role inclusion none. Queries (Q2B format) in a fake
``test_ind_ded`` split with 1p and 2p shapes; hard answers = the true answers over the facts.
The test asserts that the pipeline runs and that the trained model ranks true 1p answers above random (MRR > 0.3).
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import tempfile
from pathlib import Path

import pytest
import torch as th


def _tmp_base_dir() -> str:
    return os.environ.get("TMPDIR", str(Path.home() / "tmp"))


def _make_toy(root: Path) -> Path:
    # tables as build_eomqa.py would produce them
    inds = list(range(10))
    ind_index = {f"I{e}": i for i, e in enumerate(inds)}
    class_index = {"C100": 0, "C101": 1, "owl:Thing": 2}
    role_index = {"R0": 0, "R1": 1}
    role_assert = set()
    for i in range(9):
        role_assert.add((i, 0, i + 1))
        role_assert.add((i + 1, 1, i))
    class_assert = {(i, 0) for i in range(5)} | {(i, 1) for i in range(5, 10)}
    tables = {
        "class_index_dict": class_index, "individual_index_dict": ind_index, "object_property_index_dict": role_index,
        "top_id": 2, "inverse_of": [1, 0], "type_inv_rel": 3, "type_fwd_rel": 2,
        "entity_to_individual": {e: i for e, i in enumerate(inds)}, "entity_to_class": {100: 0, 101: 1},
        "relation_to_role": {0: 0, 1: 1},
        "gci0": th.zeros((0, 2), dtype=th.long), "gci3": th.zeros((0, 3), dtype=th.long),
        "role_inclusion": th.zeros((0, 2), dtype=th.long),
        "class_assertion": th.tensor(sorted(class_assert), dtype=th.long),
        "object_property_assertion": th.tensor(sorted(role_assert), dtype=th.long),
    }
    tdir = root / "tables"
    tdir.mkdir(parents=True)
    th.save(tables, tdir / "tables.pt")
    # queries: 1p (e, (0,)) -> {e+1}; 2p (e, (0, 0)) -> {e+2}; type⁻ anchor (100, (3,)) -> {0..4}
    data = root / "data"
    for part in ("valid", "test"):
        d = data / "test_ind_ded"
        d.mkdir(parents=True, exist_ok=True)
        queries = collections.defaultdict(set)
        easy, hard = {}, {}
        for e in range(0, 8):
            q = (e, (0,))
            queries[("e", ("r",))].add(q)
            easy[q], hard[q] = set(), {e + 1}
            q2 = (e, (0, 0))
            queries[("e", ("r", "r"))].add(q2)
            easy[q2], hard[q2] = set(), {e + 2}
        q3 = (100, (3,))
        queries[("e", ("r",))].add(q3)
        easy[q3], hard[q3] = {0, 1}, {2, 3, 4}
        for name, obj in (("queries", dict(queries)), ("easy-answers", easy), ("hard-answers", hard)):
            with open(d / f"{part}-{name}.pkl", "wb") as fh:
                pickle.dump(obj, fh)
    return root


@pytest.mark.skipif(os.environ.get("GFE_QA_SMOKE", "1") != "1", reason="set GFE_QA_SMOKE=1")
@pytest.mark.parametrize("extra", [[], ["--neg-weight", "1.0", "--neg-k", "2"]], ids=["plain", "negatives"])
def test_pipeline_toy(extra) -> None:
    from experiments.omqa import train as train_mod

    with tempfile.TemporaryDirectory(dir=_tmp_base_dir(), prefix="qa_toy_") as tmp:
        root = _make_toy(Path(tmp))
        out = Path(tmp) / "out"
        rc = train_mod.main([
            "--name", "toy", "--benchmark", "eomqa", "--tables", str(root / "tables" / "tables.pt"),
            "--data-root", str(root / "data"), "--test-cases", "test_ind_ded", "--out", str(out), "--device", "cpu",
            "--dim", "8", "--K", "1", "--Kr", "2", "--batch", "64", "--epochs", "60", "--tau-epochs", "20",
            "--eval-every", "20", "--n-eval", "200", "--exact", *extra,
        ])
        assert rc == 0
        res = json.loads((out / "toy" / "results.json").read_text())
        assert "test_test_ind_ded" in res and "1p" in res["test_test_ind_ded"]
        assert res["test_test_ind_ded"]["1p"]["mrr"] > 0.3, res["test_test_ind_ded"]
        assert "crisp_report" in res
