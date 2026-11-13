"""Fast CPU unit tests for ``experiments/omqa/faithfulness.py``."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import tempfile
from pathlib import Path

import pytest
import torch as th
import torch.nn.functional as F

from experiments.omqa import faithfulness as F_mod
from experiments.omqa.train import build_model


def _tmp_base_dir() -> str:
    base = Path(os.environ.get("TMPDIR", str(Path.home() / "tmp")))
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _inv_softplus(w: th.Tensor) -> th.Tensor:
    val = w.clamp_min(1e-6)
    return th.log(th.expm1(val))


def test_set_metrics_hand_arithmetic() -> None:
    """Verify precision, recall, F1, Jaccard, |A|, exact_match, empty_rate, complete, sound on hand examples."""
    # 5 entities (0..4), all valid candidates
    cand_valid = th.ones(5, dtype=th.bool)

    # 5 queries with known scores at margin=0.0:
    # q0: A = {0, 2, 4}, E = {0, 1}, H = {2, 3} -> C = {0, 1, 2, 3}
    #   against C: tp=2 ({0,2}), |A|=3, |C|=4, |A u C|=5 -> P=2/3, R=2/4=0.5, F1=4/7, J=2/5=0.4, EM=0, comp=0, sound=0
    #   against H (filtered: E={0,1} removed -> A_filt={2, 4}): tp=1 ({2}), |A_filt|=2, |H|=2, |u|=3 -> P=0.5, R=0.5, F1=0.5, J=1/3
    # q1: exact match: A = {1, 2, 3}, E = {1}, H = {2, 3} -> C = {1, 2, 3}, A_filt = {2, 3}
    #   against C: P=1, R=1, F1=1, J=1, |A|=3, EM=1, empty=0, comp=1, sound=1
    #   against H: P=1, R=1, F1=1, J=1, |A_filt|=2, EM=1, empty=0, comp=1, sound=1
    # q2: sound but incomplete: A = {1}, E = {}, H = {1, 2}
    #   P=1, R=0.5, F1=2/3, J=0.5, |A|=1, EM=0, empty=0, comp=0, sound=1
    # q3: complete but unsound: A = {1, 2}, E = {}, H = {1}
    #   P=0.5, R=1, F1=2/3, J=0.5, |A|=2, EM=0, empty=0, comp=1, sound=0
    # q4: empty prediction: A = {}, E = {}, H = {1}
    #   P=0, R=0, F1=0, J=0, |A|=0, EM=0, empty=1, comp=0, sound=0
    scores = th.tensor(
        [
            [0.5, -0.2, 0.3, -0.1, 0.8],   # q0: {0, 2, 4}
            [-0.5, 0.4, 0.6, 0.2, -0.3],   # q1: {1, 2, 3}
            [-0.1, 0.9, -0.4, -0.2, -0.5], # q2: {1}
            [-0.2, 0.7, 0.1, -0.3, -0.4],  # q3: {1, 2}
            [-0.1, -0.2, -0.3, -0.4, -0.5],# q4: {}
        ],
        dtype=th.float32,
    )
    cert_mask = th.tensor(
        [
            [True, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, False, False],
            [False, True, False, False, False],
            [False, True, False, False, False],
        ],
        dtype=th.bool,
    )
    hard_mask = th.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, True, True, False, False],
            [False, True, False, False, False],
            [False, True, False, False, False],
        ],
        dtype=th.bool,
    )

    m_cert, m_hard = F_mod.evaluate_batch_sets(scores, cand_valid, cert_mask, hard_mask, margin=0.0)

    # Check q0 against C and H
    assert math.isclose(float(m_cert["precision"][0]), 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(float(m_cert["recall"][0]), 0.5, rel_tol=1e-6)
    assert math.isclose(float(m_cert["f1"][0]), 4.0 / 7.0, rel_tol=1e-6)
    assert math.isclose(float(m_cert["jaccard"][0]), 0.4, rel_tol=1e-6)
    assert float(m_cert["size"][0]) == 3.0
    assert float(m_cert["exact_match"][0]) == 0.0
    assert float(m_cert["empty_rate"][0]) == 0.0
    assert float(m_cert["complete"][0]) == 0.0
    assert float(m_cert["sound"][0]) == 0.0

    assert math.isclose(float(m_hard["precision"][0]), 0.5, rel_tol=1e-6)
    assert math.isclose(float(m_hard["recall"][0]), 0.5, rel_tol=1e-6)
    assert math.isclose(float(m_hard["f1"][0]), 0.5, rel_tol=1e-6)
    assert math.isclose(float(m_hard["jaccard"][0]), 1.0 / 3.0, rel_tol=1e-6)
    assert float(m_hard["size"][0]) == 2.0
    assert float(m_hard["exact_match"][0]) == 0.0

    # Check q1 exact match
    for k in ("precision", "recall", "f1", "jaccard", "exact_match", "complete", "sound"):
        assert float(m_cert[k][1]) == 1.0
        assert float(m_hard[k][1]) == 1.0
    assert float(m_cert["size"][1]) == 3.0
    assert float(m_hard["size"][1]) == 2.0
    assert float(m_cert["empty_rate"][1]) == 0.0

    # Check q2 sound & incomplete
    assert float(m_cert["precision"][2]) == 1.0
    assert float(m_cert["recall"][2]) == 0.5
    assert math.isclose(float(m_cert["f1"][2]), 2.0 / 3.0, rel_tol=1e-6)
    assert float(m_cert["sound"][2]) == 1.0
    assert float(m_cert["complete"][2]) == 0.0
    assert float(m_cert["exact_match"][2]) == 0.0

    # Check q3 complete & unsound
    assert float(m_cert["precision"][3]) == 0.5
    assert float(m_cert["recall"][3]) == 1.0
    assert math.isclose(float(m_cert["f1"][3]), 2.0 / 3.0, rel_tol=1e-6)
    assert float(m_cert["sound"][3]) == 0.0
    assert float(m_cert["complete"][3]) == 1.0
    assert float(m_cert["exact_match"][3]) == 0.0

    # Check q4 empty prediction
    for k in ("precision", "recall", "f1", "jaccard", "size", "exact_match", "complete", "sound"):
        assert float(m_cert[k][4]) == 0.0
    assert float(m_cert["empty_rate"][4]) == 1.0


def _build_hand_fixture(root: Path) -> Tuple[Path, Path, Path]:
    """Build a tiny hand-crafted E-OMQA tables dict, GroundedEL checkpoint, and query split with known answers."""
    # Entities: individuals 0..4 (elements 3..7), classes 100->0, 101->1, owl:Thing->2
    class_index = {"C100": 0, "C101": 1, "owl:Thing": 2}
    ind_index = {f"I{i}": i for i in range(5)}
    role_index = {"R0": 0, "R1": 1}

    tables = {
        "class_index_dict": class_index,
        "individual_index_dict": ind_index,
        "object_property_index_dict": role_index,
        "top_id": 2,
        "inverse_of": [1, 0],
        "type_inv_rel": 3,
        "type_fwd_rel": 2,
        "entity_to_individual": {i: i for i in range(5)},
        "entity_to_class": {100: 0, 101: 1},
        "relation_to_role": {0: 0, 1: 1},
        "gci0": th.tensor([[0, 1]], dtype=th.long),  # C100 <= C101
        "gci3": th.zeros((0, 3), dtype=th.long),
        "role_inclusion": th.zeros((0, 2), dtype=th.long),
        "class_assertion": th.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=th.long),
        "object_property_assertion": th.tensor(
            [[0, 0, 1], [1, 1, 0], [1, 0, 2], [2, 1, 1]], dtype=th.long
        ),
    }
    tdir = root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tables_path = tdir / "tables.pt"
    th.save(tables, tables_path)

    cell_args = {
        "name": "hand_toy",
        "benchmark": "eomqa",
        "tables": "/nonexistent/orix/tables.pt",
        "data_root": "/nonexistent/orix/data",
        "test_cases": ["test_ded"],
        "out": str(root / "runs"),
        "device": "cpu",
        "seed": 0,
        "dim": 2,
        "K": 1,
        "Kr": 1,
        "batch": 16,
        "epochs": 1,
        "lr": 0.01,
        "lr_final": 0.0001,
        "tau0": 1.0,
        "tau1": 0.1,
        "tau_epochs": 1,
        "eps_min": 0.01,
        "eps_role": 0.01,
        "neg_weight": 0.0,
        "neg_k": 1,
        "constraint_mode": "fixed",
        "dual_rho": 1.0,
        "dual_max": 1000.0,
        "dual_target": 0.0,
        "lambda_reg": 1.0,
        "n_region_samples": 4,
        "max_boxes": 16,
        "scorer": "beam",
        "beam_k": 16,
        "n_eval": 200,
    }
    ns = argparse.Namespace(**cell_args)
    model = build_model(tables, ns)
    model.init_module()
    mod = model.module

    with th.no_grad():
        # Element positions:
        # 0 (C100): (0.5, 2.0)
        # 1 (C101): (0.5, 2.0)
        # 2 (owl:Thing): (0.0, 0.0)
        # 3 (I0): (0.0, 0.0)
        # 4 (I1): (1.0, 0.0)
        # 5 (I2): (2.0, 0.0)
        # 6 (I3): (2.3, 0.0)
        # 7 (I4): (5.0, 0.0)
        pos = th.tensor(
            [
                [0.5, 2.0],
                [0.5, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.3, 0.0],
                [5.0, 0.0],
            ],
            dtype=th.float32,
        )
        mod.elem_pos.weight.copy_(pos)
        mod.dead_logits.zero_()

        # Class boxes (K=1):
        # C100: cen=(0.5, 1.0), w=(0.8, 1.5) -> contains I0(0,0), I1(1,0), C100(0.5,2.0), C101(0.5,2.0)
        # C101: cen=(0.5, 1.0), w=(1.0, 1.7) -> strictly contains C100 with margin 0.2
        mod.class_centers[0, 0] = th.tensor([0.5, 1.0])
        mod.class_raw_widths[0, 0] = _inv_softplus(th.tensor([0.8, 1.5]))
        mod.class_centers[1, 0] = th.tensor([0.5, 1.0])
        mod.class_raw_widths[1, 0] = _inv_softplus(th.tensor([1.0, 1.7]))

        # Role R0 translation box (Kr=1): cen=(1.0, 0.0), w=(0.4, 0.4)
        # R1 is inverse of R0 so its box is automatically cen=(-1.0, 0.0), w=(0.4, 0.4)
        assert mod.role_centers is not None and mod.role_raw_widths is not None
        mod.role_centers[0, 0] = th.tensor([1.0, 0.0])
        mod.role_raw_widths[0, 0] = _inv_softplus(th.tensor([0.4, 0.4]))

    run_dir = root / "runs" / "hand_toy"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
    th.save({"state_dict": state, "args": cell_args, "epoch": 1, "val_mrr": 1.0}, run_dir / "best.pt")
    (run_dir / "results.json").write_text(
        json.dumps({"name": "hand_toy", "args": cell_args, "best_epoch": 1, "tested_epoch": 1}, indent=2)
    )

    # Build test_ded split:
    # 1p query q_1p = (0, (0,)):
    #   Region & beam scores: only entity 1 (I1) has score +0.4 > 0; all others <= -0.6.
    #   Ground truth: easy = {}, hard = {1} -> C = {1}, H = {1}.
    #   At margin 0.0: A = {1} -> P=1, R=1, F1=1, Jacc=1, |A|=1, EM=1, empty=0, complete=1, sound=1.
    # 2p query q_2p = (0, (0, 0)):
    #   Region score: entity 2 (I2) has +0.8, entity 3 (I3) has +0.5; all others < 0.
    #   Beam score: witness I1 has hop1=+0.4; hop2 to I2 is +0.4 (min=0.4), to I3 is +0.1 (min=0.1).
    #   Ground truth: easy = {2}, hard = {3} -> C = {2, 3}, H = {3}.
    #   At margin 0.0: A_reg = {2, 3}, A_beam = {2, 3} -> exact match on both C and H (filtered: {2,3}\{2}={3}=H).
    #   At margin 0.2: A_reg = {2, 3} (depths 0.8, 0.5 > 0.2), but A_beam = {2} (beam scores 0.4 > 0.2 >= 0.1)!
    data_root = root / "data"
    d = data_root / "test_ded"
    d.mkdir(parents=True, exist_ok=True)
    q_1p = (0, (0,))
    q_2p = (0, (0, 0))
    queries = {
        ("e", ("r",)): {q_1p},
        ("e", ("r", "r")): {q_2p},
    }
    easy = {q_1p: set(), q_2p: {2}}
    hard = {q_1p: {1}, q_2p: {3}}
    for name, obj in (("queries", queries), ("easy-answers", easy), ("hard-answers", hard)):
        with open(d / f"test-{name}.pkl", "wb") as fh:
            pickle.dump(obj, fh)

    return run_dir, tables_path, data_root


def test_hand_built_model_and_margin_monotonicity() -> None:
    """End-to-end test on hand-built model checking exact answer sets, TBox satisfaction, and margin monotonicity."""
    with tempfile.TemporaryDirectory(dir=_tmp_base_dir(), prefix="faith_toy_") as tmp:
        root = Path(tmp)
        run_dir, tables_path, data_root = _build_hand_fixture(root)

        # 1. Run CLI at default margin = 0.0
        out_m0 = root / "faith_m0.json"
        rc = F_mod.main(
            [
                "--run-dir",
                str(run_dir),
                "--ckpt",
                "best",
                "--tables",
                str(tables_path),
                "--data-root",
                str(data_root),
                "--test-cases",
                "test_ded",
                "--device",
                "cpu",
                "--out",
                str(out_m0),
            ]
        )
        assert rc == 0
        res0 = json.loads(out_m0.read_text())

        # Check top-level layout: {case: {shape: {variant: {metric: value}}, "macro": {...}}, "tbox": {...}, "args": {...}}
        assert "test_ded" in res0
        assert "tbox" in res0
        assert "args" in res0
        assert "1p" in res0["test_ded"]
        assert "2p" in res0["test_ded"]
        assert "macro" in res0["test_ded"]

        # At margin 0.0, both 1p and 2p achieve exact match = 1.0 for region and beam on both C(q) and H(q)
        for sh, exp_c_size in (("1p", 1.0), ("2p", 2.0)):
            for v in ("region", "beam"):
                m = res0["test_ded"][sh][v]
                assert m["precision"] == 1.0
                assert m["recall"] == 1.0
                assert m["f1"] == 1.0
                assert m["jaccard"] == 1.0
                assert m["size"] == exp_c_size
                assert m["exact_match"] == 1.0
                assert m["empty_rate"] == 0.0
                assert m["complete"] == 1.0
                assert m["sound"] == 1.0

                mh = res0["test_ded"][sh][f"{v}_hard"]
                assert mh["precision"] == 1.0
                assert mh["recall"] == 1.0
                assert mh["f1"] == 1.0
                assert mh["jaccard"] == 1.0
                assert mh["size"] == 1.0  # H(q_1p)={1}, H(q_2p)={3} (easy {2} filtered out)
                assert mh["exact_match"] == 1.0

        # Check TBox faithfulness: C100 <= C101 is strictly satisfied with margin 0.2
        tbox = res0["tbox"]
        assert tbox["region_viol"] == 0.0
        assert tbox["region_viol_tol1e-4"] == 0.0
        assert tbox["elem_viol"] == 0.0
        assert "gci0" in tbox["by_axiom_type"]
        assert tbox["by_axiom_type"]["gci0"]["region_viol_tol0"] == 0.0
        assert tbox["by_axiom_type"]["gci0"]["region_viol_tol1e-4"] == 0.0

        # 2. Check margin = 0.2 where region and beam diverge on 2p:
        # On 2p: A_region = {2, 3} (scores 0.8, 0.5 > 0.2), A_beam = {2} (scores 0.4 > 0.2 > 0.1).
        # Since easy={2} and hard={3}:
        # For beam against C={2,3}: A={2} -> P=1.0, R=0.5, F1=2/3, Jacc=0.5, |A|=1, EM=0, sound=1, complete=0
        # For beam against H={3} (filtered: A\{2} = {}): |A_filt|=0, empty_rate=1.0, P=0, R=0, F1=0
        out_m02 = root / "faith_m02.json"
        F_mod.main(
            [
                "--run-dir",
                str(run_dir),
                "--tables",
                str(tables_path),
                "--data-root",
                str(data_root),
                "--test-cases",
                "test_ded",
                "--margin",
                "0.2",
                "--device",
                "cpu",
                "--out",
                str(out_m02),
            ]
        )
        res02 = json.loads(out_m02.read_text())
        b2p_c = res02["test_ded"]["2p"]["beam"]
        b2p_h = res02["test_ded"]["2p"]["beam_hard"]
        assert b2p_c["size"] == 1.0
        assert b2p_c["precision"] == 1.0
        assert b2p_c["recall"] == 0.5
        assert math.isclose(b2p_c["f1"], 2.0 / 3.0, rel_tol=1e-6)
        assert b2p_c["sound"] == 1.0
        assert b2p_c["complete"] == 0.0
        assert b2p_h["size"] == 0.0
        assert b2p_h["empty_rate"] == 1.0
        assert b2p_h["recall"] == 0.0

        # 3. Check margin monotonicity across m in [-0.2, 0.0, 0.2, 0.6, 1.0]:
        # |A| (size) must be non-increasing in m and empty_rate must be non-decreasing in m
        margins = [-0.2, 0.0, 0.2, 0.6, 1.0]
        prev_res = None
        for m_val in margins:
            out_m = root / f"faith_m_{m_val}.json"
            F_mod.main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--tables",
                    str(tables_path),
                    "--data-root",
                    str(data_root),
                    "--test-cases",
                    "test_ded",
                    "--margin",
                    str(m_val),
                    "--device",
                    "cpu",
                    "--out",
                    str(out_m),
                ]
            )
            cur_res = json.loads(out_m.read_text())
            if prev_res is not None:
                for sh in ("1p", "2p", "macro"):
                    for v in ("region", "beam", "region_hard", "beam_hard"):
                        assert cur_res["test_ded"][sh][v]["size"] <= prev_res["test_ded"][sh][v]["size"] + 1e-9
                        assert cur_res["test_ded"][sh][v]["empty_rate"] >= prev_res["test_ded"][sh][v]["empty_rate"] - 1e-9
            prev_res = cur_res
        # At m = 1.0 (> max score 0.8), all sets are empty
        for sh in ("1p", "2p", "macro"):
            for v in ("region", "beam"):
                assert prev_res["test_ded"][sh][v]["size"] == 0.0
                assert prev_res["test_ded"][sh][v]["empty_rate"] == 1.0
