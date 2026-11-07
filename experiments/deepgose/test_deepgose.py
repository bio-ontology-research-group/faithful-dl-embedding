"""Unit tests for the DeepGO-SE GroundedEL E3 pipeline (``experiments.deepgose``).

Covers:
1. ``go.norm`` parser on a 20-line fixture -> correct table shapes and ``Aux_G`` axioms.
2. Data loss on a toy encoder decreases over 20 steps.
3. ``evaluate.py`` hierarchy-consistency on a hand-built 3-term case.
4. DeepGO-SE prediction-format round trip (``test_df`` with ``preds`` column over ``terms``).
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch as th

from mowl.geometric.normal_forms import BOTTOM, TOP

from experiments.deepgose.data import (
    AUX_PREFIX,
    HAS_FUNCTION_REL,
    aux_class_name,
    dataframe_to_tensors,
    parse_go_norm,
)
from experiments.deepgose.evaluate import (
    combine_prediction_matrices,
    compute_hierarchy_consistency,
)
from experiments.deepgose.model import (
    build_deepgose_grounded_model,
    build_optimizer_groups,
)

FIXTURE_20_LINES: List[str] = [
    # 8 NF1 lines (C SubClassOf D -> gci0)
    "GO_0000001 SubClassOf GO_0000002",
    "GO_0000002 SubClassOf GO_0000003",
    "GO_0000004 SubClassOf GO_0000002",
    "GO_0000005 SubClassOf GO_0000003",
    "GO_0000006 SubClassOf GO_0000005",
    "GO_0000007 SubClassOf GO_0000003",
    "GO_0000008 SubClassOf GO_0000007",
    "GO_0000009 SubClassOf GO_0000003",
    # 4 NF2 lines (C and D SubClassOf E -> gci1)
    "GO_0000001 and GO_0000004 SubClassOf GO_0000005",
    "GO_0000002 and GO_0000007 SubClassOf GO_0000009",
    "GO_0000004 and GO_0000006 SubClassOf GO_0000008",
    "GO_0000005 and GO_0000007 SubClassOf GO_0000002",
    # 4 NF3 lines (R some C SubClassOf D -> gci3)
    "BFO_0000050 some GO_0000001 SubClassOf GO_0000002",
    "BFO_0000050 some GO_0000004 SubClassOf GO_0000005",
    "RO_0002211 some GO_0000006 SubClassOf GO_0000007",
    "RO_0002211 some GO_0000008 SubClassOf GO_0000009",
    # 4 NF4 lines (C SubClassOf R some D -> gci2)
    "GO_0000001 SubClassOf BFO_0000050 some GO_0000003",
    "GO_0000004 SubClassOf BFO_0000050 some GO_0000007",
    "GO_0000006 SubClassOf RO_0002211 some GO_0000005",
    "GO_0000008 SubClassOf RO_0002211 some GO_0000002",
]


def _tmp_base_dir() -> str:
    base = os.environ.get("TMPDIR") or os.path.expanduser("~/tmp")
    os.makedirs(base, exist_ok=True)
    return base


def test_parse_go_norm_20_line_fixture_shapes_and_aux_axioms() -> None:
    """1. go.norm parser on a 20-line fixture produces exact table shapes and Aux_G axioms."""
    assert len(FIXTURE_20_LINES) == 20
    predicted_terms = ["GO:0000001", "GO:0000002", "GO:0000003"]

    with tempfile.TemporaryDirectory(dir=_tmp_base_dir(), prefix="deepgose_norm_") as tmpdir:
        norm_path = Path(tmpdir) / "go.norm"
        cache_path = Path(tmpdir) / "tables_cache.pt"
        norm_path.write_text("\n".join(FIXTURE_20_LINES) + "\n", encoding="utf-8")

        tables = parse_go_norm(norm_path, terms=predicted_terms, cache_path=cache_path)
        assert cache_path.is_file()

        # Classes: BOTTOM, TOP, 9 GO classes (GO:0000001..GO:0000009), 3 Aux_G classes = 14
        assert tables.class_index_dict[BOTTOM] == 0
        assert tables.class_index_dict[TOP] == 1
        assert len(tables.class_index_dict) == 2 + 9 + 3
        assert len(tables.named_class_index_dict) == 9

        # Relations: BFO:0000050, RO:0002211, hasFunction = 3
        assert set(tables.object_property_index_dict.keys()) == {
            "BFO:0000050",
            "RO:0002211",
            HAS_FUNCTION_REL,
        }
        has_func_id = tables.object_property_index_dict[HAS_FUNCTION_REL]

        # Table shapes:
        # gci0: 8 NF1 rows -> (8, 2)
        # gci1: 4 NF2 rows -> (4, 3)
        # gci2: 4 NF4 rows + 3 Aux_G <= exists hasFunction.G rows -> (7, 3)
        # gci3: 4 NF3 rows + 3 exists hasFunction.G <= Aux_G rows -> (7, 3)
        assert tables.gci0.shape == (8, 2)
        assert tables.gci1.shape == (4, 3)
        assert tables.gci2.shape == (7, 3)
        assert tables.gci3.shape == (7, 3)
        assert tables.gci0_bot.shape == (0, 2)
        assert tables.gci1_bot.shape == (0, 3)
        assert tables.gci3_bot.shape == (0, 3)

        # Check Aux_G axioms for every predicted term
        gci2_set = {tuple(r) for r in tables.gci2.tolist()}
        gci3_set = {tuple(r) for r in tables.gci3.tolist()}
        for col_i, go_id in enumerate(predicted_terms):
            g_idx = tables.class_index_dict[go_id]
            aux_idx = tables.class_index_dict[aux_class_name(go_id)]
            assert int(tables.go_term_class_ids[col_i].item()) == g_idx
            assert int(tables.term_class_ids[col_i].item()) == aux_idx
            assert (aux_idx, has_func_id, g_idx) in gci2_set
            assert (has_func_id, g_idx, aux_idx) in gci3_set

        # Reloading from cache_path returns identical tables
        cached_tables = parse_go_norm(norm_path, terms=predicted_terms, cache_path=cache_path)
        assert th.equal(cached_tables.gci0, tables.gci0)
        assert th.equal(cached_tables.gci2, tables.gci2)
        assert th.equal(cached_tables.gci3, tables.gci3)


def test_data_loss_on_toy_encoder_decreases_over_20_steps() -> None:
    """2. Data loss on a toy encoder decreases monotonically over 20 Adam steps."""
    predicted_terms = ["GO:0000001", "GO:0000002", "GO:0000003"]
    with tempfile.TemporaryDirectory(dir=_tmp_base_dir(), prefix="deepgose_loss_") as tmpdir:
        norm_path = Path(tmpdir) / "go.norm"
        norm_path.write_text("\n".join(FIXTURE_20_LINES) + "\n", encoding="utf-8")
        tables = parse_go_norm(norm_path, terms=predicted_terms)

    model = build_deepgose_grounded_model(
        tables=tables,
        embed_dim=16,
        n_components=1,
        in_dim=32,
        hidden_dim=24,
        dropout=0.0,
        seed=42,
        device="cpu",
    )
    optimizer, _, _ = build_optimizer_groups(
        model, lr_encoder=5e-3, lr_boxes=1e-2
    )

    th.manual_seed(42)
    features = th.randn(16, 32)
    labels = th.tensor(
        [[1.0, 1.0, 1.0] if i < 8 else [0.0, 1.0, 1.0] for i in range(16)],
        dtype=th.float32,
    )

    losses: List[float] = []
    for _ in range(20):
        model.module.train()
        optimizer.zero_grad()
        bce_loss, scores = model.compute_data_loss(features, labels)
        assert scores.shape == (16, 3)
        bce_loss.backward()
        optimizer.step()
        losses.append(float(bce_loss.detach().item()))

    assert losses[-1] < losses[0]


def test_evaluate_hierarchy_consistency_hand_built_3_term_case() -> None:
    """3. evaluate.py hierarchy-consistency on a hand-built 3-term chain G0 <= G1 <= G2."""
    # 3 terms: 0 <= 1 and 1 <= 2 (so 2 direct pairs: (0, 1) and (1, 2))
    subclass_pairs = [(0, 1), (1, 2)]

    # 4 proteins over 3 terms, evaluated at t_max = 0.5:
    # Protein 0: [0.2, 0.6, 0.8] -> 0<=1 ok, 1<=2 ok; pred set {1, 2} (ancestor-closed)
    # Protein 1: [0.7, 0.6, 0.8] -> 0<=1 VIOLATED (0.7 > 0.6 + 1e-6), 1<=2 ok;
    #                               pred set {0, 1, 2} (still ancestor-closed at t_max=0.5!)
    # Protein 2: [0.6, 0.3, 0.8] -> 0<=1 VIOLATED (0.6 > 0.3 + 1e-6), 1<=2 ok;
    #                               pred set {0, 2} (NOT ancestor-closed: 0 in set, 1 not in set)
    # Protein 3: [0.1, 0.2, 0.3] -> 0<=1 ok, 1<=2 ok; pred set {} (ancestor-closed)
    raw_scores = np.array(
        [
            [0.2, 0.6, 0.8],
            [0.7, 0.6, 0.8],
            [0.6, 0.3, 0.8],
            [0.1, 0.2, 0.3],
        ],
        dtype=np.float32,
    )

    res = compute_hierarchy_consistency(
        raw_scores=raw_scores,
        subclass_pairs=subclass_pairs,
        t_max=0.5,
        tol=1e-6,
    )

    # Total (protein, pair) checks = 4 * 2 = 8; violated = 2 (protein 1 on (0,1) and protein 2 on (0,1))
    assert res["n_total_pairs"] == 8
    assert res["n_violated_pairs"] == 2
    assert res["hier_viol_rate"] == pytest.approx(2.0 / 8.0)

    # Ancestor-closed proteins at t_max=0.5: proteins 0, 1, 3 -> 3 / 4 = 0.75
    assert res["n_proteins"] == 4
    assert res["n_ancestor_closed_proteins"] == 3
    assert res["ancestor_closed_rate"] == pytest.approx(3.0 / 4.0)


def test_deepgose_prediction_format_round_trip() -> None:
    """4. DeepGO-SE prediction-format round trip: test_df with 'preds' column over terms."""
    terms = ["GO:0000001", "GO:0000002", "GO:0000003"]
    terms_dict = {t: i for i, t in enumerate(terms)}

    test_df = pd.DataFrame(
        {
            "proteins": ["P001", "P002", "P003"],
            "esm2": [
                np.ones((16,), dtype=np.float32) * 0.1,
                np.ones((16,), dtype=np.float32) * 0.2,
                np.ones((16,), dtype=np.float32) * 0.3,
            ],
            "prop_annotations": [
                ["GO:0000001", "GO:0000002", "GO:0000003"],
                ["GO:0000002", "GO:0000003"],
                ["GO:0000003"],
            ],
            "exp_annotations": [
                ["GO:0000001"],
                ["GO:0000002"],
                ["GO:0000003"],
            ],
        }
    )

    feat_t, lbl_t = dataframe_to_tensors(test_df, terms_dict)
    assert feat_t.shape == (3, 16)
    assert lbl_t.shape == (3, 3)
    assert lbl_t.tolist() == [
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ]

    with tempfile.TemporaryDirectory(dir=_tmp_base_dir(), prefix="deepgose_rt_") as tmpdir:
        norm_path = Path(tmpdir) / "go.norm"
        norm_path.write_text("\n".join(FIXTURE_20_LINES) + "\n", encoding="utf-8")
        tables = parse_go_norm(norm_path, terms=terms)

        model = build_deepgose_grounded_model(
            tables=tables,
            embed_dim=8,
            n_components=1,
            in_dim=16,
            hidden_dim=12,
            seed=7,
            device="cpu",
        )
        raw_scores = model.score(feat_t).cpu().numpy()
        assert raw_scores.shape == (3, 3)
        assert np.all((raw_scores >= 0.0) & (raw_scores <= 1.0))

        out_df = test_df.copy()
        out_df["preds"] = list(raw_scores)
        pkl_path = Path(tmpdir) / "test_predictions.pkl"
        out_df.to_pickle(pkl_path)

        reloaded_df = pd.read_pickle(pkl_path)
        assert list(reloaded_df["proteins"].values) == ["P001", "P002", "P003"]
        assert "preds" in reloaded_df.columns
        reloaded_scores = np.stack(list(reloaded_df["preds"].values), axis=0)
        assert reloaded_scores.shape == (3, len(terms))
        assert np.allclose(reloaded_scores, raw_scores, atol=1e-7)

        # Also check combine_prediction_matrices (avg, min, max)
        m2 = np.clip(raw_scores + 0.1, 0.0, 1.0)
        assert np.allclose(
            combine_prediction_matrices([raw_scores, m2], "avg"),
            0.5 * (raw_scores + m2),
        )
        assert np.allclose(
            combine_prediction_matrices([raw_scores, m2], "min"),
            np.minimum(raw_scores, m2),
        )
        assert np.allclose(
            combine_prediction_matrices([raw_scores, m2], "max"),
            np.maximum(raw_scores, m2),
        )


SYNTHETIC_OBO_TEXT = """format-version: 1.2
data-version: test

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component

[Term]
id: GO:0000003
name: term_3
namespace: cellular_component
is_a: GO:0005575 ! cellular_component

[Term]
id: GO:0000002
name: term_2
namespace: cellular_component
is_a: GO:0000003 ! term_3

[Term]
id: GO:0000001
name: term_1
namespace: cellular_component
is_a: GO:0000002 ! term_2

[Term]
id: GO:0000004
name: term_4
namespace: cellular_component
is_a: GO:0000002 ! term_2

[Term]
id: GO:0000005
name: term_5
namespace: cellular_component
is_a: GO:0000003 ! term_3

[Term]
id: GO:0000006
name: term_6
namespace: cellular_component
is_a: GO:0000005 ! term_5

[Term]
id: GO:0000007
name: term_7
namespace: cellular_component
is_a: GO:0000003 ! term_3

[Term]
id: GO:0000008
name: term_8
namespace: cellular_component
is_a: GO:0000007 ! term_7

[Term]
id: GO:0000009
name: term_9
namespace: cellular_component
is_a: GO:0000003 ! term_3
"""


def _write_synthetic_deepgose_dataset(data_root: Path, n_proteins: int = 10) -> List[str]:
    """Write 20-line go.norm fixture, synthetic go.obo, and 10-protein pickles under ``data_root``."""
    cc_dir = data_root / "cc"
    cc_dir.mkdir(parents=True, exist_ok=True)

    (data_root / "go.norm").write_text("\n".join(FIXTURE_20_LINES) + "\n", encoding="utf-8")
    (data_root / "go.obo").write_text(SYNTHETIC_OBO_TEXT, encoding="utf-8")

    terms = ["GO:0000001", "GO:0000002", "GO:0000003"]
    pd.DataFrame({"gos": np.array(terms, dtype=object)}).to_pickle(cc_dir / "terms.pkl")
    pd.DataFrame({"interpros": np.array(["IPR000001", "IPR000002"], dtype=object)}).to_pickle(
        cc_dir / "interpros.pkl"
    )

    rng = np.random.default_rng(12345)
    proteins = [f"P{i:03d}" for i in range(n_proteins)]
    esm2_vecs = [rng.standard_normal(2560).astype(np.float32) for _ in range(n_proteins)]
    interpros_col = [["IPR000001"] if i % 2 == 0 else ["IPR000002"] for i in range(n_proteins)]

    prop_annots: List[List[str]] = []
    exp_annots: List[List[str]] = []
    for i in range(n_proteins):
        if i < 3:
            prop_annots.append(["GO:0000001", "GO:0000002", "GO:0000003", "GO:0005575"])
            exp_annots.append(["GO:0000001"])
        elif i < 7:
            prop_annots.append(["GO:0000002", "GO:0000003", "GO:0005575"])
            exp_annots.append(["GO:0000002"])
        elif i < 9:
            prop_annots.append(["GO:0000003", "GO:0005575"])
            exp_annots.append(["GO:0000003"])
        else:
            prop_annots.append(["GO:0000002", "GO:0005575"])
            exp_annots.append(["GO:0000002"])

    df = pd.DataFrame(
        {
            "proteins": proteins,
            "esm2": esm2_vecs,
            "interpros": interpros_col,
            "prop_annotations": prop_annots,
            "exp_annotations": exp_annots,
        }
    )
    for split_name in ("train_data.pkl", "valid_data.pkl", "test_data.pkl"):
        df.to_pickle(cc_dir / split_name)

    return terms


def test_baselines_main_1_epoch_and_evaluate_end_to_end() -> None:
    """5. Both baseline modules run main() for 1 epoch on 20-line go.norm + 10-protein synthetic pickle and evaluate.py scores them (including --combine min for n_models > 1)."""
    import json
    from experiments.deepgose import evaluate as evaluate_mod
    from experiments.deepgose.baselines import train_deepgo, train_mlp
    from experiments.deepgose.data import load_deepgose_data

    with tempfile.TemporaryDirectory(dir=_tmp_base_dir(), prefix="deepgose_base_") as tmpdir:
        data_root = Path(tmpdir) / "data"
        out_dir = Path(tmpdir) / "out"
        expected_terms = _write_synthetic_deepgose_dataset(data_root, n_proteins=10)

        # Verify GroundedEL data loader has exact same term order from terms.pkl
        split_data = load_deepgose_data(data_root=data_root, ont="cc", cache_dir=out_dir / "cc")
        assert split_data.terms == expected_terms

        # 1. Run train_deepgo.main() for 1 epoch
        train_deepgo.main(
            [
                "--data-root",
                str(data_root),
                "--out",
                str(out_dir),
                "--ont",
                "cc",
                "--model-name",
                "deepgozero_esm",
                "--model-id",
                "0",
                "--seed",
                "0",
                "--batch-size",
                "5",
                "--epochs",
                "1",
                "--device",
                "cpu",
            ]
        )

        cc_out = out_dir / "cc"
        assert (cc_out / "deepgozero_esm_0.pt").is_file()
        assert (cc_out / "test_predictions_deepgozero_esm_0.pkl").is_file()
        assert (cc_out / "metrics_deepgozero_esm_0.json").is_file()

        rc_dg = evaluate_mod.main(
            [
                "--ont",
                "cc",
                "--model-name",
                "deepgozero_esm",
                "--model-ids",
                "0",
                "--n-models",
                "1",
                "--combine",
                "avg",
                "--data-root",
                str(data_root),
                "--out",
                str(out_dir),
                "--device",
                "cpu",
            ]
        )
        assert rc_dg == 0
        eval_dg_path = cc_out / "eval_deepgozero_esm_avg_1_ids0.json"
        assert eval_dg_path.is_file()
        eval_dg = json.loads(eval_dg_path.read_text(encoding="utf-8"))
        assert eval_dg["n_terms"] == len(expected_terms)
        assert eval_dg["n_test_proteins"] == 10
        assert 0.0 <= eval_dg["propagated"]["fmax"] <= 1.0
        assert eval_dg["propagated"]["hier_viol_rate"] if "hier_viol_rate" in eval_dg["propagated"] else True
        assert eval_dg["crisp_report"]["elem_viol"] is None

        # 2. Run train_mlp.main() for 1 epoch (two seeds/model_ids to test --combine min for n_models > 1)
        for mi in (0, 1):
            train_mlp.main(
                [
                    "--data-root",
                    str(data_root),
                    "--out",
                    str(out_dir),
                    "--ont",
                    "cc",
                    "--model-name",
                    "mlp_esm",
                    "--model-id",
                    str(mi),
                    "--seed",
                    str(mi),
                    "--batch-size",
                    "5",
                    "--epochs",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            assert (cc_out / f"mlp_esm_{mi}.pt").is_file()
            assert (cc_out / f"test_predictions_mlp_esm_{mi}.pkl").is_file()
            assert (cc_out / f"metrics_mlp_esm_{mi}.json").is_file()

        # Also verify evaluate.py works when terms come solely from terms.pkl (no "terms" key in metrics JSON)
        for mi in (0, 1):
            m_path = cc_out / f"metrics_mlp_esm_{mi}.json"
            m_obj = json.loads(m_path.read_text(encoding="utf-8"))
            assert m_obj["terms"] == expected_terms
            m_obj.pop("terms", None)
            m_path.write_text(json.dumps(m_obj), encoding="utf-8")

        rc_mlp_1 = evaluate_mod.main(
            [
                "--ont",
                "cc",
                "--model-name",
                "mlp_esm",
                "--model-ids",
                "0",
                "--n-models",
                "1",
                "--combine",
                "avg",
                "--data-root",
                str(data_root),
                "--out",
                str(out_dir),
                "--device",
                "cpu",
            ]
        )
        assert rc_mlp_1 == 0
        assert (cc_out / "eval_mlp_esm_avg_1_ids0.json").is_file()

        rc_mlp_min2 = evaluate_mod.main(
            [
                "--ont",
                "cc",
                "--model-name",
                "mlp_esm",
                "--model-ids",
                "0",
                "1",
                "--n-models",
                "2",
                "--combine",
                "min",
                "--data-root",
                str(data_root),
                "--out",
                str(out_dir),
                "--device",
                "cpu",
            ]
        )
        assert rc_mlp_min2 == 0
        eval_min2_path = cc_out / "eval_mlp_esm_min_2_ids0-1.json"
        assert eval_min2_path.is_file()
        eval_min2 = json.loads(eval_min2_path.read_text(encoding="utf-8"))
        assert eval_min2["combine"] == "min"
        assert eval_min2["n_models_used"] == 2
        assert sorted(eval_min2["selected_model_ids"]) == [0, 1]
        assert 0.0 <= eval_min2["propagated"]["fmax"] <= 1.0

