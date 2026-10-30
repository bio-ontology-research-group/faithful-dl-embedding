"""Unit and end-to-end tests for Task 3: ABox evaluator, metrics, and runner branch."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Dict, Iterator, List, Optional, Tuple
import uuid

import numpy as np
import pytest
import torch

from experiments.abox_eval import (
    abox_modelness,
    count_missing_entities,
    fmax,
    hierarchy_violation_rate,
    ranking,
    score_candidates,
    three_way_metrics,
)
from experiments.config import methods_for_dataset
from experiments.run import execute_run, run_filename


@pytest.fixture()
def scratch_dir() -> Iterator[Path]:
    """Provide an isolated temporary directory under ``~/tmp/`` (never ``/tmp``)."""
    base = Path("~/tmp/gfe-test-abox").expanduser() / f"run_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


class _ScriptedABoxModule(torch.nn.Module):
    """Scripted module with known (individual, class) score table and margin-dependent assertion losses."""

    def __init__(
        self,
        score_table: np.ndarray,
        margin: float = 0.05,
        gamma: float = 0.02,
        support_opa: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_table", torch.as_tensor(score_table, dtype=torch.float32)
        )
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.margin = float(margin)
        self.gamma = float(gamma)
        self.support_opa = bool(support_opa)
        self.seen_margins: List[Tuple[float, float]] = []
        self.max_batch_seen: int = 0

    def forward(self, data: torch.Tensor, gci_name: str, neg: bool = False):
        self.max_batch_seen = max(self.max_batch_seen, int(data.shape[0]))
        self.seen_margins.append((float(self.margin), float(self.gamma)))
        if gci_name == "class_assertion":
            ind_idx = data[:, 0].long()
            cls_idx = data[:, 1].long()
            base = self._table[ind_idx, cls_idx]
            # Subtract (margin + gamma) so strict (margin=0, gamma=0) loss equals base
            return torch.relu(base - float(self.margin) - float(self.gamma))
        if gci_name == "object_property_assertion":
            if not self.support_opa:
                return NotImplementedError()
            # First row has strict loss 0.0, subsequent rows have strict loss 0.01
            n = int(data.shape[0])
            strict = torch.full((n,), 0.01, dtype=torch.float32, device=data.device)
            if n > 0:
                strict[0] = 0.0
            return torch.relu(strict - float(self.margin) - float(self.gamma))
        raise NotImplementedError(f"Unsupported gci_name {gci_name!r}")


class _ScriptedRankClassesModule(torch.nn.Module):
    """Scripted module exposing ``rank_classes(ind_idx)`` like ``CanonicalMixtureModule``."""

    def __init__(self, score_table: np.ndarray) -> None:
        super().__init__()
        self._table = torch.as_tensor(score_table, dtype=torch.float32)
        self.n_classes = int(score_table.shape[1])
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.rank_classes_calls: int = 0

    def rank_classes(self, ind_idx: torch.Tensor) -> torch.Tensor:
        self.rank_classes_calls += 1
        return self._table.index_select(0, ind_idx.cpu()).to(ind_idx.device)

    def forward(self, data: torch.Tensor, gci_name: str, neg: bool = False):
        raise AssertionError("score_candidates should use rank_classes when available")


class _ScriptedABoxModel:
    """Lightweight scripted mOWL model wrapper for unit testing ``abox_eval``."""

    def __init__(
        self,
        module: torch.nn.Module,
        individual_index_dict: Dict[str, int],
        class_index_dict: Dict[str, int],
        training_datasets: Optional[Dict[str, torch.Tensor]] = None,
        margin: float = 0.05,
    ) -> None:
        self.module = module
        self.individual_index_dict = dict(individual_index_dict)
        self.class_index_dict = dict(class_index_dict)
        self.training_datasets = training_datasets or {}
        self.margin = float(margin)
        self.device = "cpu"


def test_filter_keeps_target() -> None:
    """Verify masking rules (KB entailed, other held-out positives, skip list), target never masked, and pessimistic ties."""
    # 2 individuals, 5 candidates C0..C4
    # Individual 0 scores: C0=0.1, C1=0.2, C2=0.3, C3=0.3, C4=0.9
    # Individual 1 scores: C0=0.5, C1=0.5, C2=0.5, C3=0.8, C4=0.1
    scores = np.array(
        [
            [0.1, 0.2, 0.3, 0.3, 0.9],
            [0.5, 0.5, 0.5, 0.8, 0.1],
        ],
        dtype=np.float64,
    )

    # Suppose for ind 0:
    # - C0 is KB-entailed (masked via skip_mask)
    # - C1 and C2 are both held-out positives in the same split
    # - Even if skip_mask accidentally marks target C2 as True, target C2 must NEVER be masked!
    skip_mask = np.array(
        [
            [True, False, True, False, False],  # C0 entailed; C2 also in skip_mask
            [False, False, False, False, True],  # For ind 1, C4 is in skip list
        ],
        dtype=bool,
    )
    positives = [
        (0, 1),  # ind 0, target C1 (score 0.2): C0 masked (skip), C2 masked (other pos) -> unmasked {C1:0.2, C3:0.3, C4:0.9} -> rank 1
        (0, 2),  # ind 0, target C2 (score 0.3): C0 masked (skip), C1 masked (other pos), C2 kept! -> unmasked {C2:0.3, C3:0.3, C4:0.9} -> tied with C3 -> pessimistic rank 2
        (1, 0),  # ind 1, target C0 (score 0.5): C4 masked (skip) -> unmasked {C0:0.5, C1:0.5, C2:0.5, C3:0.8} -> 3-way tie -> pessimistic rank 3
    ]

    res = ranking(scores, positives, skip_mask=skip_mask)
    assert res["ranks"] == [1, 2, 3]
    expected_mrr = (1.0 + 0.5 + 1.0 / 3.0) / 3.0
    assert res["mrr"] == pytest.approx(expected_mrr)
    assert res["hits@1"] == pytest.approx(1.0 / 3.0)
    assert res["hits@5"] == pytest.approx(1.0)
    assert res["hits@10"] == pytest.approx(1.0)
    assert res["coverage"] == pytest.approx(1.0)
    assert res["n_positives_total"] == 3
    assert res["n_positives_rankable"] == 3
    assert res["n_positives_unrankable"] == 0
    assert res["all_mrr"] == pytest.approx(expected_mrr)


def test_coverage_and_unrankable_positives() -> None:
    """Verify rankable-only vs all-positives metrics when held-out positives fall outside candidates."""
    # 1 individual, 4 candidates C0..C3
    scores = np.array([[0.2, 0.1, 0.4, 0.8]], dtype=np.float64)
    # 2 rankable positives: (0, 1) has rank 1; (0, 0) has rank 1 after masking (0, 1)
    # 2 unrankable positives: one passed with cand_idx=None, one via n_unrankable=1
    positives = [(0, 1), (0, 0), (0, None)]
    res = ranking(scores, positives, skip_mask=None, n_unrankable=1)

    assert res["n_positives_rankable"] == 2
    assert res["n_positives_unrankable"] == 2
    assert res["n_positives_total"] == 4
    assert res["coverage"] == pytest.approx(0.5)

    # Both rankable positives achieve rank 1 (since each masks the other)
    assert res["ranks"] == [1, 1]
    assert res["mrr"] == pytest.approx(1.0)
    assert res["hits@1"] == pytest.approx(1.0)
    assert res["rankable_only"]["mrr"] == pytest.approx(1.0)

    # In all-positives, the 2 unrankable positives receive rank = n_candidates + 1 = 5
    expected_all_mrr = (1.0 + 1.0 + 0.2 + 0.2) / 4.0
    assert res["all_mrr"] == pytest.approx(expected_all_mrr)
    assert res["all_hits@1"] == pytest.approx(0.5)
    assert res["all_hits@5"] == pytest.approx(1.0)
    assert res["all_positives"]["mrr"] == pytest.approx(expected_all_mrr)


def test_fmax_hand_computed() -> None:
    """Verify CAFA-style protein-centric Fmax on a hand-computed hierarchy case."""
    # Hierarchy: C0 ⊑ C1 ⊑ C2, and C3 ⊑ C2 (4 candidates: 0, 1, 2, 3)
    hierarchy = {
        0: [0, 1, 2],
        1: [1, 2],
        2: [2],
        3: [2, 3],
    }
    # Protein 0 true leaf: {0} -> ancestor-closed T_0 = {0, 1, 2} (|T_0| = 3)
    # Protein 1 true leaf: {3} -> ancestor-closed T_1 = {2, 3}    (|T_1| = 2)
    true_sets = [[0], [3]]

    # Raw scores (lower = more plausible):
    # Protein 0: C0=0.1, C1=0.8, C2=0.9, C3=0.9
    #   After upward propagation (prop_score[a] = min_{c <= a} score[c]):
    #   prop_scores[0] = [0.1, 0.1, 0.1, 0.9]
    # Protein 1: C0=0.9, C1=0.4, C2=0.9, C3=0.2
    #   prop_scores[1] = [0.9, 0.4, 0.2, 0.2]
    scores = np.array(
        [
            [0.1, 0.8, 0.9, 0.9],
            [0.9, 0.4, 0.9, 0.2],
        ],
        dtype=np.float64,
    )

    # Hand-check thresholds:
    # - tau = 0.1:
    #   P_0 = {0, 1, 2} (tp=3, |P_0|=3, pr_0=1.0, rc_0=1.0)
    #   P_1 = empty     (tp=0, |P_1|=0, rc_1=0.0)
    #   M(0.1) = 1 -> P = 1.0, R = (1.0 + 0.0)/2 = 0.5 -> F1 = 2*1.0*0.5 / 1.5 = 2/3
    # - tau = 0.2:
    #   P_0 = {0, 1, 2} (tp=3, pr_0=1.0, rc_0=1.0)
    #   P_1 = {2, 3}    (tp=2, pr_1=1.0, rc_1=1.0)
    #   M(0.2) = 2 -> P = 1.0, R = 1.0 -> F1 = 1.0!
    val = fmax(scores, true_sets, hierarchy, n_thresholds=100)
    assert val == pytest.approx(1.0)

    # Imperfect prediction case: swap C0 and C3 scores for Protein 1 so C1 (0.2) is predicted before C3 (0.6)
    # Protein 0: prop_scores[0] = [0.1, 0.1, 0.1, 0.9], T_0 = {0, 1, 2}
    # Protein 1: raw = [0.9, 0.2, 0.9, 0.6] -> prop_scores[1] = [0.9, 0.2, 0.2, 0.6], T_1 = {2, 3}
    # At tau = 0.2:
    #   P_0 = {0, 1, 2} -> pr_0 = 1.0, rc_0 = 1.0
    #   P_1 = {1, 2}    -> tp_1 = |{2}| = 1 -> pr_1 = 1/2 = 0.5, rc_1 = 1/2 = 0.5
    #   P(0.2) = 0.75, R(0.2) = 0.75 -> F1(0.2) = 0.75
    # At tau = 0.6:
    #   P_0 = {0, 1, 2} -> pr_0 = 1.0, rc_0 = 1.0
    #   P_1 = {1, 2, 3} -> tp_1 = |{2, 3}| = 2 -> pr_1 = 2/3, rc_1 = 1.0
    #   P(0.6) = (1.0 + 2/3)/2 = 5/6, R(0.6) = 1.0 -> F1(0.6) = 2*(5/6)*1 / (11/6) = 10/11
    scores_imperfect = np.array(
        [
            [0.1, 0.8, 0.9, 0.9],
            [0.9, 0.2, 0.9, 0.6],
        ],
        dtype=np.float64,
    )
    val_imp = fmax(scores_imperfect, true_sets, hierarchy, n_thresholds=100)
    assert val_imp == pytest.approx(10.0 / 11.0)


def test_hierarchy_violation_sampling() -> None:
    """Verify hierarchy_violation_rate ignores G==G', respects 1e-6 tolerance, and returns None below 50 pairs."""
    # Null rule when < 50 (a, G ⊑ G') instances exist (e.g. 10 individuals x 4 strict pairs = 40 < 50)
    small_scores = np.zeros((10, 5), dtype=np.float64)
    small_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 0), (1, 1)]
    assert hierarchy_violation_rate(small_scores, small_pairs) is None

    # 25 individuals x 2 strict pairs (0 -> 1, 1 -> 2) = 50 instances (>= 50)
    # Include reflexive pairs (0, 0), (1, 1), (2, 2) to verify they are filtered out.
    n_ind = 25
    scores = np.zeros((n_ind, 3), dtype=np.float64)
    # For the first 10 individuals, make (0, 1) a violation: score(a, 0) = 0.1 < score(a, 1) = 0.5
    # and make (1, 2) satisfied: score(a, 1) = 0.5 >= score(a, 2) = 0.2
    scores[:10, 0] = 0.1
    scores[:10, 1] = 0.5
    scores[:10, 2] = 0.2
    # For the remaining 15 individuals, make score(a, 0) = score(a, 1) - 0.5e-6 (within 1e-6 tol -> NOT a violation)
    scores[10:, 0] = 0.5 - 0.5e-6
    scores[10:, 1] = 0.5
    scores[10:, 2] = 0.5

    pairs_with_reflexive = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
    hvr = hierarchy_violation_rate(scores, pairs_with_reflexive, seed=0)
    # Exactly 10 violations out of 50 (a, G ⊑ G') instances -> 10 / 50 = 0.20
    assert hvr == pytest.approx(10.0 / 50.0)


def test_three_way_metrics_both_label_columns() -> None:
    """Verify three-way metrics on KB label and label_gold, including < 50 null rule."""
    # 20 individuals x 10 candidates = 200 entries:
    # 60 entailed (+), 80 unknown (?), 60 contradicted (-)
    scores = np.empty((20, 10), dtype=np.float64)
    labels_kb = np.full((20, 10), "unknown", dtype=object)
    labels_gold = np.full((20, 10), "unknown", dtype=object)

    flat_scores = scores.reshape(-1)
    flat_kb = labels_kb.reshape(-1)
    flat_gold = labels_gold.reshape(-1)

    # Under KB labels: perfect ordering (+ has 0.1, ? has 0.5, - has 0.9)
    flat_kb[:60] = "entailed"
    flat_kb[60:140] = "unknown"
    flat_kb[140:] = "contradicted"
    flat_scores[:60] = 0.1
    flat_scores[60:140] = 0.5
    flat_scores[140:] = 0.9

    # Under gold labels: invert + and - on the same scores so ordering is completely reversed
    flat_gold[:60] = "contradicted_gold"
    flat_gold[60:140] = "unknown"
    flat_gold[140:] = "entailed_gold"

    res_kb = three_way_metrics(scores, labels_kb, seed=0)
    assert res_kb["n_pos"] == 60
    assert res_kb["n_und"] == 80
    assert res_kb["n_neg"] == 60
    assert res_kb["auc_pos_neg"] == pytest.approx(1.0)
    assert res_kb["auc_pos_und"] == pytest.approx(1.0)
    assert res_kb["pair_fvr"] == pytest.approx(0.0)
    assert res_kb["fvr"] == pytest.approx(0.0)

    res_gold = three_way_metrics(scores, labels_gold, seed=0)
    assert res_gold["n_pos"] == 60
    assert res_gold["n_und"] == 80
    assert res_gold["n_neg"] == 60
    assert res_gold["auc_pos_neg"] == pytest.approx(0.0)
    assert res_gold["auc_pos_und"] == pytest.approx(0.0)
    assert res_gold["pair_fvr"] == pytest.approx(1.0)
    assert res_gold["fvr"] == pytest.approx(1.0)

    # B1 case: 0 contradicted entries (< 50), 60 entailed, 140 unknown
    labels_no_neg = labels_kb.copy()
    labels_no_neg.reshape(-1)[140:] = "unknown"
    res_no_neg = three_way_metrics(scores, labels_no_neg, seed=0)
    assert res_no_neg["n_neg"] == 0
    assert res_no_neg["auc_pos_neg"] is None
    assert res_no_neg["pair_fvr"] is None
    assert res_no_neg["fvr"] is None
    # AUC(+/?) is still defined because n_pos >= 50 and n_und >= 50
    assert res_no_neg["auc_pos_und"] == pytest.approx(1.0)


def test_abox_modelness_margin_restore() -> None:
    """Verify abox_modelness sets margin/gamma to 0 during evaluation and restores them afterwards."""
    # 3 individuals x 2 classes:
    # At margin=0, gamma=0: (0,0)->0.0, (1,0)->0.0, (2,1)->0.02 (not <= 1e-6)
    # At margin=0.05, gamma=0.02: all three would have loss 0.0 if margin were not zeroed!
    score_table = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.5],
            [0.5, 0.02],
        ],
        dtype=np.float32,
    )
    ca_ds = torch.tensor([[0, 0], [1, 0], [2, 1]], dtype=torch.long)
    opa_ds = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)

    module = _ScriptedABoxModule(
        score_table, margin=0.05, gamma=0.02, support_opa=True
    )
    module.train()
    model = _ScriptedABoxModel(
        module=module,
        individual_index_dict={"a0": 0, "a1": 1, "a2": 2},
        class_index_dict={"C0": 0, "C1": 1},
        training_datasets={"class_assertion": ca_ds, "object_property_assertion": opa_ds},
        margin=0.05,
    )

    # 2/3 class assertions satisfied + 1/2 object property assertions satisfied = 3/5 = 0.6
    frac = abox_modelness(model)
    assert frac == pytest.approx(3.0 / 5.0)
    # Verify evaluation happened with margin=0.0 and gamma=0.0
    assert all(m == 0.0 and g == 0.0 for m, g in module.seen_margins)
    # Verify original margins and training mode were restored
    assert module.margin == pytest.approx(0.05)
    assert module.gamma == pytest.approx(0.02)
    assert model.margin == pytest.approx(0.05)
    assert module.training is True


def test_score_candidates_batching_missing_and_rank_classes() -> None:
    """Verify score_candidates missing entity +inf assignment, chunking, and rank_classes path."""
    table = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    mod = _ScriptedABoxModule(table, margin=0.0, gamma=0.0)
    model = _ScriptedABoxModel(
        module=mod,
        individual_index_dict={"a0": 0, "a1": 1},
        class_index_dict={"C0": 0, "C1": 1, "C2": 2},
    )

    individuals = ["a0", "a_missing", "a1"]
    candidates = ["C2", "C_missing", "C0"]

    missing_info = count_missing_entities(model, individuals, candidates)
    assert missing_info == {"n_missing_individuals": 1, "n_missing_candidates": 1}

    scores = score_candidates(model, individuals, candidates, batch_size=2)
    assert scores.shape == (3, 3)
    assert mod.max_batch_seen <= 2
    assert scores[0, 0] == pytest.approx(0.3)
    assert scores[0, 2] == pytest.approx(0.1)
    assert scores[2, 0] == pytest.approx(0.6)
    assert scores[2, 2] == pytest.approx(0.4)
    # Missing individual row and missing candidate column must be +inf
    assert np.all(np.isinf(scores[1, :]))
    assert np.all(np.isinf(scores[:, 1]))

    # Verify rank_classes fast path when present on module
    rc_mod = _ScriptedRankClassesModule(table)
    rc_model = _ScriptedABoxModel(
        module=rc_mod,
        individual_index_dict={"a0": 0, "a1": 1},
        class_index_dict={"C0": 0, "C1": 1, "C2": 2},
    )
    rc_scores = score_candidates(rc_model, individuals, candidates)
    assert rc_mod.rank_classes_calls >= 1
    assert np.allclose(rc_scores[[0, 2]][:, [0, 2]], scores[[0, 2]][:, [0, 2]])
    assert np.all(np.isinf(rc_scores[1, :]))
    assert np.all(np.isinf(rc_scores[:, 1]))


def test_run_elbe_on_helis_smoke(scratch_dir: Path) -> None:
    """End-to-end smoke test: ELBE on helis for 3 epochs through run.py writes a JSON with every key."""
    assert "canonical-mixture" in methods_for_dataset("helis")
    assert "canonical-mixture" in methods_for_dataset("yeast-function")
    assert "canonical-mixture" not in methods_for_dataset("pizza")

    out_dir = scratch_dir / "smoke_out"
    runs_dir = scratch_dir / "smoke_runs"

    payload = execute_run(
        dataset="helis",
        method="elbe",
        dim=16,
        seed=0,
        epochs=3,
        out_dir=out_dir,
        runs_dir=runs_dir,
    )

    json_path = out_dir / run_filename("helis", "elbe", 16, 0)
    assert json_path.exists()
    disk_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert disk_payload == payload

    for top_key in ("config", "metrics", "timings", "env", "validation_curve"):
        assert top_key in disk_payload

    cfg = disk_payload["config"]
    assert cfg["dataset"] == "helis"
    assert cfg["task"] == "abox"
    assert cfg["method"] == "elbe"
    assert cfg["epochs"] == 3
    assert isinstance(cfg["validation_curve"], list) and len(cfg["validation_curve"]) >= 1

    m = disk_payload["metrics"]
    required_metric_keys = (
        "rank_mrr",
        "rank_hits@1",
        "rank_hits@5",
        "rank_hits@10",
        "rank_auc",
        "rank_all_mrr",
        "rank_all_hits@1",
        "rank_all_hits@5",
        "rank_all_hits@10",
        "rank_all_auc",
        "rank_coverage",
        "n_positives_total",
        "n_positives_rankable",
        "n_positives_unrankable",
        "ranking_rankable",
        "ranking_all",
        "fmax",
        "hierarchy_violation_rate",
        "abox_modelness",
        "faithfulness_kb",
        "faithfulness_gold",
        "auc_pos_neg",
        "auc_pos_und",
        "pair_fvr",
        "fvr",
        "auc_pos_neg_gold",
        "auc_pos_und_gold",
        "pair_fvr_gold",
        "fvr_gold",
        "n_missing_individuals",
        "n_missing_candidates",
        "val_mrr_curve",
        "best_val_mrr",
        "best_val_epoch",
        "param_count",
        "mean_epoch_time_s",
        "total_train_time_s",
    )
    for k in required_metric_keys:
        assert k in m, f"Missing metric key {k!r}"

    assert m["fmax"] is None  # fmax is yeast-function only
    assert 0.0 <= float(m["rank_mrr"]) <= 1.0
    assert 0.0 <= float(m["abox_modelness"]) <= 1.0
    assert 0.0 <= float(m["hierarchy_violation_rate"]) <= 1.0
    # Spec §3.2 Amendment: KB label on helis has 0 contradicted pairs (auc_pos_neg is None, auc_pos_und is float);
    # label_gold has >600k contradicted pairs so all four three-way metrics are floats.
    assert m["faithfulness_kb"]["auc_pos_und"] is not None
    assert m["faithfulness_kb"]["auc_pos_neg"] is None
    assert m["faithfulness_gold"]["auc_pos_neg"] is not None
    assert m["faithfulness_gold"]["auc_pos_und"] is not None
    assert m["faithfulness_gold"]["pair_fvr"] is not None
    assert m["faithfulness_gold"]["fvr"] is not None
