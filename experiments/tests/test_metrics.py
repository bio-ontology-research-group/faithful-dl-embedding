"""Tests for experiment evaluation metrics and canonical model cache (Task 7)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import mowl
mowl.init_jvm("8g")

import numpy as np
import pytest
import torch

from mowl.datasets import PathDataset
from mowl.geometric import (
    BOTTOM,
    TOP,
    CanonicalModel,
    HellyCertificate,
    certificate_triangles,
    helly_check,
)
from mowl.models import BoxSquaredEL, ELBE, MultiBoxEL

import experiments.cm_cache as cm_cache_mod
from experiments.cm_cache import (
    cache_path_for_bundle,
    get_canonical_model,
    main as cm_cache_main,
)
from experiments.config import RAW_DATA_ROOT
from experiments.datasets import DatasetBundle, load_dataset
from experiments.metrics import (
    _CPUFilteredRowLookup,
    _compute_fast_head_centric_ranking_metrics,
    _is_named_class,
    build_faithfulness_sets,
    class_regions,
    cost_metrics,
    faithfulness_metrics,
    helly_metrics,
    modelness,
    ranking_metrics,
)

NS = "http://mowl.borg/geo#"
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
T3_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "t3.owl"


@pytest.fixture
def t3_bundle() -> DatasetBundle:
    """Return a DatasetBundle for the T3 fixture ontology."""
    ds = PathDataset(str(T3_FIXTURE_PATH))
    return DatasetBundle(
        name="t3",
        dataset=ds,
        deductive_closure_ontology=None,
        task="satisfaction",
        paths={"ontology": str(T3_FIXTURE_PATH)},
    )


def _write_handmade_t3_multibox(model: MultiBoxEL) -> None:
    """Write the exact 1-D 2-component MultiBoxEL solution of T3 under torch.no_grad().

    Intervals:
    - X1 = [0, 1] u [2, 3]
    - X2 = [0, 1] u [4, 5]
    - X3 = [2, 3] u [4, 5]
    - A  = [0, 1] (duplicated across both components)
    - B  = [4, 5] (duplicated across both components)
    - C  = [2, 3] (duplicated across both components)
    - Z  = [0, 1] (duplicated across both components)
    - owl:Thing = [-1, 6] (duplicated across both components)
    """
    spec: Dict[str, List[Tuple[float, float]]] = {
        f"{NS}X1": [(0.0, 1.0), (2.0, 3.0)],
        f"{NS}X2": [(0.0, 1.0), (4.0, 5.0)],
        f"{NS}X3": [(2.0, 3.0), (4.0, 5.0)],
        f"{NS}A": [(0.0, 1.0), (0.0, 1.0)],
        f"{NS}B": [(4.0, 5.0), (4.0, 5.0)],
        f"{NS}C": [(2.0, 3.0), (2.0, 3.0)],
        f"{NS}Z": [(0.0, 1.0), (0.0, 1.0)],
        OWL_THING: [(-1.0, 6.0), (-1.0, 6.0)],
    }
    with torch.no_grad():
        for iri, comps in spec.items():
            if iri not in model.class_index_dict:
                continue
            idx = model.class_index_dict[iri]
            centers = [(lo + hi) / 2.0 for lo, hi in comps]
            offsets = [(hi - lo) / 2.0 for lo, hi in comps]
            model.module.class_center.weight[idx] = torch.tensor(
                centers, dtype=model.module.class_center.weight.dtype
            )
            model.module.class_offset.weight[idx] = torch.tensor(
                offsets, dtype=model.module.class_offset.weight.dtype
            )


def test_modelness_restores_margin(t3_bundle: DatasetBundle) -> None:
    """Verify modelness temporarily sets margin/gamma to 0 and restores it afterwards."""
    ds = t3_bundle.dataset

    elbe = ELBE(ds, embed_dim=4, margin=0.17)
    assert elbe.module.margin == pytest.approx(0.17)
    m_elbe = modelness(elbe)
    assert elbe.module.margin == pytest.approx(0.17)
    assert "modelness_all" in m_elbe
    assert "modelness_gci0" in m_elbe
    assert "modelness_gci1" in m_elbe
    assert "modelness_gci1_bot" in m_elbe
    for val in m_elbe.values():
        assert 0.0 <= val <= 1.0

    b2 = BoxSquaredEL(ds, embed_dim=4, margin=0.09)
    assert b2.module.gamma == pytest.approx(0.09)
    m_b2 = modelness(b2)
    assert b2.module.gamma == pytest.approx(0.09)
    assert "modelness_all" in m_b2

    # On the hand-set 1-D solution with non-zero margin=-0.25 (strict interior margin that
    # would fail if gamma were not zeroed during measurement), modelness at margin=0 must
    # be 1.0 across all GCIs and must restore gamma=-0.25 afterwards.
    mb = MultiBoxEL(ds, embed_dim=1, n_components=2, margin=-0.25)
    _write_handmade_t3_multibox(mb)
    assert mb.module.gamma == pytest.approx(-0.25)
    m_mb = modelness(mb)
    assert mb.module.gamma == pytest.approx(-0.25)
    assert m_mb["modelness_gci0"] == pytest.approx(1.0)
    assert m_mb["modelness_gci1"] == pytest.approx(1.0)
    assert m_mb["modelness_gci1_bot"] == pytest.approx(1.0)
    assert m_mb["modelness_all"] == pytest.approx(1.0)


def test_helly_metrics_box2el_random_init_violates_t3(t3_bundle: DatasetBundle) -> None:
    """Verify helly_metrics on BoxSquaredEL on T3: random init, positive controls, and unmapped certs."""
    cm = get_canonical_model(t3_bundle)
    certs = certificate_triangles(cm)
    assert len(certs) == 1

    torch.manual_seed(0)
    b2 = BoxSquaredEL(t3_bundle.dataset, embed_dim=4)
    res = helly_metrics(b2, t3_bundle, cm, certs)
    assert res["n_certificates"] == 1
    assert res["n_certificates_unmapped"] == 0
    assert 0.0 <= res["helly_violation_rate"] <= 1.0
    assert 0.0 <= res["degeneracy_rate"] <= 1.0

    # Positive control 1: identical non-degenerate boxes -> helly_violation_rate 1.0, degeneracy 0.0
    with torch.no_grad():
        b2.module.class_center.weight.zero_()
        b2.module.class_offset.weight.fill_(1.0)
    res_ident = helly_metrics(b2, t3_bundle, cm, certs)
    assert res_ident["n_certificates"] == 1
    assert res_ident["n_certificates_unmapped"] == 0
    assert res_ident["helly_violation_rate"] == pytest.approx(1.0)
    assert res_ident["degeneracy_rate"] == pytest.approx(0.0)

    # Positive control 2: all offsets 0 -> degeneracy_rate 1.0
    with torch.no_grad():
        b2.module.class_offset.weight.zero_()
    res_zero_off = helly_metrics(b2, t3_bundle, cm, certs)
    assert res_zero_off["n_certificates"] == 1
    assert res_zero_off["n_certificates_unmapped"] == 0
    assert res_zero_off["degeneracy_rate"] == pytest.approx(1.0)

    # Unmapped certificate handling: excluded from rates, counted in n_certificates_unmapped;
    # ValueError when all certificates are unmapped.
    unmapped_cert = replace(certs[0], c1=(f"{NS}Unmapped1",))
    with torch.no_grad():
        b2.module.class_center.weight.zero_()
        b2.module.class_offset.weight.fill_(1.0)
    res_mixed = helly_metrics(b2, t3_bundle, cm, [certs[0], unmapped_cert])
    assert res_mixed["n_certificates"] == 2
    assert res_mixed["n_certificates_unmapped"] == 1
    assert res_mixed["helly_violation_rate"] == pytest.approx(1.0)
    assert res_mixed["degeneracy_rate"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="missing from model.class_index_dict"):
        helly_metrics(b2, t3_bundle, cm, [unmapped_cert])

    # Fix 5: When n_certificates == 0, helly_violation_rate and degeneracy_rate must be None (null)
    res_empty = helly_metrics(b2, t3_bundle, cm, [])
    assert res_empty["n_certificates"] == 0
    assert res_empty["n_certificates_unmapped"] == 0
    assert res_empty["helly_violation_rate"] is None
    assert res_empty["degeneracy_rate"] is None


def test_helly_metrics_handmade_multibox_zero(t3_bundle: DatasetBundle) -> None:
    """Verify hand-built 1-D 2-component MultiBoxEL solution of T3 achieves zero violation and degeneracy."""
    cm = get_canonical_model(t3_bundle)
    certs = certificate_triangles(cm)
    assert len(certs) == 1

    mb = MultiBoxEL(t3_bundle.dataset, embed_dim=1, n_components=2, margin=0.0)
    _write_handmade_t3_multibox(mb)

    res = helly_metrics(mb, t3_bundle, cm, certs)
    assert res["n_certificates"] == 1
    assert res["helly_violation_rate"] == 0.0
    assert res["degeneracy_rate"] == 0.0

    # Also verify helly_metrics works on certificates whose c3 is a multi-name tuple (tk4)
    b4 = load_dataset("tk4")
    cm4 = get_canonical_model(b4)
    rep4 = helly_check(cm4, k_max=4, max_certificates=100)
    assert any(len(c.c3) > 1 for c in rep4.certificates_nondegenerate)
    mb4 = MultiBoxEL(b4.dataset, embed_dim=2, n_components=2)
    res4 = helly_metrics(mb4, b4, cm4, rep4.certificates_nondegenerate)
    assert res4["n_certificates"] == len(rep4.certificates_nondegenerate)
    assert 0.0 <= res4["helly_violation_rate"] <= 1.0
    assert 0.0 <= res4["degeneracy_rate"] <= 1.0


def test_faithfulness_sets_are_disjoint_and_sized() -> None:
    """Verify T+, T?, T- sets are pairwise disjoint, contain only named classes, and respect n_sample."""
    bundle = load_dataset("pizza")
    cm = get_canonical_model(bundle)
    model = BoxSquaredEL(bundle.dataset, embed_dim=8)

    n_sample = 120
    t_pos, t_und, t_neg = build_faithfulness_sets(
        bundle, cm, model.class_index_dict, seed=0, n_sample=n_sample
    )

    set_pos = set(t_pos)
    set_und = set(t_und)
    set_neg = set(t_neg)

    assert len(set_pos) == len(t_pos) > 0
    assert len(set_und) == len(t_und) == n_sample
    assert len(set_neg) == len(t_neg) == n_sample

    assert set_pos.isdisjoint(set_und)
    assert set_pos.isdisjoint(set_neg)
    assert set_und.isdisjoint(set_neg)

    for a, b in set_pos | set_und | set_neg:
        assert a not in (TOP, BOTTOM)
        assert b not in (TOP, BOTTOM)
        assert not a.startswith(("http://mowl.borg/aux/", "http://mowl.borg/aux#"))
        assert not b.startswith(("http://mowl.borg/aux/", "http://mowl.borg/aux#"))
        assert a in model.named_class_index_dict
        assert b in model.named_class_index_dict

    # Prefix-only auxiliary filter check: substring '/aux' in a domain IRI is NOT filtered out
    assert not _is_named_class("http://mowl.borg/aux/3f2a9c01be#AuxiliaryEntity1")
    assert not _is_named_class("http://mowl.borg/aux#AuxiliaryEntity1")
    assert _is_named_class("http://example.org/onto/auxin_response")
    assert _is_named_class("http://example.org/onto/aux#ClassA")

    for a, b in t_neg:
        assert not cm.joint_satisfiable((a, b))
    for a, b in t_und:
        assert cm.joint_satisfiable((a, b))
        assert not cm.entails(a, b)

    fm = faithfulness_metrics(model, bundle, cm, seed=0, n_sample=n_sample)
    assert fm["n_pos"] == len(t_pos)
    assert fm["n_und"] == n_sample
    assert fm["n_neg"] == n_sample
    assert fm["auc_pos_neg"] is not None and 0.0 <= fm["auc_pos_neg"] <= 1.0
    assert fm["pair_fvr"] is not None and 0.0 <= fm["pair_fvr"] <= 1.0
    assert fm["fvr"] is not None and 0.0 <= fm["fvr"] <= 1.0

    # Verify n_neg < 50 returns None metrics
    fm_small = faithfulness_metrics(model, bundle, cm, seed=0, n_sample=20)
    assert fm_small["n_neg"] == 20
    assert fm_small["auc_pos_neg"] is None
    assert fm_small["pair_fvr"] is None
    assert fm_small["fvr"] is None


def test_class_regions_shapes(t3_bundle: DatasetBundle) -> None:
    """Verify class_regions returns (n, k, d) numpy arrays with non-negative offsets for all 3 models."""
    ds = t3_bundle.dataset
    d = 6

    for model, expected_k in (
        (ELBE(ds, embed_dim=d), 1),
        (BoxSquaredEL(ds, embed_dim=d), 1),
        (MultiBoxEL(ds, embed_dim=d, n_components=3), 3),
    ):
        n = len(model.class_index_dict)
        centers, offsets = class_regions(model)
        assert isinstance(centers, np.ndarray)
        assert isinstance(offsets, np.ndarray)
        assert centers.shape == (n, expected_k, d)
        assert offsets.shape == (n, expected_k, d)
        assert np.all(offsets >= 0.0)


def test_ranking_and_cost_metrics() -> None:
    """Verify ranking_metrics prefixes keys with rank_, restores mode/device, and cost_metrics reports parameters."""
    bundle = load_dataset("pizza")
    mb = MultiBoxEL(bundle.dataset, embed_dim=8, n_components=2, device="cpu")
    mb.module.train()
    assert mb.module.training is True
    orig_device = next(mb.module.parameters()).device

    # Default device (None -> module's own device) and restore train mode + device
    rank = ranking_metrics(mb, bundle)
    assert mb.module.training is True
    assert next(mb.module.parameters()).device == orig_device

    mb.module.eval()
    rank_eval = ranking_metrics(mb, bundle, device="cpu")
    assert mb.module.training is False
    assert rank_eval == rank

    assert len(rank) > 0
    assert all(k.startswith("rank_") for k in rank)
    for expected_key in (
        "rank_mr",
        "rank_mrr",
        "rank_f_mr",
        "rank_f_mrr",
        "rank_auc",
        "rank_f_auc",
        "rank_hits@1",
        "rank_hits@3",
        "rank_hits@10",
        "rank_f_hits@1",
        "rank_f_hits@3",
        "rank_f_hits@10",
    ):
        assert expected_key in rank
        assert isinstance(rank[expected_key], float)

    cost = cost_metrics(mb, {"epoch_times": [0.1, 0.3], "total_train_time_s": 0.4})
    assert cost["param_count"] == sum(p.numel() for p in mb.module.parameters())
    assert cost["n_components"] == 2
    assert 0.0 <= cost["mean_effective_components"] <= 2.0
    assert cost["mean_epoch_time_s"] == pytest.approx(0.2)
    assert cost["total_train_time_s"] == pytest.approx(0.4)


def test_cm_cache_roundtrip_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_canonical_model writes a protocol-5 pickle with code-tag key, checks memory first, and reloads."""
    import shutil
    import uuid

    tmp_path = (RAW_DATA_ROOT / "test-cm-cache" / uuid.uuid4().hex).resolve()
    assert not str(tmp_path).startswith("/tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        b = load_dataset("tk3")
        expected_pkl = cache_path_for_bundle(b, cache_dir=tmp_path)
        # Key format: <dataset>-<40-hex-ont-sha1>-<12-hex-code-tag>.pkl
        parts = expected_pkl.stem.split("-")
        assert parts[0] == "tk3"
        assert len(parts[1]) == 40
        assert len(parts[2]) == 12

        cm1 = get_canonical_model(b, cache_dir=tmp_path, force_rebuild=True)
        assert isinstance(cm1, CanonicalModel)
        assert expected_pkl.exists()
        with expected_pkl.open("rb") as f:
            header = f.read(2)
        assert header == b"\x80\x05"

        # In-memory cache hit must return before hashing the ontology file
        def _forbid_sha1(_p: Path) -> str:
            raise AssertionError("_sha1_file should not be called on in-memory cache hit")

        monkeypatch.setattr(cm_cache_mod, "_sha1_file", _forbid_sha1)
        cm_mem = get_canonical_model(b, cache_dir=tmp_path, force_rebuild=False)
        assert cm_mem is cm1
        monkeypatch.undo()

        # Reload from protocol-5 pickle on a fresh bundle (no in-memory cache)
        b_fresh = load_dataset("tk3")
        cm2 = get_canonical_model(b_fresh, cache_dir=tmp_path, force_rebuild=False)
        assert cm2 is not cm1
        assert cm2.names == cm1.names
        assert cm2.axioms == cm1.axioms
        assert cm2.completion == cm1.completion
        assert cm2.entails(f"{NS}A1", f"{NS}X2")

        # CLI invocation into the isolated cache_dir
        cm_cache_main(["--dataset", "tk3", "--cache-dir", str(tmp_path)])
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_filtered_ranking_does_not_filter_test_tuple() -> None:
    """Fix 1(b): Verify filtered ranking resets the test tuple's own filter label to 1 before ranking."""
    from mowl.evaluation.base import BaseRankingEvaluator

    class _HandScoreEvaluator(BaseRankingEvaluator):
        def __init__(self, score_matrix: torch.Tensor) -> None:
            heads = torch.tensor([0, 1], dtype=torch.long)
            tails = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            super().__init__(heads, tails, batch_size=2, device="cpu")
            self._score_matrix = score_matrix

        def get_expanded_scores(self, evaluation_model, batch, mode):
            h_indices = batch[:, 0].tolist()
            return self._score_matrix[h_indices], None

    # 2 test tuples: (head=0, tail=2) and (head=1, tail=1)
    # Row 0 scores: tail 0 has 0.1 (other known positive), tail 2 (target) has 0.2, tail 1 has 0.5, tail 3 has 0.9.
    #   Unfiltered order: [0, 2, 1, 3] -> target tail 2 has rank 2.
    #   Filtered order (when (0,0) is filtered and target (0,2) is NOT filtered even if marked 10000 in filtering_labels):
    #     filtered scores = [1000.0, 0.5, 0.2, 0.9] -> order [2, 1, 3, 0] -> target tail 2 has rank 1.
    # Row 1 scores: tail 2 has 0.1 (other known positive), tail 0 has 0.3, tail 1 (target) has 0.4, tail 3 has 0.8.
    #   Unfiltered order: [2, 0, 1, 3] -> target tail 1 has rank 3.
    #   Filtered order (when (1,2) is filtered and target (1,1) is NOT filtered even if marked 10000 in filtering_labels):
    #     filtered scores = [0.3, 0.4, 1000.0, 0.8] -> order [0, 1, 3, 2] -> target tail 1 has rank 2.
    scores = torch.tensor(
        [
            [0.1, 0.5, 0.2, 0.9],
            [0.3, 0.4, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )
    test_data = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    # Mark both the other known positives (0,0), (1,2) AND the test tuples themselves (0,2), (1,1) as 10000
    filter_data = torch.tensor([[0, 0], [0, 2], [1, 2], [1, 1]], dtype=torch.long)
    dummy_module = torch.nn.Linear(1, 1)

    for use_cpu_lookup in (False, True):
        ev = _HandScoreEvaluator(scores)
        ev.update_filtering_labels(filter_data)
        raw_labels = ev.filtering_labels.clone()
        assert raw_labels[0, 2].item() == pytest.approx(10000.0)
        assert raw_labels[1, 1].item() == pytest.approx(10000.0)
        if use_cpu_lookup:
            ev.filtering_labels = _CPUFilteredRowLookup(raw_labels, "cpu")  # type: ignore[assignment]

        res = _compute_fast_head_centric_ranking_metrics(ev, dummy_module, test_data)
        # Unfiltered: ranks [2, 3] -> mr = 2.5, mrr = (1/2 + 1/3)/2 = 5/12, hits@1 = 0.0, hits@3 = 1.0
        assert res["mr"] == pytest.approx(2.5)
        assert res["mrr"] == pytest.approx((0.5 + 1.0 / 3.0) / 2.0)
        assert res["hits@1"] == pytest.approx(0.0)
        assert res["hits@3"] == pytest.approx(1.0)

        # Filtered: ranks [1, 2] -> f_mr = 1.5, f_mrr = (1.0 + 0.5)/2 = 0.75, f_hits@1 = 0.5, f_hits@3 = 1.0
        assert res["f_mr"] == pytest.approx(1.5)
        assert res["f_mrr"] == pytest.approx(0.75)
        assert res["f_hits@1"] == pytest.approx(0.5)
        assert res["f_hits@3"] == pytest.approx(1.0)

        # Underlying filter matrix must not have been mutated in place
        stored = ev.filtering_labels._cpu_labels if use_cpu_lookup else ev.filtering_labels
        assert stored[0, 2].item() == pytest.approx(10000.0)
        assert stored[1, 1].item() == pytest.approx(10000.0)


def _supports_to_center_inradius(h: Tuple[float, float, float]) -> Tuple[Tuple[float, float], float]:
    """Invert supports(c, r) = c @ NORMALS.T + r for sum(h) >= 0."""
    h1, h2, h3 = h
    r = (h1 + h2 + h3) / 3.0
    cy = r - h1
    cx = (h2 - h3) / float(np.sqrt(3.0))
    return (cx, cy), r


def test_helly_metrics_prism_exact_t3_zero(t3_bundle: DatasetBundle) -> None:
    """Verify PrismRegion and helly_metrics on the exact T3 prism configuration achieve HVR 0 and Deg 0.

    Degeneracy threshold rule (Task 6 dispatch):
    - A prism region is degenerate iff ``sum(supports) <= deg_eps`` (where ``sum(supports) = 3 * r_in``,
      so ``r_in <= deg_eps / 3``, corresponding to ``sum(supports) <= 1e-3 * 3`` when scaled by 3
      relative to inradius) or any box side ``(upper - lower) <= deg_eps`` (``side <= 1e-3``,
      whereas standalone box models test half-side ``offset <= deg_eps``).
    """
    from mowl.models import PrismEL
    from experiments.metrics import PrismRegion

    # 1. Direct unit test of PrismRegion with supports (3,3,-1), (3,-1,3), (-1,3,3) and identical boxes
    sup = np.array(
        [
            [3.0, 3.0, -1.0],
            [3.0, -1.0, 3.0],
            [-1.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    lo = np.zeros((3, 2), dtype=np.float64)
    hi = np.ones((3, 2), dtype=np.float64)
    prisms = PrismRegion(supports=sup, lower=lo, upper=hi)

    p1, p2, p3 = prisms[0], prisms[1], prisms[2]
    assert not p1.is_empty() and not p1.is_degenerate()
    assert not p2.is_empty() and not p2.is_degenerate()
    assert not p3.is_empty() and not p3.is_degenerate()

    # Degeneracy boundary checks: sum(supports) <= 1e-3 or box side (upper - lower) <= 1e-3
    deg_tri = PrismRegion(
        supports=np.array([[1e-3 / 3.0, 1e-3 / 3.0, 1e-3 / 3.0]]),
        lower=np.zeros((1, 2)),
        upper=np.ones((1, 2)),
    )
    assert deg_tri.is_degenerate(deg_eps=1e-3)
    nondeg_tri = PrismRegion(
        supports=np.array([[1e-3, 1e-3, 1e-3]]),  # sum = 3e-3 > 1e-3
        lower=np.zeros((1, 2)),
        upper=np.ones((1, 2)),
    )
    assert not nondeg_tri.is_degenerate(deg_eps=1e-3)
    assert nondeg_tri.is_degenerate(deg_eps=1e-3 * 3.0)

    deg_box = PrismRegion(
        supports=np.array([[1.0, 1.0, 1.0]]),
        lower=np.zeros((1, 2)),
        upper=np.array([[1e-3, 1.0]]),  # side <= 1e-3
    )
    assert deg_box.is_degenerate(deg_eps=1e-3)

    # Pairwise non-empty
    assert not p1.intersect(p2).is_empty()
    assert not p2.intersect(p3).is_empty()
    assert not p1.intersect(p3).is_empty()

    # Triple intersection has supports (-1, -1, -1), sum = -3 < -1e-6 -> empty
    triple = p1.intersect(p2).intersect(p3)
    assert np.allclose(triple.supports, np.array([-1.0, -1.0, -1.0]))
    assert triple.is_empty()

    # 2. Full helly_metrics test on PrismEL with the exact T3 prism configuration
    cm = get_canonical_model(t3_bundle)
    certs = certificate_triangles(cm)
    assert len(certs) == 1

    prism = PrismEL(t3_bundle.dataset, embed_dim=4, margin=0.0)
    support_spec: Dict[str, Tuple[float, float, float]] = {
        f"{NS}X1": (3.0, 3.0, -1.0),
        f"{NS}X2": (3.0, -1.0, 3.0),
        f"{NS}X3": (-1.0, 3.0, 3.0),
        f"{NS}A": (3.0, -1.0, -1.0),
        f"{NS}B": (-1.0, -1.0, 3.0),
        f"{NS}C": (-1.0, 3.0, -1.0),
        f"{NS}Z": (3.0, -1.0, -1.0),
        OWL_THING: (4.0, 4.0, 4.0),
    }
    with torch.no_grad():
        prism.module.box_center.weight.zero_()
        prism.module.box_offset.weight.fill_(1.0)
        prism.module.tri_center.weight.zero_()
        prism.module.tri_inradius.weight.fill_(2.0)
        for iri, h in support_spec.items():
            if iri not in prism.class_index_dict:
                continue
            idx = prism.class_index_dict[iri]
            (cx, cy), r = _supports_to_center_inradius(h)
            prism.module.tri_center.weight[idx] = torch.tensor(
                [cx, cy], dtype=prism.module.tri_center.weight.dtype
            )
            prism.module.tri_inradius.weight[idx] = torch.tensor(
                [r], dtype=prism.module.tri_inradius.weight.dtype
            )

    reg = class_regions(prism)
    assert isinstance(reg, PrismRegion)
    s_arr, l_arr, u_arr = reg
    assert s_arr.shape == (len(prism.class_index_dict), 3)
    assert l_arr.shape == (len(prism.class_index_dict), 2)
    assert u_arr.shape == (len(prism.class_index_dict), 2)

    res = helly_metrics(prism, t3_bundle, cm, certs)
    assert res["n_certificates"] == 1
    assert res["n_certificates_unmapped"] == 0
    assert res["helly_violation_rate"] == pytest.approx(0.0)
    assert res["degeneracy_rate"] == pytest.approx(0.0)

    m_strict = modelness(prism)
    assert m_strict["modelness_all"] == pytest.approx(1.0)


def test_point_modelness_metric(t3_bundle: DatasetBundle) -> None:
    """Verify point_modelness returns None for axiom objective and pos/hardneg dict for point/mixed/hybrid."""
    from mowl.models import MultiELBE
    from experiments.metrics import point_modelness

    m_axiom = MultiBoxEL(t3_bundle.dataset, embed_dim=2, n_components=2, objective="axiom")
    assert point_modelness(m_axiom) is None

    for obj in ("point", "mixed", "hybrid"):
        m_obj = MultiELBE(
            t3_bundle.dataset,
            embed_dim=2,
            n_components=2,
            objective=obj,
            points_per_epoch=16,
            points_seed=0,
            device="cpu",
        )
        m_obj.train(epochs=1)
        pm = point_modelness(m_obj)
        assert isinstance(pm, dict), f"Expected dict for objective={obj!r}, got {pm!r}"
        assert set(pm.keys()) == {"pos_frac", "hardneg_frac"}
        assert 0.0 <= pm["pos_frac"] <= 1.0
        assert 0.0 <= pm["hardneg_frac"] <= 1.0

