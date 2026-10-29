"""Tests for experiment dataset generators and loader (Task 6)."""
from __future__ import annotations

import os
from pathlib import Path

import mowl
mowl.init_jvm("8g")

import pytest
from mowl.datasets import PathDataset
from mowl.geometric import (
    BOTTOM,
    NF1,
    TOP,
    CanonicalModel,
    certificate_triangles,
    helly_check,
    normalise,
)
from mowl.projection import TaxonomyProjector

from experiments.config import DATASETS, PIZZA_RAW_PATH
from experiments.datasets import DatasetBundle, load_dataset

NS = "http://mowl.borg/geo#"


def _subsumption_pairs(ontology) -> set[tuple[str, str]]:
    """Extract (sub, sup) atomic subsumption pairs from an OWLOntology."""
    proj = TaxonomyProjector()
    return {(str(e.src), str(e.dst)) for e in proj.project(ontology)}


def test_tk_has_expected_certificates():
    """Verify certificate_triangles and helly_check on synthetic T_k ontologies.

    - For tk3: certificate_triangles(cm) has 1 certificate over (X1, X2, X3) with
      pairwise names {A1, A2, A3}.
    - For tk4: T_k (k >= 4) names its (k-1)-wise conjunctions (binary-normalised with
      auxiliary names), not all pairwise ones of a triangle, so certificate_triangles(cm)
      is empty while helly_check(cm, k_max=4).certificates_nondegenerate is non-empty
      and finds the size-4 minimal unsatisfiable set over (X1, X2, X3, X4).
    - For all k in {3, 4, 5, 6}: all X_i, A_i, and Z are satisfiable, every (k-1)-subset
      of {X_1..X_k} is jointly satisfiable, and the k-fold conjunction X_1 ⊓ ... ⊓ X_k
      is unsatisfiable.
    """
    b3 = load_dataset("tk3")
    cm3 = CanonicalModel.from_dataset(b3.dataset)
    tri3 = certificate_triangles(cm3)
    assert len(tri3) == 1
    c3 = tri3[0]
    assert (c3.c1, c3.c2, c3.c3) == ((f"{NS}X1",), (f"{NS}X2",), (f"{NS}X3",))
    assert set(c3.pairwise_names) == {f"{NS}A1", f"{NS}A2", f"{NS}A3"}

    b4 = load_dataset("tk4")
    cm4 = CanonicalModel.from_dataset(b4.dataset)
    assert certificate_triangles(cm4) == []
    rep4 = helly_check(cm4, k_max=4, max_certificates=100)
    assert len(rep4.certificates_nondegenerate) > 0
    x_certs_4 = [
        c
        for c in rep4.certificates_nondegenerate
        if (c.c1, c.c2, c.c3)
        == ((f"{NS}X1",), (f"{NS}X2",), (f"{NS}X3", f"{NS}X4"))
    ]
    assert len(x_certs_4) == 1

    for k in (3, 4, 5, 6):
        bundle = load_dataset(f"tk{k}")
        cm = CanonicalModel.from_dataset(bundle.dataset)
        xs = [f"{NS}X{i}" for i in range(1, k + 1)]
        as_ = [f"{NS}A{i}" for i in range(1, k + 1)]
        for x in xs:
            assert cm.is_satisfiable(x), f"Expected {x} satisfiable in tk{k}"
        for a in as_:
            assert cm.is_satisfiable(a), f"Expected {a} satisfiable in tk{k}"
        assert cm.is_satisfiable(f"{NS}Z"), f"Expected Z satisfiable in tk{k}"
        for i in range(k):
            subset = [xs[j] for j in range(k) if j != i]
            assert cm.joint_satisfiable(subset), f"Expected {subset} satisfiable in tk{k}"
        assert not cm.joint_satisfiable(xs), f"Expected {xs} unsatisfiable in tk{k}"


def test_pizza_split_disjoint_and_closed():
    """Verify Pizza-EL train/valid/test split invariants and deductive closure."""
    bundle = load_dataset("pizza")
    assert bundle.task == "subsumption"
    assert bundle.deductive_closure_ontology is not None

    ds = bundle.dataset
    train_no = normalise(ds.ontology)
    train_pairs = {(ax.sub, ax.sup) for ax in train_no.axioms if isinstance(ax, NF1)}
    valid_pairs = _subsumption_pairs(ds.validation)
    test_pairs = _subsumption_pairs(ds.testing)
    dc_pairs = _subsumption_pairs(bundle.deductive_closure_ontology)

    assert len(valid_pairs) > 0
    assert len(test_pairs) > 0

    # Train/valid/test atomic subsumptions must be pairwise disjoint
    assert train_pairs.isdisjoint(valid_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert valid_pairs.isdisjoint(test_pairs)

    # Every valid/test axiom must be entailed by the deductive closure of the full EL fragment
    full_cm = CanonicalModel.from_dataset(PathDataset(str(PIZZA_RAW_PATH)))
    for sub, sup in valid_pairs | test_pairs:
        assert sub not in (TOP, BOTTOM)
        assert sup not in (TOP, BOTTOM)
        assert sub != sup
        assert "/aux" not in sub and "aux#" not in sub
        assert "/aux" not in sup and "aux#" not in sup
        assert full_cm.is_satisfiable(sub)
        assert full_cm.is_satisfiable(sup)
        assert full_cm.entails(sub, sup)
        assert (sub, sup) in dc_pairs

    # All classes of valid, test, and deductive_closure must occur in train
    train_classes = {str(c.toStringID()) for c in ds.ontology.getClassesInSignature()}
    valid_classes = {str(c.toStringID()) for c in ds.validation.getClassesInSignature()}
    test_classes = {str(c.toStringID()) for c in ds.testing.getClassesInSignature()}
    dc_classes = {
        str(c.toStringID()) for c in bundle.deductive_closure_ontology.getClassesInSignature()
    }
    assert valid_classes <= train_classes
    assert test_classes <= train_classes
    assert dc_classes <= train_classes

    # Verify evaluation_classes property works as required by SubsumptionEvaluator
    eval_heads, eval_tails = ds.evaluation_classes
    assert len(eval_heads) > 0
    assert len(eval_tails) == len(eval_heads)


def test_load_dataset_all():
    """Verify every dataset id in DATASETS loads and exposes the expected DatasetBundle contract."""
    assert DATASETS == ["tk3", "tk4", "tk5", "tk6", "pizza", "go", "foodon"]
    for name in DATASETS:
        bundle = load_dataset(name)
        assert isinstance(bundle, DatasetBundle)
        assert bundle.name == name
        assert bundle.dataset is not None
        assert bundle.dataset.ontology is not None
        assert len(bundle.paths) > 0
        for p in bundle.paths.values():
            assert os.path.exists(p), f"Missing path {p} for dataset {name}"

        if name.startswith("tk"):
            assert bundle.task == "satisfaction"
            assert bundle.deductive_closure_ontology is None
        else:
            assert bundle.task == "subsumption"
            assert bundle.deductive_closure_ontology is not None
            assert bundle.dataset.validation is not None
            assert bundle.dataset.testing is not None
            eval_heads, eval_tails = bundle.dataset.evaluation_classes
            assert len(eval_heads) > 0
            assert len(eval_tails) > 0
