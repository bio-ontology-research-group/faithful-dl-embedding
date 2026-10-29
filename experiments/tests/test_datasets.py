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
from experiments.data.make_pizza import is_auxiliary_iri
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
    """Verify Pizza-EL train/valid/test split invariants, closure, and non-auxiliary filtering."""
    bundle = load_dataset("pizza")
    assert bundle.task == "subsumption"
    assert bundle.deductive_closure_ontology is not None

    ds = bundle.dataset
    train_no = normalise(ds.ontology)
    train_pairs = {(ax.sub, ax.sup) for ax in train_no.axioms if isinstance(ax, NF1)}
    valid_pairs = _subsumption_pairs(ds.validation)
    test_pairs = _subsumption_pairs(ds.testing)
    dc_pairs = _subsumption_pairs(bundle.deductive_closure_ontology)

    assert len(valid_pairs) == 16
    assert len(test_pairs) == 16
    assert len(dc_pairs) == 224

    # Train/valid/test atomic subsumptions must be pairwise disjoint
    assert train_pairs.isdisjoint(valid_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert valid_pairs.isdisjoint(test_pairs)

    # Fix 1(a): ontology_deductive_closure.owl is the deductive closure of the training
    # ontology (cm_train); since 0/16 valid and 0/16 test axioms are entailed by train,
    # none appears in dc_pairs.
    cm_train = CanonicalModel.from_normalised(train_no)
    sat_non_aux = {
        n
        for n in (set(cm_train.names) - {TOP, BOTTOM})
        if not is_auxiliary_iri(n)
    }
    expected_train_dc = {
        (a, b)
        for a in sat_non_aux
        for b in sat_non_aux
        if a != b and cm_train.entails(a, b)
    }
    assert dc_pairs == expected_train_dc
    assert dc_pairs.isdisjoint(valid_pairs)
    assert dc_pairs.isdisjoint(test_pairs)

    for sub, sup in valid_pairs | test_pairs:
        assert sub not in (TOP, BOTTOM)
        assert sup not in (TOP, BOTTOM)
        assert sub != sup
        assert not is_auxiliary_iri(sub)
        assert not is_auxiliary_iri(sup)

    # I1: No deductive closure axiom is auxiliary or involves TOP/BOTTOM
    for sub, sup in dc_pairs:
        assert sub not in (TOP, BOTTOM)
        assert sup not in (TOP, BOTTOM)
        assert sub != sup
        assert not is_auxiliary_iri(sub), f"Auxiliary subclass in closure: {sub}"
        assert not is_auxiliary_iri(sup), f"Auxiliary superclass in closure: {sup}"

    # I2: Equivalence-derived NF1s must never be held out
    pizza_ns = "http://www.co-ode.org/ontologies/pizza/pizza.owl#"
    equiv_nf1s = {
        (f"{pizza_ns}SpicyPizza", f"{pizza_ns}Pizza"),
        (f"{pizza_ns}CheeseyPizza", f"{pizza_ns}Pizza"),
        (f"{pizza_ns}MeatyPizza", f"{pizza_ns}Pizza"),
        (f"{pizza_ns}SpicyPizzaEquivalent", f"{pizza_ns}Pizza"),
        (f"{pizza_ns}SpicyTopping", f"{pizza_ns}PizzaTopping"),
    }
    assert (valid_pairs | test_pairs).isdisjoint(equiv_nf1s)
    assert equiv_nf1s <= train_pairs

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
    assert any(is_auxiliary_iri(c) for c in train_classes)
    assert not any(is_auxiliary_iri(c) for c in dc_classes)

    # I1: Verify evaluation_classes property excludes auxiliary classes
    eval_heads, eval_tails = ds.evaluation_classes
    assert len(eval_heads) > 0
    assert len(eval_tails) == len(eval_heads)
    for cls_iri in list(eval_heads.as_str) + list(eval_tails.as_str):
        assert not is_auxiliary_iri(cls_iri), f"Auxiliary class in evaluation_classes: {cls_iri}"


@pytest.mark.slow
@pytest.mark.skipif(
    not PIZZA_RAW_PATH.expanduser().exists(),
    reason=f"Raw pizza.owl absent at {PIZZA_RAW_PATH}",
)
def test_pizza_raw_entailment_and_asserted_candidates():
    """Verify held-out Pizza axioms come from raw asserted SubClassOf(named, named) and are entailed by raw pizza.owl."""
    from org.semanticweb.owlapi.model import AxiomType, ClassExpressionType

    bundle = load_dataset("pizza")
    ds = bundle.dataset
    valid_pairs = _subsumption_pairs(ds.validation)
    test_pairs = _subsumption_pairs(ds.testing)
    dc_pairs = _subsumption_pairs(bundle.deductive_closure_ontology)

    raw_ds = PathDataset(str(PIZZA_RAW_PATH.expanduser().resolve()))
    raw_asserted_pairs = set()
    for owl_ax in raw_ds.ontology.getAxioms(AxiomType.SUBCLASS_OF):
        sub_expr = owl_ax.getSubClass()
        sup_expr = owl_ax.getSuperClass()
        if (
            sub_expr.getClassExpressionType() == ClassExpressionType.OWL_CLASS
            and sup_expr.getClassExpressionType() == ClassExpressionType.OWL_CLASS
        ):
            raw_asserted_pairs.add(
                (
                    str(sub_expr.asOWLClass().toStringID()),
                    str(sup_expr.asOWLClass().toStringID()),
                )
            )
    assert len(raw_asserted_pairs) == 84

    full_cm = CanonicalModel.from_dataset(raw_ds)
    sat_non_aux = {
        n
        for n in (set(full_cm.names) - {TOP, BOTTOM})
        if not is_auxiliary_iri(n)
    }
    valid_candidates = {
        (s, t)
        for s, t in raw_asserted_pairs
        if s in sat_non_aux and t in sat_non_aux and s != t
    }
    assert len(valid_candidates) == 81

    for sub, sup in valid_pairs | test_pairs:
        assert (sub, sup) in valid_candidates
        assert full_cm.is_satisfiable(sub)
        assert full_cm.is_satisfiable(sup)
        assert full_cm.entails(sub, sup)

    train_cm = CanonicalModel.from_dataset(ds)
    expected_dc = {
        (a, b)
        for a in sat_non_aux
        for b in sat_non_aux
        if a != b and train_cm.entails(a, b)
    }
    assert dc_pairs == expected_dc
    assert dc_pairs.isdisjoint(valid_pairs | test_pairs)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            d,
            marks=pytest.mark.slow if d in ("go", "foodon") else (),
        )
        for d in DATASETS
    ],
)
def test_load_dataset_all(name: str):
    """Verify every dataset id in DATASETS loads and exposes the expected DatasetBundle contract."""
    assert DATASETS == ["tk3", "tk4", "tk5", "tk6", "pizza", "go", "foodon"]
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
