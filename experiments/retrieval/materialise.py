#!/usr/bin/env python3
"""One-pass materialisation of a KB's ABox, used as a cheap filter for query sampling.

Inputs
------
* the EL projection ``<kb>_el.owl`` (class membership is realised once by ELK on it);
* the full KB ``<kb>.owl`` (its object-property assertions and RBox axioms give the role edges).

Outputs
-------
A :class:`Materialisation`: for every named class of the projection a boolean membership vector
over the projection's named individuals, and for every named object property a sparse
individual-by-individual edge matrix. Expressions of the benchmark DSL are then evaluated by
set operations (``Materialisation.evaluate``). The CLI writes a JSON summary
(``materialisation.json``) of what was materialised.

What is materialised (exactly)
------------------------------
Class membership: ``getInstances(C, direct=false)`` from ELK on the EL projection for every
named class ``C`` after one ``precomputeInferences(CLASS_HIERARCHY, CLASS_ASSERTIONS)``.

Role edges: all asserted ``ObjectPropertyAssertion`` axioms of the full KB (an assertion on
``inverse(p)`` is stored as the reversed ``p`` edge), closed to a fixpoint under these RBox
axioms of the full KB, using only named properties and their inverses:

* ``SubObjectPropertyOf(P, Q)`` with ``P``, ``Q`` each a named property or its inverse;
* ``EquivalentObjectProperties`` (as sub-properties in both directions);
* ``InverseObjectProperties(p, q)`` (``p`` ⊑ ``q⁻`` and ``q`` ⊑ ``p⁻``);
* ``SymmetricObjectProperty(p)`` (``p`` ⊑ ``p⁻``);
* ``TransitiveObjectProperty(p)`` (transitive closure of ``p`` edges).

Not included: property chains (``SubObjectPropertyOf(ObjectPropertyChain…)``), edges implied by
nominals/``ObjectHasValue``, by ``sameAs`` or by (inverse-)functionality, any data properties,
and class memberships entailed only by the non-EL part of the full KB (e.g. via inverse roles or
universal restrictions). Existential answers obtained through anonymous successors (``A ⊑ ∃r.B``)
are only present for named classes (via ELK), not for complex query fillers. The filter is
therefore incomplete with respect to the reasoner; stored reference answers never come from it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from common import DATA_ROOT, NAMES, iri, load_ontology


@dataclass
class Materialisation:
    """Entailed class membership and closed role edges over a fixed individual order.

    Attributes
    ----------
    individuals : list of str
        Sorted individual IRIs; vector position ``k`` is ``individuals[k]``.
    classes : list of str
        Sorted named classes of the EL projection (without Thing/Nothing).
    roles : list of str
        Sorted named object properties of the EL projection (sampling vocabulary).
    members : dict
        Class IRI to boolean membership vector of length ``len(individuals)``.
    edges : dict
        Role IRI to ``float32`` CSR matrix ``E`` with ``E[a, b] = 1`` iff ``(a, b)`` is an edge.
    summary : dict
        Counts and timings, JSON-serialisable.
    """

    individuals: list[str]
    classes: list[str]
    roles: list[str]
    members: dict[str, np.ndarray]
    edges: dict[str, sp.csr_matrix]
    summary: dict[str, Any] = field(default_factory=dict)
    _transposed: dict[str, sp.csr_matrix] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        """Number of individuals."""
        return len(self.individuals)

    def _empty(self) -> np.ndarray:
        return np.zeros(self.n, dtype=bool)

    def _matrix(self, role: str, inverse: bool) -> sp.csr_matrix | None:
        matrix = self.edges.get(role)
        if matrix is None or not inverse:
            return matrix
        if role not in self._transposed:
            self._transposed[role] = matrix.T.tocsr()
        return self._transposed[role]

    def evaluate(self, expr: Any) -> np.ndarray:
        """Evaluate a DSL expression to a boolean answer vector.

        Parameters
        ----------
        expr : str or list
            Class IRI, ``["and", e1, e2, ...]``, ``["some", role, e]`` or
            ``["some", ["inv", role], e]``.

        Returns
        -------
        numpy.ndarray
            Boolean vector over ``individuals``.
        """
        if isinstance(expr, str):
            vec = self.members.get(expr)
            return vec.copy() if vec is not None else self._empty()
        op = expr[0]
        if op == "and":
            out = np.ones(self.n, dtype=bool)
            for item in expr[1:]:
                out &= self.evaluate(item)
            return out
        if op == "some":
            role, filler = expr[1], expr[2]
            inverse = isinstance(role, list)
            matrix = self._matrix(role[1] if inverse else role, inverse)
            fill = self.evaluate(filler)
            if matrix is None or not fill.any():
                return self._empty()
            return np.asarray(matrix @ fill.astype(np.float32)).ravel() > 0
        raise ValueError(f"Unknown expression operator: {op!r}")

    def answers(self, expr: Any) -> list[str]:
        """Sorted IRIs of the individuals in the filter answer set of ``expr``."""
        return [self.individuals[k] for k in np.flatnonzero(self.evaluate(expr))]


def _prop(expr: Any) -> tuple[str, bool] | None:
    """Return ``(iri, inverted)`` for a named property or its inverse, else ``None``."""
    inverted = False
    while expr.isAnonymous():
        if str(expr.getClass().getSimpleName()) != "OWLObjectInverseOfImpl":
            return None
        expr = expr.getInverse()
        inverted = not inverted
    if expr.isOWLTopObjectProperty() or expr.isOWLBottomObjectProperty():
        return None
    return iri(expr.asOWLObjectProperty()), inverted


def rbox_rules(ontology: Any) -> tuple[list[tuple[tuple[str, bool], tuple[str, bool]]], set[str], dict[str, int]]:
    """Collect sub-property rules and transitive properties from the RBox.

    Returns
    -------
    rules : list
        Pairs ``(P, Q)`` meaning edges of ``P`` are edges of ``Q``; ``P = (iri, inverted)``.
    transitive : set of str
        Named transitive properties.
    counts : dict
        Number of RBox axioms used, by axiom type.
    """
    from org.semanticweb.owlapi.model import AxiomType
    from org.semanticweb.owlapi.model.parameters import Imports
    rules: list = []
    transitive: set[str] = set()
    counts = {"SubObjectPropertyOf": 0, "EquivalentObjectProperties": 0, "InverseObjectProperties": 0,
              "SymmetricObjectProperty": 0, "TransitiveObjectProperty": 0, "ignored_chains_or_complex": 0}
    for ax in ontology.getAxioms(AxiomType.SUB_OBJECT_PROPERTY, Imports.EXCLUDED):
        sub, sup = _prop(ax.getSubProperty()), _prop(ax.getSuperProperty())
        if sub and sup:
            rules.append((sub, sup)); counts["SubObjectPropertyOf"] += 1
        else:
            counts["ignored_chains_or_complex"] += 1
    for ax in ontology.getAxioms(AxiomType.EQUIVALENT_OBJECT_PROPERTIES, Imports.EXCLUDED):
        props = [_prop(p) for p in ax.getProperties()]
        props = [p for p in props if p]
        for a in props:
            for b in props:
                if a != b:
                    rules.append((a, b))
        counts["EquivalentObjectProperties"] += 1
    for ax in ontology.getAxioms(AxiomType.INVERSE_OBJECT_PROPERTIES, Imports.EXCLUDED):
        first, second = _prop(ax.getFirstProperty()), _prop(ax.getSecondProperty())
        if first and second:
            rules.append((first, (second[0], not second[1])))
            rules.append((second, (first[0], not first[1])))
            counts["InverseObjectProperties"] += 1
    for ax in ontology.getAxioms(AxiomType.SYMMETRIC_OBJECT_PROPERTY, Imports.EXCLUDED):
        p = _prop(ax.getProperty())
        if p:
            rules.append((p, (p[0], not p[1]))); counts["SymmetricObjectProperty"] += 1
    for ax in ontology.getAxioms(AxiomType.TRANSITIVE_OBJECT_PROPERTY, Imports.EXCLUDED):
        p = _prop(ax.getProperty())
        if p:
            transitive.add(p[0]); counts["TransitiveObjectProperty"] += 1
    return rules, transitive, counts


def close_edges(edges: dict[str, sp.csr_matrix], rules, transitive: set[str], max_rounds: int = 100) -> tuple[dict[str, sp.csr_matrix], int]:
    """Close edge matrices under sub-property rules and transitivity to a fixpoint.

    Parameters
    ----------
    edges : dict
        Role IRI to boolean-valued CSR matrix (modified copies are returned).
    rules : list
        ``((p, inv_p), (q, inv_q))`` pairs from :func:`rbox_rules`.
    transitive : set of str
        Transitive properties.
    max_rounds : int
        Safety bound on fixpoint iterations.

    Returns
    -------
    (dict, int)
        Closed matrices and the number of rounds used.
    """
    n = next(iter(edges.values())).shape[0] if edges else 0
    closed = {p: (m > 0).astype(np.float32).tocsr() for p, m in edges.items()}

    def get(p: str) -> sp.csr_matrix:
        if p not in closed:
            closed[p] = sp.csr_matrix((n, n), dtype=np.float32)
        return closed[p]

    for rounds in range(1, max_rounds + 1):
        before = sum(m.nnz for m in closed.values())
        for (p, p_inv), (q, q_inv) in rules:
            src = get(p)
            if p_inv != q_inv:
                src = src.T.tocsr()
            closed[q] = ((get(q) + src) > 0).astype(np.float32).tocsr()
        for p in transitive:
            m = get(p)
            while True:
                grown = ((m + m @ m) > 0).astype(np.float32).tocsr()
                if grown.nnz == m.nnz:
                    break
                m = grown
            closed[p] = m
        if sum(m.nnz for m in closed.values()) == before:
            return closed, rounds
    raise RuntimeError("role closure did not reach a fixpoint")


def materialise(el_ontology: Any, full_ontology: Any) -> Materialisation:
    """Realise the EL projection once with ELK and close the full KB's asserted role edges.

    Parameters
    ----------
    el_ontology : OWLOntology
        EL projection; defines individuals, classes and sampling roles, and is realised by ELK.
    full_ontology : OWLOntology
        Full KB; provides asserted object-property assertions and the RBox axioms listed in the
        module docstring.

    Returns
    -------
    Materialisation

    Raises
    ------
    RuntimeError
        If the EL projection is inconsistent according to ELK.
    """
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.model import AxiomType
    from org.semanticweb.owlapi.model.parameters import Imports
    from org.semanticweb.owlapi.reasoner import InferenceType
    started = time.monotonic()
    individuals = sorted(iri(i) for i in el_ontology.getIndividualsInSignature(Imports.EXCLUDED))
    position = {x: k for k, x in enumerate(individuals)}
    classes = sorted(iri(c) for c in el_ontology.getClassesInSignature(Imports.EXCLUDED)
                     if not c.isOWLThing() and not c.isOWLNothing())
    roles = sorted(iri(r) for r in el_ontology.getObjectPropertiesInSignature(Imports.EXCLUDED)
                   if not r.isOWLTopObjectProperty() and not r.isOWLBottomObjectProperty())
    reasoner = ElkReasonerFactory().createReasoner(el_ontology)
    try:
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY, InferenceType.CLASS_ASSERTIONS)
        if not reasoner.isConsistent():
            raise RuntimeError("EL projection is inconsistent according to ELK")
        realised = time.monotonic()
        df = el_ontology.getOWLOntologyManager().getOWLDataFactory()
        from org.semanticweb.owlapi.model import IRI
        members: dict[str, np.ndarray] = {}
        for c in classes:
            vec = np.zeros(len(individuals), dtype=bool)
            for node in reasoner.getInstances(df.getOWLClass(IRI.create(c)), False).getNodes():
                for ind in node.getEntities():
                    k = position.get(iri(ind))
                    if k is not None:
                        vec[k] = True
            members[c] = vec
    finally:
        reasoner.dispose()
    membership_done = time.monotonic()
    rows: dict[str, tuple[list[int], list[int]]] = {}
    asserted = 0
    for ax in full_ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION, Imports.EXCLUDED):
        prop = _prop(ax.getProperty())
        s, o = position.get(iri(ax.getSubject())), position.get(iri(ax.getObject()))
        if prop is None or s is None or o is None:
            continue
        if prop[1]:
            s, o = o, s
        r, c = rows.setdefault(prop[0], ([], []))
        r.append(s); c.append(o); asserted += 1
    n = len(individuals)
    edges = {p: sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n)) for p, (r, c) in rows.items()}
    asserted_distinct = {p: int((m > 0).nnz) for p, m in edges.items()}
    rules, transitive, rbox_counts = rbox_rules(full_ontology)
    closed, rounds = close_edges(edges, rules, transitive) if edges else ({}, 0)
    finished = time.monotonic()
    summary = {
        "individuals": n, "classes": len(classes), "sampling_roles": len(roles),
        "class_memberships": int(sum(int(v.sum()) for v in members.values())),
        "nonempty_classes": int(sum(1 for v in members.values() if v.any())),
        "asserted_role_assertions": asserted,
        "asserted_distinct_edges": int(sum(asserted_distinct.values())),
        "closed_edges": int(sum(m.nnz for m in closed.values())),
        "closed_edges_by_role": {p: int(m.nnz) for p, m in sorted(closed.items())},
        "rbox_axioms_used": rbox_counts, "closure_rounds": rounds,
        "seconds": {"elk_realisation": round(realised - started, 2), "membership_extraction": round(membership_done - realised, 2),
                    "role_closure": round(finished - membership_done, 2), "total": round(finished - started, 2)},
        "membership_source": "ELK 0.4.3 getInstances(C, direct=false) on the EL projection",
        "edge_source": "asserted ObjectPropertyAssertion of the full KB closed under SubObjectPropertyOf, "
                       "EquivalentObjectProperties, InverseObjectProperties, SymmetricObjectProperty, TransitiveObjectProperty",
    }
    return Materialisation(individuals, classes, roles, members, closed, summary)


def materialise_kb(kb: str, root: Path = DATA_ROOT) -> Materialisation:
    """Load ``<kb>_el.owl`` and ``<kb>.owl`` from the data directory and materialise them."""
    directory = root / "kbs" / kb
    _, el = load_ontology(directory / f"{kb}_el.owl")
    _, full = load_ontology(directory / f"{kb}.owl")
    return materialise(el, full)


def main() -> None:
    """Materialise one KB and write ``materialisation.json`` (summary only)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", choices=NAMES, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--heap-gb", type=int, default=None, help="JVM heap (GiB); capped by free memory and 20")
    args = parser.parse_args()
    from common import start_jvm
    start_jvm(args.heap_gb)
    mat = materialise_kb(args.kb, args.data_dir)
    out = args.data_dir / "kbs" / args.kb / "materialisation.json"
    out.write_text(json.dumps(mat.summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in mat.summary.items() if k != "closed_edges_by_role"}, indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
