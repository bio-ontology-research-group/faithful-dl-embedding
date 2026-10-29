#!/usr/bin/env python3
"""Generate synthetic Helly stress-suite ontologies T_k (k = 3, 4, 5, 6).

For a given k >= 3, T_k contains:
- Concept names X_1, ..., X_k;
- For each i in 1..k, a concept name A_i ≡ ⊓_{j != i} X_j, binary-normalised with
  shared auxiliary equivalence names Aux_<idx_1>_..._<idx_m> ≡ left ⊓ right when
  k >= 4 (for k = 3 the conjunction has two operands and needs no auxiliary name);
- Pairwise disjointness A_i ⊓ A_j ⊑ ⊥ for all 1 <= i < j <= k;
- One unconstrained declared concept name Z.

Guarantees
----------
- Every generated ontology is loadable via ``mowl.datasets.PathDataset`` and
  normalises cleanly via ``mowl.geometric.normalise`` with ``dropped_axioms == 0``.
- In the ``CanonicalModel`` of ``T_k``, all ``X_1..X_k``, ``A_1..A_k``, and ``Z``
  are satisfiable, every ``(k - 1)``-subset of ``{X_1..X_k}`` is jointly
  satisfiable, and the ``k``-fold conjunction ``X_1 ⊓ ... ⊓ X_k`` is unsatisfiable.
- For ``k = 3``, ``certificate_triangles(cm)`` returns the unique certificate over
  ``(X1, X2, X3)`` with pairwise names ``{A1, A2, A3}``.
- For ``k >= 4``, ``certificate_triangles(cm)`` is empty (because T_k names its
  ``(k - 1)``-wise conjunctions rather than all three pairwise edges of a 3-clique
  whose triple is unsatisfiable), while ``helly_check(cm, k_max=k).certificates_nondegenerate``
  is non-empty and includes the size-``k`` certificate with ``c1=(X1,)``,
  ``c2=(X2,)``, ``c3=(X3, ..., Xk)``.

Non-guarantees
--------------
- Does not generate role axioms or ABox assertions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mowl

from experiments.config import SYNTHETIC_DIR, SYNTHETIC_NS, SYNTHETIC_ONT_BASE


def build_tk_ontology(k: int):
    """Construct the OWLAPI ``OWLOntology`` for synthetic benchmark ``T_k``.

    Parameters
    ----------
    k : int
        Arity parameter ``k >= 3``.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Produces exact binary ``EquivalentClasses`` definitions for ``A_1..A_k``
      (and intermediate ``Aux_*`` prefixes for ``k >= 4``), pairwise disjointness
      ``A_i ⊓ A_j ⊑ ⊥`` for ``i < j``, and a declaration for ``Z``.

    Non-guarantees
    --------------
    - Requires the JVM to be started via ``mowl.init_jvm`` before calling.
    """
    if k < 3:
        raise ValueError(f"T_k requires k >= 3, got {k}")

    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{SYNTHETIC_ONT_BASE}/tk{k}")
    bot = adapter.create_class(BOT)

    xs = {i: adapter.create_class(f"{SYNTHETIC_NS}X{i}") for i in range(1, k + 1)}
    as_ = {i: adapter.create_class(f"{SYNTHETIC_NS}A{i}") for i in range(1, k + 1)}
    aux_cache: Dict[Tuple[int, ...], object] = {}

    def get_prefix_conjunction(indices: Tuple[int, ...]):
        if len(indices) == 1:
            return xs[indices[0]]
        if indices in aux_cache:
            return aux_cache[indices]
        left = get_prefix_conjunction(indices[:-1])
        right = xs[indices[-1]]
        aux_name = "Aux_" + "_".join(str(idx) for idx in indices)
        aux_cls = adapter.create_class(f"{SYNTHETIC_NS}{aux_name}")
        mgr.addAxiom(
            ont,
            adapter.create_equivalent_classes(
                aux_cls, adapter.create_object_intersection_of(left, right)
            ),
        )
        aux_cache[indices] = aux_cls
        return aux_cls

    for i in range(1, k + 1):
        others = tuple(j for j in range(1, k + 1) if j != i)
        left = get_prefix_conjunction(others[:-1])
        right = xs[others[-1]]
        mgr.addAxiom(
            ont,
            adapter.create_equivalent_classes(
                as_[i], adapter.create_object_intersection_of(left, right)
            ),
        )

    for i in range(1, k + 1):
        for j in range(i + 1, k + 1):
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(as_[i], as_[j]), bot
                ),
            )

    z_cls = adapter.create_class(f"{SYNTHETIC_NS}Z")
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(z_cls))
    return ont


def verify_tk_ontology(ont, k: int) -> None:
    """Verify ``CanonicalModel`` satisfiability and Helly certificate invariants on ``T_k``.

    Parameters
    ----------
    ont : org.semanticweb.owlapi.model.OWLOntology
        Constructed ``T_k`` ontology.
    k : int
        Arity parameter ``k >= 3``.

    Guarantees
    ----------
    - Raises ``AssertionError`` if any ``X_i``, ``A_i``, or ``Z`` is unsatisfiable,
      if any ``(k - 1)``-subset of ``X_1..X_k`` is unsatisfiable, if the ``k``-fold
      conjunction ``X_1..X_k`` is satisfiable, or if certificate checks fail.

    Non-guarantees
    --------------
    - Does not write files to disk.
    """
    from mowl.geometric import CanonicalModel, certificate_triangles, helly_check, normalise

    no = normalise(ont)
    assert no.dropped_axioms == 0, f"Expected 0 dropped axioms for tk{k}, got {no.dropped_axioms}"
    cm = CanonicalModel.from_normalised(no)

    xs = [f"{SYNTHETIC_NS}X{i}" for i in range(1, k + 1)]
    as_ = [f"{SYNTHETIC_NS}A{i}" for i in range(1, k + 1)]
    for x in xs:
        assert cm.is_satisfiable(x), f"X class {x} is unsatisfiable in tk{k}"
    for a in as_:
        assert cm.is_satisfiable(a), f"A class {a} is unsatisfiable in tk{k}"
    assert cm.is_satisfiable(f"{SYNTHETIC_NS}Z"), f"Z is unsatisfiable in tk{k}"

    for i in range(k):
        subset = [xs[j] for j in range(k) if j != i]
        assert cm.joint_satisfiable(subset), f"(k-1)-subset {subset} unsatisfiable in tk{k}"
    assert not cm.joint_satisfiable(xs), f"Full k-tuple {xs} is satisfiable in tk{k}"

    tris = certificate_triangles(cm)
    if k == 3:
        assert len(tris) == 1, f"Expected 1 certificate triangle on tk3, got {len(tris)}"
    else:
        assert tris == [], f"Expected 0 certificate triangles on tk{k}, got {tris}"

    rep = helly_check(cm, k_max=k, max_certificates=100)
    assert len(rep.certificates_nondegenerate) > 0, (
        f"Expected non-empty certificates_nondegenerate on tk{k}"
    )


def generate_all_synthetic(
    out_dir: Path = SYNTHETIC_DIR, ks: Iterable[int] = (3, 4, 5, 6)
) -> Dict[str, Path]:
    """Build, verify, and save ``tk{k}.owl`` for every ``k`` in ``ks``.

    Parameters
    ----------
    out_dir : Path, optional
        Output directory (default: ``experiments/data/synthetic``).
    ks : Iterable[int], optional
        Values of ``k`` to generate (default: ``(3, 4, 5, 6)``).

    Returns
    -------
    Dict[str, Path]
        Mapping ``"tk{k}" -> saved OWL path``.

    Guarantees
    ----------
    - Saves valid OWL files verified against ``CanonicalModel`` and reloads each
      via ``PathDataset`` to confirm compatibility.

    Non-guarantees
    --------------
    - Overwrites existing ``tk{k}.owl`` files in ``out_dir``.
    """
    mowl.init_jvm("4g")
    from org.semanticweb.owlapi.model import IRI
    from mowl.datasets import PathDataset
    from mowl.owlapi import OWLAPIAdapter

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    for k in ks:
        name = f"tk{k}"
        ont = build_tk_ontology(k)
        verify_tk_ontology(ont, k)
        out_path = out_dir / f"{name}.owl"
        ont.getOWLOntologyManager().saveOntology(
            ont, IRI.create("file:" + str(out_path.resolve()))
        )
        # Verify reload via PathDataset
        ds = PathDataset(str(out_path))
        verify_tk_ontology(ds.ontology, k)
        saved[name] = out_path
    return saved


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for generating synthetic ``T_k`` OWL datasets."""
    parser = argparse.ArgumentParser(description="Generate synthetic Helly T_k ontologies.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SYNTHETIC_DIR,
        help="Output directory for tk{k}.owl files.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Values of k to generate.",
    )
    args = parser.parse_args(argv)
    saved = generate_all_synthetic(out_dir=args.out_dir, ks=args.ks)
    for name, path in saved.items():
        print(f"Generated and verified {name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
