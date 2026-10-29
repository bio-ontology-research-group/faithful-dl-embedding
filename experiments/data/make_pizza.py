#!/usr/bin/env python3
"""Build the Pizza-EL subsumption benchmark split from pizza.owl (Spec §5.1, §7).

Takes ``~/tmp/gfe-data/pizza/pizza.owl``, extracts its normalised EL++ fragment via
``mowl.geometric.normalise``, computes the full atomic deductive closure via
``CanonicalModel``, holds out raw asserted atomic subsumptions ``A ⊑ B``
(``SubClassOf(OWLClass, OWLClass)`` in ``pizza.owl``) between distinct satisfiable
non-auxiliary named classes (seed 0; 10 %/10 % raised to 20 %/20 % when 10 % yields
fewer than 20 test axioms, per Spec §7), and writes:
- ``ontology.owl`` (training TBox and RBox, declaring the full EL signature)
- ``valid.owl`` (held-out validation atomic subsumptions)
- ``test.owl`` (held-out test atomic subsumptions)
- ``ontology_deductive_closure.owl`` (all non-trivial entailed atomic subsumptions
  between satisfiable non-auxiliary concept names in the full EL fragment)

Guarantees
----------
- Held-out candidates are strictly raw asserted ``SubClassOf(named, named)`` axioms
  of ``pizza.owl`` over satisfiable non-auxiliary classes (never ``NF1`` projections
  of ``EquivalentClasses`` definitions such as ``SpicyPizza ⊑ Pizza``).
- Training, validation, and testing atomic subsumptions are pairwise disjoint.
- Every validation and testing axiom is entailed by the canonical model of the
  full EL fragment and present in ``ontology_deductive_closure.owl``.
- No auxiliary normalisation class (``http://mowl.borg/aux/...`` or ``...aux#...``)
  occurs in ``valid.owl``, ``test.owl``, or ``ontology_deductive_closure.owl``.
- Every class appearing in ``valid.owl``, ``test.owl``, or
  ``ontology_deductive_closure.owl`` occurs in ``ontology.owl``.
- Copies generated files into ``experiments/data/pizza-el/`` when their combined
  size is under 1 MB.

Non-guarantees / Protocol Notes
-------------------------------
- Does not preserve axioms outside the EL++ fragment supported by
  ``mowl.geometric.normalise`` (dropped axioms are counted in metadata).
- ``ontology_deductive_closure.owl`` intentionally omits all subsumptions involving
  unsatisfiable named classes (``CheeseyVegetableTopping`` and ``IceCream``), as well
  as ``owl:Thing``, ``owl:Nothing``, and auxiliary normalisation classes (M2).
- Records the number of held-out validation and test axioms still entailed by the
  training ontology alone (trivial positives, M1: ``0`` validation and ``0`` test
  axioms under the I2 asserted-subsumption split).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any, Dict, List

import mowl
import numpy as np

from experiments.config import PIZZA_EL_DIR, PIZZA_RAW_PATH, PIZZA_REPO_DIR


def is_auxiliary_iri(iri: str) -> bool:
    """Return ``True`` iff ``iri`` is an auxiliary entity introduced during normalisation.

    Parameters
    ----------
    iri : str
        Entity IRI string.

    Returns
    -------
    bool
        ``True`` if ``"/aux"`` or ``"aux#"`` occurs in ``iri``.

    Guarantees
    ----------
    - Matches both ``mowl.ontology.normalize`` (``http://mowl.borg/aux/<hash>#...``)
      and ``mowl.geometric.normal_forms`` (``http://mowl.borg/geo/aux#...``) IRIs.

    Non-guarantees
    --------------
    - Assumes original domain ontologies do not use ``/aux`` or ``aux#`` in named
      entity IRIs.
    """
    return "/aux" in iri or "aux#" in iri


def build_pizza_el(
    raw_pizza_path: Path = PIZZA_RAW_PATH,
    out_dir: Path = PIZZA_EL_DIR,
    repo_mirror_dir: Path | None = PIZZA_REPO_DIR,
    seed: int = 0,
    initial_frac: float = 0.10,
    fallback_frac: float = 0.20,
    min_test_axioms: int = 20,
) -> Dict[str, Any]:
    """Build the Pizza-EL dataset files and return summary metadata.

    Parameters
    ----------
    raw_pizza_path : Path, optional
        Path to raw ``pizza.owl`` (default: ``~/tmp/gfe-data/pizza/pizza.owl``).
    out_dir : Path, optional
        Output directory for generated OWL files (default: ``~/tmp/gfe-data/pizza-el``).
    repo_mirror_dir : Path or None, optional
        If provided and total generated OWL size is ``< 1 MB``, mirror the generated
        files into this repository directory for committing.
    seed : int, optional
        Random seed for the train/valid/test permutation (default: ``0``).
    initial_frac : float, optional
        Initial validation and test split fraction (default: ``0.10``).
    fallback_frac : float, optional
        Fallback validation and test split fraction used when ``initial_frac`` yields
        fewer than ``min_test_axioms`` test axioms (default: ``0.20``).
    min_test_axioms : int, optional
        Minimum test axiom threshold before raising split fraction (default: ``20``).

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary with split counts, fractions, file paths, and byte sizes.

    Guarantees
    ----------
    - Deterministic given ``seed`` and ``raw_pizza_path``.
    - Writes ``ontology.owl``, ``valid.owl``, ``test.owl``, and
      ``ontology_deductive_closure.owl`` into ``out_dir``.

    Non-guarantees
    --------------
    - Overwrites existing OWL files in ``out_dir`` (and ``repo_mirror_dir``).
    """
    mowl.init_jvm("4g")
    from org.semanticweb.owlapi.model import (
        AxiomType,
        ClassExpressionType,
        IRI,
        OWLOntologyID,
        SetOntologyID,
    )
    from mowl.datasets import PathDataset
    from mowl.geometric import (
        BOTTOM,
        NF1,
        TOP,
        CanonicalModel,
        NormalisedOntology,
        normalise,
        to_owl,
    )

    raw_pizza_path = Path(raw_pizza_path).expanduser().resolve()
    if not raw_pizza_path.exists():
        raise FileNotFoundError(f"Raw pizza.owl not found at {raw_pizza_path}")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_raw = PathDataset(str(raw_pizza_path))
    no_full = normalise(ds_raw)
    cm_full = CanonicalModel.from_normalised(no_full)

    # I1 & M2: Satisfiable non-auxiliary concept names only (excludes TOP, BOTTOM,
    # unsatisfiable named classes CheeseyVegetableTopping and IceCream, and
    # auxiliary normaliser classes under http://mowl.borg/aux/).
    sat_non_aux_names = sorted(
        n
        for n in (set(cm_full.names) - {TOP, BOTTOM})
        if not is_auxiliary_iri(n)
    )
    sat_non_aux_set = set(sat_non_aux_names)

    # I2: Extract raw asserted SubClassOf(named, named) axioms from pizza.owl (84 exist)
    # and intersect with satisfiable non-auxiliary concept names (never NF1s that come
    # from normalising EquivalentClasses definitions).
    raw_asserted_nf1 = set()
    for owl_ax in ds_raw.ontology.getAxioms(AxiomType.SUBCLASS_OF):
        sub_expr = owl_ax.getSubClass()
        sup_expr = owl_ax.getSuperClass()
        if (
            sub_expr.getClassExpressionType() == ClassExpressionType.OWL_CLASS
            and sup_expr.getClassExpressionType() == ClassExpressionType.OWL_CLASS
        ):
            s_iri = str(sub_expr.asOWLClass().toStringID())
            t_iri = str(sup_expr.asOWLClass().toStringID())
            raw_asserted_nf1.add(NF1(s_iri, t_iri))

    full_axiom_set = set(no_full.axioms)
    candidates = sorted(
        {
            ax
            for ax in raw_asserted_nf1
            if ax in full_axiom_set
            and ax.sub in sat_non_aux_set
            and ax.sup in sat_non_aux_set
            and ax.sub != ax.sup
        },
        key=lambda a: (a.sub, a.sup),
    )

    n_cand = len(candidates)
    used_frac = initial_frac
    n_val = int(round(used_frac * n_cand))
    n_test = int(round(used_frac * n_cand))
    raised_fraction = False
    if n_test < min_test_axioms:
        used_frac = fallback_frac
        n_val = int(round(used_frac * n_cand))
        n_test = int(round(used_frac * n_cand))
        raised_fraction = True

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cand)
    valid_axioms = sorted(
        [candidates[i] for i in perm[:n_val]], key=lambda a: (a.sub, a.sup)
    )
    test_axioms = sorted(
        [candidates[i] for i in perm[n_val : n_val + n_test]],
        key=lambda a: (a.sub, a.sup),
    )
    held_out = set(valid_axioms) | set(test_axioms)
    train_axioms = [ax for ax in no_full.axioms if ax not in held_out]

    # M1: Count held-out validation/test axioms still entailed by the training ontology
    no_train = NormalisedOntology(
        axioms=train_axioms,
        concept_names=set(no_full.concept_names),
        role_names=set(no_full.role_names),
        dropped_axioms=0,
    )
    cm_train = CanonicalModel.from_normalised(no_train)
    n_valid_entailed_by_train = sum(
        1 for ax in valid_axioms if cm_train.entails(ax.sub, ax.sup)
    )
    n_test_entailed_by_train = sum(
        1 for ax in test_axioms if cm_train.entails(ax.sub, ax.sup)
    )

    # I1 & M2: Full EL fragment deductive closure over satisfiable non-auxiliary names
    dc_axioms = [
        NF1(a, b)
        for a in sat_non_aux_names
        for b in sat_non_aux_names
        if a != b and cm_full.entails(a, b)
    ]

    train_ont = to_owl(no_train)
    valid_ont = to_owl(
        NormalisedOntology(
            axioms=valid_axioms,
            concept_names={ax.sub for ax in valid_axioms}
            | {ax.sup for ax in valid_axioms},
            role_names=set(),
            dropped_axioms=0,
        )
    )
    test_ont = to_owl(
        NormalisedOntology(
            axioms=test_axioms,
            concept_names={ax.sub for ax in test_axioms}
            | {ax.sup for ax in test_axioms},
            role_names=set(),
            dropped_axioms=0,
        )
    )
    dc_ont = to_owl(
        NormalisedOntology(
            axioms=dc_axioms,
            concept_names={ax.sub for ax in dc_axioms} | {ax.sup for ax in dc_axioms},
            role_names=set(),
            dropped_axioms=0,
        )
    )

    files = {
        "ontology": (out_dir / "ontology.owl", train_ont),
        "valid": (out_dir / "valid.owl", valid_ont),
        "test": (out_dir / "test.owl", test_ont),
        "deductive_closure": (out_dir / "ontology_deductive_closure.owl", dc_ont),
    }
    paths_out: Dict[str, str] = {}
    sizes_out: Dict[str, int] = {}
    for key, (path, ont) in files.items():
        mgr = ont.getOWLOntologyManager()
        mgr.applyChange(
            SetOntologyID(
                ont, OWLOntologyID(IRI.create(f"http://mowl.borg/geo/pizza-el/{key}"))
            )
        )
        mgr.saveOntology(
            ont, IRI.create("file:" + str(path.resolve()))
        )
        paths_out[key] = str(path)
        sizes_out[key] = path.stat().st_size

    total_bytes = sum(sizes_out.values())
    mirrored = False
    if repo_mirror_dir is not None and total_bytes < 1_000_000:
        repo_mirror_dir = Path(repo_mirror_dir).expanduser().resolve()
        repo_mirror_dir.mkdir(parents=True, exist_ok=True)
        for path, _ in files.values():
            dst = repo_mirror_dir / path.name
            if dst.resolve() != path.resolve():
                shutil.copy2(path, dst)
        mirrored = True

    return {
        "raw_pizza_path": str(raw_pizza_path),
        "out_dir": str(out_dir),
        "repo_mirror_dir": str(repo_mirror_dir) if mirrored else None,
        "seed": seed,
        "initial_frac": initial_frac,
        "used_frac": used_frac,
        "raised_fraction": raised_fraction,
        "n_full_axioms": len(no_full.axioms),
        "n_dropped_axioms": no_full.dropped_axioms,
        "n_raw_asserted_nf1": len(raw_asserted_nf1),
        "n_sat_non_aux_names": len(sat_non_aux_names),
        "n_candidate_nf1": n_cand,
        "n_train_axioms": len(train_axioms),
        "n_train_nf1": sum(isinstance(ax, NF1) for ax in train_axioms),
        "n_valid": len(valid_axioms),
        "n_test": len(test_axioms),
        "n_valid_entailed_by_train": n_valid_entailed_by_train,
        "n_test_entailed_by_train": n_test_entailed_by_train,
        "n_deductive_closure": len(dc_axioms),
        "paths": paths_out,
        "sizes_bytes": sizes_out,
        "total_bytes": total_bytes,
    }


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for building the Pizza-EL subsumption benchmark."""
    parser = argparse.ArgumentParser(description="Build Pizza-EL subsumption split.")
    parser.add_argument(
        "--raw-pizza",
        type=Path,
        default=PIZZA_RAW_PATH,
        help="Path to raw pizza.owl.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PIZZA_EL_DIR,
        help="Output directory for Pizza-EL OWL files.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for splitting."
    )
    args = parser.parse_args(argv)
    info = build_pizza_el(
        raw_pizza_path=args.raw_pizza, out_dir=args.out_dir, seed=args.seed
    )
    print(
        f"Built Pizza-EL in {info['out_dir']} "
        f"(raw_asserted={info['n_raw_asserted_nf1']}, "
        f"candidates={info['n_candidate_nf1']}, frac={info['used_frac']:.2f}, "
        f"raised={info['raised_fraction']}, train_axioms={info['n_train_axioms']}, "
        f"valid={info['n_valid']} (entailed_by_train={info['n_valid_entailed_by_train']}), "
        f"test={info['n_test']} (entailed_by_train={info['n_test_entailed_by_train']}), "
        f"deductive_closure={info['n_deductive_closure']}, "
        f"total_bytes={info['total_bytes']})"
    )


if __name__ == "__main__":
    main()
