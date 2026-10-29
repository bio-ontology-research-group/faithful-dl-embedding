#!/usr/bin/env python3
"""Recompute and write experiments/results/dataset_stats.md (Task 6, I4).

Computes dataset statistics across all ``DATASETS`` (``tk3``, ``tk4``, ``tk5``,
``tk6``, ``pizza``, ``go``, ``foodon``) from ``experiments.datasets.load_dataset``
and ``mowl.geometric``, using ``experiments.cm_cache.get_canonical_model`` when
available on the branch (guarded import) and falling back to
``CanonicalModel.from_normalised`` otherwise.

Guarantees
----------
- Writes ``experiments/results/dataset_stats.md`` with the main statistics table,
  file sizes and timing breakdown, and protocol notes (including I1, I2, M1, M2).
- Uses ``helly_check(cm, k_max=k)`` for ``tk*`` and ``k_max=4`` for ``pizza``.

Non-guarantees
--------------
- Does not run ``helly_check`` on ``go`` or ``foodon``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence

import mowl

from experiments.config import DATASETS, PIZZA_RAW_PATH, RESULTS_DIR
from experiments.data.make_pizza import is_auxiliary_iri
from experiments.datasets import load_dataset


def _format_mb(n_bytes: int | None) -> str:
    """Format byte size as ``'<bytes> (<mb> MB)'`` or ``'—'``."""
    if n_bytes is None:
        return "—"
    mb = n_bytes / (1024.0 * 1024.0)
    if mb < 0.1:
        return f"{n_bytes:,} ({mb:.3f} MB)"
    return f"{n_bytes:,} ({mb:.2f} MB)"


def compute_dataset_row(name: str) -> Dict[str, Any]:
    """Compute statistics and timings for a single dataset identifier.

    Parameters
    ----------
    name : str
        Dataset identifier in ``DATASETS``.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed counts, byte sizes, and timings.
    """
    mowl.init_jvm("8g")
    from org.semanticweb.owlapi.model.parameters import Imports
    from mowl.datasets import PathDataset
    from mowl.geometric import (
        BOTTOM,
        NF1,
        NF2,
        NF3,
        NF4,
        NF6,
        NF7,
        TOP,
        CanonicalModel,
        certificate_triangles,
        helly_check,
        normalise,
    )
    from mowl.projection import TaxonomyProjector

    try:
        from experiments.cm_cache import get_canonical_model as _get_cached_cm
    except ImportError:
        _get_cached_cm = None

    t0 = time.perf_counter()
    bundle = load_dataset(name)
    load_time = time.perf_counter() - t0
    ds = bundle.dataset

    t0 = time.perf_counter()
    no = normalise(ds.ontology)
    norm_time = time.perf_counter() - t0

    nf1 = sum(isinstance(a, NF1) for a in no.axioms)
    nf2 = sum(isinstance(a, NF2) for a in no.axioms)
    nf3 = sum(isinstance(a, NF3) for a in no.axioms)
    nf4 = sum(isinstance(a, NF4) for a in no.axioms)
    nf6 = sum(isinstance(a, NF6) for a in no.axioms)
    nf7 = sum(isinstance(a, NF7) for a in no.axioms)

    bot_nf1 = sum(isinstance(a, NF1) and a.sup == BOTTOM for a in no.axioms)
    bot_nf2 = sum(isinstance(a, NF2) and a.sup == BOTTOM for a in no.axioms)
    bot_nf4 = sum(isinstance(a, NF4) and a.sup == BOTTOM for a in no.axioms)
    bot_total = bot_nf1 + bot_nf2 + bot_nf4

    t0 = time.perf_counter()
    if _get_cached_cm is not None:
        cm = _get_cached_cm(bundle)
    else:
        cm = CanonicalModel.from_normalised(no)
    cm_time = time.perf_counter() - t0

    sat_names = len(cm.names)
    unsat_names = len((set(no.concept_names) - {BOTTOM}) - set(cm.names))

    t0 = time.perf_counter()
    tris = certificate_triangles(cm)
    tri_time = time.perf_counter() - t0
    n_tri = len(tris)

    helly_str = "—"
    if name.startswith("tk"):
        k = int(name[2:])
        rep = helly_check(cm, k_max=k, max_certificates=100)
        helly_str = f"{len(rep.certificates_nondegenerate)} / {len(rep.certificates_strong)}"
    elif name == "pizza":
        rep = helly_check(cm, k_max=4, max_certificates=100)
        helly_str = (
            f"{len(rep.certificates_nondegenerate)} / {len(rep.certificates_strong)} (`k_max=4`)"
        )

    dropped_str = f"{no.dropped_axioms:,}"
    if name == "pizza" and PIZZA_RAW_PATH.expanduser().exists():
        no_raw = normalise(PathDataset(str(PIZZA_RAW_PATH.expanduser().resolve())))
        dropped_str = (
            f"{no.dropped_axioms} ({no_raw.dropped_axioms} dropped when extracting full EL fragment)"
        )

    if bundle.task == "satisfaction":
        split_str = f"{len(no.axioms):,} / 0 / 0"
        dc_str = "—"
        n_valid_entailed_by_train = None
        n_test_entailed_by_train = None
    else:
        n_val = ds.validation.getTBoxAxioms(Imports.EXCLUDED).size()
        n_test = ds.testing.getTBoxAxioms(Imports.EXCLUDED).size()
        n_dc = bundle.deductive_closure_ontology.getTBoxAxioms(Imports.EXCLUDED).size()
        n_tbox = ds.ontology.getTBoxAxioms(Imports.EXCLUDED).size()
        if name == "pizza":
            split_str = f"{len(no.axioms):,} / {n_val:,} / {n_test:,}"
            proj = TaxonomyProjector()
            val_pairs = [(str(e.src), str(e.dst)) for e in proj.project(ds.validation)]
            test_pairs = [(str(e.src), str(e.dst)) for e in proj.project(ds.testing)]
            n_valid_entailed_by_train = sum(1 for s, t in val_pairs if cm.entails(s, t))
            n_test_entailed_by_train = sum(1 for s, t in test_pairs if cm.entails(s, t))
        else:
            split_str = (
                f"{len(no.axioms):,} ({n_tbox:,} OWL TBox) / {n_val:,} / {n_test:,}"
            )
            n_valid_entailed_by_train = None
            n_test_entailed_by_train = None
        dc_str = f"{n_dc:,}"

    sizes: Dict[str, int | None] = {
        "tarfile": None,
        "ontology": None,
        "valid": None,
        "test": None,
        "deductive_closure": None,
    }
    for key in sizes:
        if key in bundle.paths and Path(bundle.paths[key]).exists():
            sizes[key] = Path(bundle.paths[key]).stat().st_size

    n_aux_train = sum(1 for c in no.concept_names if is_auxiliary_iri(c))

    return {
        "name": name,
        "task": bundle.task,
        "ds_classes": len(ds.classes),
        "norm_concepts": len(no.concept_names),
        "n_aux_train": n_aux_train,
        "roles": len(no.role_names),
        "total_norm_axioms": len(no.axioms),
        "nf1": nf1,
        "nf2": nf2,
        "nf3": nf3,
        "nf4": nf4,
        "nf6": nf6,
        "nf7": nf7,
        "bot_str": f"{bot_total:,} ({bot_nf1} / {bot_nf2:,} / {bot_nf4})",
        "dropped_str": dropped_str,
        "sat_names": sat_names,
        "unsat_names": unsat_names,
        "n_tri": n_tri,
        "helly_str": helly_str,
        "split_str": split_str,
        "dc_str": dc_str,
        "n_valid_entailed_by_train": n_valid_entailed_by_train,
        "n_test_entailed_by_train": n_test_entailed_by_train,
        "sizes": sizes,
        "load_time": load_time,
        "norm_time": norm_time,
        "cm_time": cm_time,
        "tri_time": tri_time,
    }


def render_dataset_stats_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    """Render ``experiments/results/dataset_stats.md`` content from computed rows."""
    lines: List[str] = [
        "# Experiment Dataset Statistics (`Task 6`)",
        "",
        "Generated by `experiments/data/dataset_stats.py` from the training ontologies of the 7 datasets (`DATASETS = [\"tk3\", \"tk4\", \"tk5\", \"tk6\", \"pizza\", \"go\", \"foodon\"]`) using `experiments.datasets.load_dataset` and `mowl.geometric` (`normalise`, `CanonicalModel`, `certificate_triangles`, and `helly_check`).",
        "",
        "## 1. Main Dataset Statistics Table",
        "",
        "| Dataset | Task | Dataset Classes (`len(ds.classes)`) | Normalised Concept Names (incl. `⊤`, `⊥`, aux) | Roles (`len(no.role_names)`) | Total Normalised Train Axioms | NF1 (`A ⊑ B` / `⊥`) | NF2 (`A1 ⊓ A2 ⊑ B` / `⊥`) | NF3 (`A ⊑ ∃r.B`) | NF4 (`∃r.A ⊑ B` / `⊥`) | NF6 (`r ⊑ s`) | NF7 (`r1 ∘ r2 ⊑ s`) | `⊥`-Axioms (NF1 / NF2 / NF4) | Dropped Axioms (Train) | CM Satisfiable Names (`len(cm.names)`) | CM Unsat Names | `n_certificate_triangles` (Train) | `helly_check` Nondeg / Strong Certs (`k_max=k`) | Train / Valid / Test Split Sizes | Deductive Closure Axioms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| `{r['name']}` | `{r['task']}` | {r['ds_classes']:,} | {r['norm_concepts']:,} | "
            f"{r['roles']:,} | {r['total_norm_axioms']:,} | {r['nf1']:,} | {r['nf2']:,} | "
            f"{r['nf3']:,} | {r['nf4']:,} | {r['nf6']:,} | {r['nf7']:,} | "
            f"{r['bot_str']} | {r['dropped_str']} | {r['sat_names']:,} | {r['unsat_names']:,} | "
            f"{r['n_tri']:,} | {r['helly_str']} | {r['split_str']} | {r['dc_str']} |"
        )

    lines.extend(
        [
            "",
            "## 2. File Sizes and Timing Breakdown",
            "",
            "| Dataset | Tarball Size (bytes / MB) | `ontology.owl` Size (bytes / MB) | `valid.owl` Size (bytes / MB) | `test.owl` Size (bytes / MB) | `ontology_deductive_closure.owl` Size (bytes / MB) | Load Time (s) | Normalise Time (s) | CM Saturation Time (s) | `certificate_triangles` Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for r in rows:
        s = r["sizes"]
        lines.append(
            f"| `{r['name']}` | {_format_mb(s['tarfile'])} | {_format_mb(s['ontology'])} | "
            f"{_format_mb(s['valid'])} | {_format_mb(s['test'])} | "
            f"{_format_mb(s['deductive_closure'])} | {r['load_time']:.2f} | "
            f"{r['norm_time']:.2f} | {r['cm_time']:.4f} | {r['tri_time']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Protocol Notes & Recorded Rulings",
            "",
            "1. **Synthetic `T_k` (`tk4`–`tk6`) Certificate Structure**:",
            "   - In `T_k` for `k >= 4`, the ontology defines the `(k - 1)`-wise conjunctions `A_i ≡ ⊓_{j != i} X_j` (binary-normalised with shared prefix auxiliary equivalences `Aux_1_2 ≡ X_1 ⊓ X_2`, etc.) rather than naming all three pairwise edges of an unsatisfiable 3-clique, and every 3-subset of `{X_1, ..., X_k}` is satisfiable.",
            "   - Consequently, `certificate_triangles(cm)` returns `[]` (`n_certificate_triangles = 0`) on `tk4`, `tk5`, and `tk6`, while `helly_check(cm, k_max=k)` finds the minimal unsatisfiable set of size `k` as a non-degenerate `HellyCertificate` with `c1=(X1,)`, `c2=(X2,)`, `c3=(X3, ..., Xk)` (a tuple of `k - 2` names) and `pairwise_names=(Aux_1_2, A2, A1)`.",
            "   - The Helly metrics in Task 7 therefore evaluate `helly_check` certificates with tuple `c3` on `tk4`–`tk6`.",
            "2. **Pizza-EL Asserted Subsumption Candidates & Split Fraction Ruling (Spec §5.1, §7; Rulings I1, I2)**:",
            "   - Normalising `~/tmp/gfe-data/pizza/pizza.owl` produces `662` normalised EL++ axioms (`90` `NF1`, `404` `NF2`, `153` `NF3`, `9` `NF4`, `4` `NF6`, `2` `NF7`; `52` non-EL++ axioms dropped) across `109` concept names (`97` satisfiable original named classes, `2` unsatisfiable named classes `CheeseyVegetableTopping` and `IceCream`, `8` auxiliary normaliser classes under `http://mowl.borg/aux/`, plus `owl:Thing` and `owl:Nothing`) and `8` roles.",
            "   - Per Ruling I2, held-out candidates are restricted to raw asserted `SubClassOf(named, named)` axioms in `pizza.owl` (`84` exist in `pizza.owl`) intersected with the `97` satisfiable non-auxiliary named classes. Excluding the `3` raw asserted subsumptions whose subclass is unsatisfiable (`CheeseyVegetableTopping ⊑ CheeseTopping`, `CheeseyVegetableTopping ⊑ VegetableTopping`, `IceCream ⊑ Food`) leaves `81` candidate asserted atomic subsumptions (and never includes the `5` `NF1` axioms produced by normalising `EquivalentClasses` definitions: `CheeseyPizza ⊑ Pizza`, `MeatyPizza ⊑ Pizza`, `SpicyPizza ⊑ Pizza`, `SpicyPizzaEquivalent ⊑ Pizza`, `SpicyTopping ⊑ PizzaTopping`).",
            "   - With `seed = 0`, an initial `10 % / 10 %` split yields `round(0.10 * 81) = 8 < 20` test axioms. Per Spec §7, the split fractions were raised to `20 % / 20 %`, yielding `16` validation axioms (`valid.owl`), `16` test axioms (`test.owl`), and `630` training axioms (`ontology.owl`, containing `58` `NF1` axioms).",
            "   - Because the four generated Pizza-EL OWL files total `299,592` bytes (`< 1 MB`), they are stored in `~/tmp/gfe-data/pizza-el/` and mirrored in `experiments/data/pizza-el/`. `load_dataset(\"pizza\")` prefers the committed mirror `experiments/data/pizza-el/` and verifies/syncs `~/tmp/gfe-data/pizza-el/` by SHA-256 checksum (Ruling M4).",
            "3. **Auxiliary Normaliser Classes & Training Deductive Closure Filtering (Rulings I1, M2, Task 8 Fix 1a)**:",
            "   - Per Ruling I1, the `8` auxiliary classes introduced by `mowl.ontology.normalize` (`http://mowl.borg/aux/...#AuxiliaryEntity...`) remain in the training ontology (`ontology.owl`) for model training, but are excluded from `ontology_deductive_closure.owl` and from `dataset.evaluation_classes` (`100` evaluation classes: `99` original Pizza named classes including `owl:Nothing` plus `owl:Thing`).",
            "   - Per Ruling M2 and Task 8 Fix 1(a) (spec correction matching mOWL's GO/FoodOn datasets), `ontology_deductive_closure.owl` (`224` non-trivial atomic subsumptions) is the deductive closure of the **training** ontology `ontology.owl` (`cm_train`) built exclusively over the `97` satisfiable non-auxiliary named classes (omitting the `2` unsatisfiable named classes `CheeseyVegetableTopping` and `IceCream`, `owl:Thing`, `owl:Nothing`, and auxiliary normaliser classes). Since `0 / 16` held-out validation and `0 / 16` held-out test axioms are entailed by `ontology.owl`, none appears in `ontology_deductive_closure.owl`.",
            "4. **Held-Out Pizza Axioms Still Entailed by the Training Ontology (Trivial Positives, Ruling M1)**:",
            "   - Under the Ruling I2 split (`81` raw asserted `SubClassOf(named, named)` candidates), `0 / 16` validation axioms and `0 / 16` test axioms are entailed by the training ontology `ontology.owl` (`cm_train`) alone (i.e., `0` trivial positives in `valid.owl` and `0` in `test.owl`, compared with `1 / 17` validation and `2 / 17` test axioms when equivalence-derived `NF1`s were previously eligible for hold-out).",
            "",
        ]
    )
    return "\n".join(lines)


def generate_dataset_stats(
    out_path: Path = RESULTS_DIR / "dataset_stats.md",
    datasets: Sequence[str] = DATASETS,
) -> Path:
    """Compute statistics for ``datasets`` and write markdown table to ``out_path``."""
    rows = [compute_dataset_row(name) for name in datasets]
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dataset_stats_markdown(rows), encoding="utf-8")
    return out_path


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for regenerating ``experiments/results/dataset_stats.md``."""
    parser = argparse.ArgumentParser(
        description="Recompute and write experiments/results/dataset_stats.md."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "dataset_stats.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Dataset identifiers to include.",
    )
    args = parser.parse_args(argv)
    out = generate_dataset_stats(out_path=args.out, datasets=args.datasets)
    print(f"Wrote dataset statistics to {out}")


if __name__ == "__main__":
    main()
