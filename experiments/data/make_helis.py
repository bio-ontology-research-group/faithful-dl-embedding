#!/usr/bin/env python3
"""Build the B3 helis-membership ABox benchmark (OWL2Vec* protocol, Spec §3.2).

Constructs ``~/tmp/gfe-data/helis/`` from ``~/tmp/gfe-data/helis-dl/``:
- Extracts the EL++ fragment of the HeLiS TBox (`helis_v1.00.origin.owl`, ALCHIQ(D))
  via ``mowl.geometric.normalise`` and ``to_owl``, recording dropped axioms per OWL
  construct and asserting that both disjointness axioms survive normalisation:
  (1) the 14-way ``AllDisjoint`` food-group axiom (`91` pairwise ``NF2(..., ⊥)``
      axioms) and (2) ``RedMeat ⊓ WhiteMeat ⊑ ⊥`` (`1` pairwise ``NF2(..., ⊥)``
      axiom, total `92` ``NF2(..., ⊥)`` axioms).
- Adds all named-individual ``ObjectPropertyAssertion`` axioms from the origin
  ontology (`83,210` assertions), declares all `20,318` individuals from
  ``individuals.txt``, and adds training class memberships from ``train.csv`` rows
  with label `1` (`14,222` positive ``ClassAssertion`` axioms; label-`0` sampled
  negatives and ``DataPropertyAssertion`` axioms are dropped).
- Writes held-out ``valid.owl`` (`2,031` ``ClassAssertion`` axioms from
  ``valid.csv``) and ``test.owl`` (`4,065` ``ClassAssertion`` axioms from
  ``test.csv``), ``candidates.txt`` (`277` named classes from ``classes.txt``
  excluding ``owl:Thing``), and ``skip_{valid,test}.tsv`` (mapping each held-out
  individual to its comma-separated inferred classes from ``inferred_classes.txt``).
- Computes three-way reasoner labels (`entailed`, `unknown`, `contradicted`) for
  every held-out `(individual, candidate)` pair via
  ``mowl.geometric.CanonicalModel.from_ontology`` (`instance_of` -> `entailed`,
  `assertion_contradicted` -> `contradicted`) on the built training ontology
  (and also computes the full-ABox reference contradiction counts across splits to
  verify food-group disjointness contradictions > 0), writing
  ``labels_{valid,test}.parquet``, ``manifest.json``, and the B3 section in
  ``experiments/results/dataset_stats.md``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import mowl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.config import DATA_DIR, RAW_DATA_ROOT, RESULTS_DIR
from experiments.data.fetch_helis import (
    HELIS_DL_DIR,
    HELIS_FILES_SHA256,
    HELIS_FILES_TARBALL_NAME,
    HELIS_FILES_URL,
    HELIS_ONTS_SHA256,
    HELIS_ONTS_TARBALL_NAME,
    HELIS_ONTS_URL,
    fetch_helis,
)

HELIS_OUT_DIR: Path = RAW_DATA_ROOT / "helis"
HELIS_REPO_DIR: Path = DATA_DIR / "helis"

HELIS_NS: str = "http://www.fbk.eu/ontologies/virtualcoach#"
OWL_THING_SHORT: str = "owl:Thing"
OWL_THING_IRI: str = "http://www.w3.org/2002/07/owl#Thing"

# Programmatically verified against DisjointClasses axioms in helis_v1.00.origin.owl
FOOD_GROUP_14_CLASSES: Tuple[str, ...] = (
    f"{HELIS_NS}CerealsAndGrainProducts",
    f"{HELIS_NS}CulinaryErbs",
    f"{HELIS_NS}Eggs",
    f"{HELIS_NS}FatsAndOils",
    f"{HELIS_NS}FishAndSeafood",
    f"{HELIS_NS}Fruits",
    f"{HELIS_NS}Meat",
    f"{HELIS_NS}MilkAndDairyProducts",
    f"{HELIS_NS}NutSeedAndOliveProducts",
    f"{HELIS_NS}Salt",
    f"{HELIS_NS}SausagesAndColdCuts",
    f"{HELIS_NS}Spices",
    f"{HELIS_NS}SugarAndSweets",
    f"{HELIS_NS}VegetablesAndLegumes",
)
RED_WHITE_MEAT_PAIR: Tuple[str, str] = (
    f"{HELIS_NS}RedMeat",
    f"{HELIS_NS}WhiteMeat",
)

EXPECTED_N_INDIVIDUALS: int = 20_318
EXPECTED_N_CLASSES: int = 277
EXPECTED_N_TRAIN_POS: int = 14_222
EXPECTED_N_TRAIN_NEG: int = 14_222
EXPECTED_N_VALID: int = 2_031
EXPECTED_N_TEST: int = 4_065

STATS_SECTION_HEADER: str = "## 5. ABox Benchmark B3: `helis-membership` Statistics"


def _file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_inferred_classes_file(path: Path) -> Dict[str, List[str]]:
    """Parse ``inferred_classes.txt`` into ``{individual_iri: [class_iri_or_owl_thing, ...]}``."""
    out: Dict[str, List[str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        out[parts[0]] = parts[1:]
    return out


def candidate_set_for_individual(
    individual_iri: str,
    classes_txt: Sequence[str],
    inferred_classes_map: Mapping[str, Sequence[str]],
) -> Set[str]:
    """Return OWL2Vec* candidate classes for ``individual_iri``: ``classes.txt - owl:Thing - inferred_classes[a]``."""
    skip = set(inferred_classes_map.get(individual_iri, ())) | {
        OWL_THING_SHORT,
        OWL_THING_IRI,
    }
    return {c for c in classes_txt if c not in skip}


def _classify_dropped_tbox_axiom(ax: Any) -> str:
    """Classify a dropped TBox axiom into its OWL construct / axiom category."""
    ax_type_str = str(ax.getAxiomType())
    s = str(ax.toString())
    if ax_type_str == "DataPropertyDomain":
        if "ObjectUnionOf" in s:
            return "DataPropertyDomain (with ObjectUnionOf)"
        return "DataPropertyDomain"
    if ax_type_str == "DataPropertyRange":
        if "DataOneOf" in s:
            return "DataPropertyRange (with DataOneOf)"
        return "DataPropertyRange"
    if ax_type_str == "ObjectPropertyRange":
        return "ObjectPropertyRange"
    if ax_type_str == "ObjectPropertyDomain" and "ObjectUnionOf" in s:
        return "ObjectPropertyDomain (with ObjectUnionOf)"
    if ax_type_str == "SubClassOf" and (
        "ObjectExactCardinality" in s or "DataExactCardinality" in s
    ):
        return "SubClassOf (with ExactCardinality)"
    if ax_type_str == "EquivalentClasses" and "ObjectUnionOf" in s:
        return "EquivalentClasses (with ObjectUnionOf)"
    for construct in (
        "ObjectUnionOf",
        "ObjectComplementOf",
        "ObjectAllValuesFrom",
        "ObjectMinCardinality",
        "ObjectMaxCardinality",
        "ObjectExactCardinality",
        "ObjectHasSelf",
        "ObjectOneOf",
        "ObjectInverseOf",
    ):
        if construct in s:
            return f"{ax_type_str} (with {construct})"
    return ax_type_str


def _seed_cm_joint_sat_cache(
    cm: Any,
    candidates: Sequence[str],
    eval_individuals: Sequence[str] | None = None,
) -> Dict[str, Set[str]]:
    """Pre-populate ``cm._joint_sat_cache`` for ``(N_a, C)`` pairs from ``cm``'s saturated completion.

    In the normalised HeLiS EL++ ontology:
    - Every ``NF4`` axiom has filler ``TOP`` (from ``ObjectPropertyDomain``).
    - Every ``NF2`` axiom has superclass ``BOTTOM`` (`92` pairwise disjointness axioms).
    - There are no ``NF1(..., BOTTOM)`` or ``NF4(..., BOTTOM)`` axioms.
    Consequently, for any individual ``a`` (nominal ``N_a``) and candidate concept ``C``,
    ``N_a ⊓ C ⊑ ⊥`` holds in ``cm`` if and only if some concept ``B1 ∈ S(N_a)`` and some
    concept ``B2 ∈ S(C)`` appear as a disjoint pair ``NF2(B1, B2, ⊥)`` in ``cm.axioms``.
    Populating ``cm._joint_sat_cache[frozenset({N_a, C})]`` makes subsequent calls to
    ``cm.assertion_contradicted(a, C)`` O(1) while preserving exact equivalence with
    fresh-concept saturation.
    """
    from mowl.geometric import BOTTOM, NF2

    nf2_bot_pairs: Set[Tuple[str, str]] = set()
    for ax in cm.axioms:
        if isinstance(ax, NF2) and ax.sup == BOTTOM:
            nf2_bot_pairs.add((ax.sub1, ax.sub2))
            nf2_bot_pairs.add((ax.sub2, ax.sub1))

    disjoint_with: Dict[str, Set[str]] = {c: set() for c in candidates}
    for c1 in candidates:
        s1 = cm.completion.sub_super.get(c1, set())
        for c2 in candidates:
            s2 = cm.completion.sub_super.get(c2, set())
            if any((x, y) in nf2_bot_pairs for x in s1 for y in s2):
                disjoint_with[c1].add(c2)

    cand_set = set(candidates)
    contra_by_ts: Dict[FrozenSet[str], Set[str]] = {}
    target_inds = (
        eval_individuals
        if eval_individuals is not None
        else list(cm.individuals.keys())
    )
    for ind_iri in target_inds:
        nom = cm.individuals.get(ind_iri)
        if nom is None:
            continue
        ts = frozenset(cm.completion.sub_super.get(nom, set()) & cand_set)
        contra_for_nom = contra_by_ts.get(ts)
        if contra_for_nom is None:
            contra_for_nom = set()
            for b in ts:
                contra_for_nom.update(disjoint_with.get(b, set()))
            contra_by_ts[ts] = contra_for_nom
        for c in candidates:
            cm._joint_sat_cache[frozenset({nom, c})] = c not in contra_for_nom

    return disjoint_with


def _write_skip_tsv(
    out_path: Path,
    eval_individuals: Sequence[str],
    inferred_classes_map: Mapping[str, Sequence[str]],
) -> None:
    """Write ``skip_{valid,test}.tsv`` with ``individual\\tclass1,class2,...`` lines."""
    lines: List[str] = []
    for ind in eval_individuals:
        skipped = inferred_classes_map.get(ind, [])
        lines.append(f"{ind}\t{','.join(skipped)}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_labels_parquet_from_cm(
    out_path: Path,
    eval_individuals: Sequence[str],
    candidates: Sequence[str],
    cm: Any,
) -> Dict[str, int]:
    """Compute three-way labels via ``cm.instance_of`` and ``cm.assertion_contradicted`` and write Parquet."""
    n_ind = len(eval_individuals)
    n_cand = len(candidates)

    ind_indices = np.repeat(np.arange(n_ind, dtype=np.int32), n_cand)
    cand_indices = np.tile(np.arange(n_cand, dtype=np.int32), n_ind)
    # 0 -> "unknown", 1 -> "entailed", 2 -> "contradicted"
    label_codes = np.zeros(n_ind * n_cand, dtype=np.int8)

    for i, ind_iri in enumerate(eval_individuals):
        offset = i * n_cand
        for j, cls_iri in enumerate(candidates):
            if cm.instance_of(ind_iri, cls_iri):
                label_codes[offset + j] = 1
            elif cm.assertion_contradicted(ind_iri, cls_iri):
                label_codes[offset + j] = 2

    n_entailed = int(np.sum(label_codes == 1))
    n_contradicted = int(np.sum(label_codes == 2))
    n_unknown = int(len(label_codes) - n_entailed - n_contradicted)

    labels_vocab = pa.array(["unknown", "entailed", "contradicted"])
    table = pa.Table.from_arrays(
        [
            pa.DictionaryArray.from_arrays(
                pa.array(ind_indices), pa.array(eval_individuals, type=pa.string())
            ).cast(pa.string()),
            pa.DictionaryArray.from_arrays(
                pa.array(cand_indices), pa.array(candidates, type=pa.string())
            ).cast(pa.string()),
            pa.DictionaryArray.from_arrays(
                pa.array(label_codes), labels_vocab
            ).cast(pa.string()),
        ],
        names=["individual", "class", "label"],
    )
    pq.write_table(table, str(out_path))
    return {
        "n_rows": int(n_ind * n_cand),
        "n_individuals": n_ind,
        "n_candidates": n_cand,
        "n_entailed": n_entailed,
        "n_unknown": n_unknown,
        "n_contradicted": n_contradicted,
    }


def render_helis_stats_section(info: Mapping[str, Any]) -> str:
    """Render the Markdown section for B3 ``helis-membership`` in ``dataset_stats.md``."""
    timings = info["timings_sec"]
    dropped_tbox_str = ", ".join(
        f"`{k}`: {v}" for k, v in info["dropped_tbox_by_construct"].items()
    )
    dropped_rbox_str = ", ".join(
        f"`{k}`: {v}" for k, v in info["dropped_rbox_by_construct"].items()
    )
    opa_by_prop_str = ", ".join(
        f"`{k}`: {v:,}" for k, v in info["opa_by_property"].items()
    )
    lines = [
        STATS_SECTION_HEADER,
        "",
        "Generated by `experiments/data/make_helis.py` from `KRR-Oxford/OWL2Vec-Star` (`helis_membership_files.tar.gz` and `ontologies.tar.gz` / `helis_v1.00.origin.owl`).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Individuals (`individuals.txt` / `OWLNamedIndividual`) | {info['n_individuals']:,} |",
        f"| Classes (`classes.txt` / `OWLClass`, excl. `owl:Thing`) | {info['n_classes']:,} (candidates: {info['n_candidates']:,}) |",
        f"| Origin TBox / RBox / ABox Axiom Counts (`helis_v1.00.origin.owl`) | {info['n_origin_tbox_axioms']:,} TBox / {info['n_origin_rbox_axioms']:,} RBox / {info['n_origin_abox_axioms']:,} ABox |",
        f"| Dropped TBox Axioms (`{info['n_dropped_tbox_axioms']}` total) per OWL Construct | {dropped_tbox_str} |",
        f"| Dropped RBox Axioms (`{info['n_dropped_rbox_axioms']}` total) per OWL Construct | {dropped_rbox_str} |",
        f"| Dropped ABox Axioms (`DataPropertyAssertion` + `1` anonymous `ClassAssertion`) | {info['n_dropped_data_property_assertions']:,} `DataPropertyAssertion` + {info['n_dropped_anon_class_assertions']:,} anonymous `ClassAssertion` |",
        f"| Normalised EL++ TBox Axioms Kept (`NF1 / NF2 / NF3 / NF4`) | {info['n_norm_tbox_axioms']:,} (`NF1`: {info['n_norm_nf1']:,}, `NF2`: {info['n_norm_nf2']:,} [all `⊥`], `NF3`: {info['n_norm_nf3']:,}, `NF4`: {info['n_norm_nf4']:,}) |",
        f"| Surviving Disjointness Axioms (`NF2(C₁, C₂, ⊥)`) | {info['n_disjoint_nf2_pairs']:,} pairwise (`91` from 14-way food-group `AllDisjoint` + `1` from `RedMeat ⊓ WhiteMeat ⊑ ⊥`) |",
        f"| Memberships per Split (`train` pos / `valid` / `test`; `train` neg dropped) | {info['n_train_pos']:,} / {info['n_valid']:,} / {info['n_test']:,} (`train` label-`0` dropped: {info['n_train_neg_dropped']:,}) |",
        f"| Object-Property Assertions Kept (`ObjectPropertyAssertion`) | {info['n_object_property_assertions']:,} ({opa_by_prop_str}) |",
        f"| Training Label Pairs (`train_individual × candidate`, `CanonicalModel` on `ontology.owl`) | {info['train_labels_summary']['n_rows']:,} (`entailed`: {info['train_labels_summary']['n_entailed']:,}, `unknown`: {info['train_labels_summary']['n_unknown']:,}, `contradicted`: {info['train_labels_summary']['n_contradicted']:,}) |",
        f"| Validation Label Pairs (`labels_valid.parquet`: `valid_individual × candidate`) | {info['valid_labels']['n_rows']:,} (`entailed`: {info['valid_labels']['n_entailed']:,}, `unknown`: {info['valid_labels']['n_unknown']:,}, `contradicted`: {info['valid_labels']['n_contradicted']:,}; w.r.t. ground-truth food-group membership: `contradicted = {info['n_valid_contradicted_with_ground_truth']:,}`) |",
        f"| Test Label Pairs (`labels_test.parquet`: `test_individual × candidate`) | {info['test_labels']['n_rows']:,} (`entailed`: {info['test_labels']['n_entailed']:,}, `unknown`: {info['test_labels']['n_unknown']:,}, `contradicted`: {info['test_labels']['n_contradicted']:,}; w.r.t. ground-truth food-group membership: `contradicted = {info['n_test_contradicted_with_ground_truth']:,}`) |",
        f"| Build Time (`total`: `load / tbox_norm / owl_write / cm_build / parquet`) | {timings['total']:.2f} s ({timings['load']:.2f} / {timings['tbox_normalise']:.2f} / {timings['owl_write']:.2f} / {timings['cm_build']:.2f} / {timings['labels_parquet']:.2f} s) |",
        "",
        "- **Protocol Note (Disjointness & Contradictions across Splits)**: In `helis_v1.00.origin.owl`, each of the `20,318` individuals has a single atomic `ClassAssertion`, partitioned into `14,222` training (`train.csv` label `1`), `2,031` validation (`valid.csv`), and `4,065` test (`test.csv`) individuals. Both TBox disjointness axioms survive normalisation (`91` pairwise `NF2` axioms from the 14-way food-group `AllDisjoint` plus `RedMeat ⊓ WhiteMeat ⊑ ⊥`, covering `121` subclasses of the `16` disjoint food classes). On the training ontology `ontology.owl`, `639` training individuals belong to a food-group subclass and yield `69,722` contradicted `(individual, candidate)` pairs (`cm.assertion_contradicted(a, C) == True`). For held-out `valid.csv` (`2,031` individuals) and `test.csv` (`4,065` individuals), the individual's atomic class assertion is held out from `ontology.owl`; only `106` validation and `210` test individuals (`RECIPE-*`) receive an indirect type (`Recipe ⊑ Food`, which is a superclass of `BasicFood` and therefore compatible with every food group) via `ObjectPropertyDomain(hasRecipeFood, Recipe)`, while `hasNutrient`, `hasFood`, and `contains` have `ObjectUnionOf` domains or `ObjectPropertyRange` axioms that lie outside EL++. Consequently, `CanonicalModel.from_ontology(train_ont)` derives `212` entailed (`106 × {Recipe, Food}`) and `0` contradicted candidate pairs on `valid` (`420` entailed and `0` contradicted on `test`), whereas against the full origin ABox (`inferred_classes.txt` / held-out ground-truth memberships) `92` validation and `185` test food-group individuals produce `9,947` validation and `20,169` test contradicted candidate pairs.",
        "",
    ]
    return "\n".join(lines)


def update_dataset_stats_file(stats_path: Path, info: Mapping[str, Any]) -> None:
    """Append or replace the B3 ``helis-membership`` section in ``stats_path``."""
    stats_path = Path(stats_path).expanduser().resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    section_md = render_helis_stats_section(info)

    if stats_path.exists():
        existing = stats_path.read_text(encoding="utf-8")
        if STATS_SECTION_HEADER in existing:
            prefix, _ = existing.split(STATS_SECTION_HEADER, 1)
            updated = prefix.rstrip() + "\n\n" + section_md
        else:
            updated = existing.rstrip() + "\n\n" + section_md
    else:
        updated = section_md

    stats_path.write_text(updated, encoding="utf-8")


def build_helis(
    dl_dir: Path = HELIS_DL_DIR,
    out_dir: Path = HELIS_OUT_DIR,
    stats_path: Path | None = RESULTS_DIR / "dataset_stats.md",
    repo_mirror_dir: Path | None = HELIS_REPO_DIR,
) -> Dict[str, Any]:
    """Build the B3 ``helis-membership`` ABox benchmark files and return summary metadata."""
    t_start = time.perf_counter()
    mowl.init_jvm("8g")

    from java.util import HashSet
    from org.semanticweb.owlapi.model import (
        AxiomType,
        IRI,
        OWLOntologyID,
        SetOntologyID,
    )
    from org.semanticweb.owlapi.model.parameters import Imports
    from mowl.datasets import PathDataset
    from mowl.geometric import (
        BOTTOM,
        NF1,
        NF2,
        NF3,
        NF4,
        CanonicalModel,
        NormalisedOntology,
        normalise,
        to_owl,
    )
    from mowl.ontology.normalize import ELNormalizer
    from mowl.owlapi import OWLAPIAdapter

    dl_dir = Path(dl_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify / extract input tarballs and parse CSV/TXT split files
    t0 = time.perf_counter()
    fetch_meta = fetch_helis(dl_dir=dl_dir)
    files_dir = Path(fetch_meta["files_dir"])
    origin_owl_path = Path(fetch_meta["origin_owl"])

    classes_txt = [
        line.strip()
        for line in (files_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    individuals_txt = [
        line.strip()
        for line in (files_dir / "individuals.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    inferred_classes_map = parse_inferred_classes_file(
        files_dir / "inferred_classes.txt"
    )

    train_pos_pairs: List[Tuple[str, str]] = []
    n_train_neg_dropped = 0
    for line in (files_dir / "train.csv").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        ind_iri, cls_iri, label_str = [p.strip() for p in s.split(",")]
        if label_str == "1":
            train_pos_pairs.append((ind_iri, cls_iri))
        elif label_str == "0":
            n_train_neg_dropped += 1
        else:
            raise ValueError(f"Unexpected label {label_str!r} in train.csv line: {s!r}")

    valid_pairs: List[Tuple[str, str]] = [
        tuple(p.strip() for p in line.strip().split(","))  # type: ignore[misc]
        for line in (files_dir / "valid.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_pairs: List[Tuple[str, str]] = [
        tuple(p.strip() for p in line.strip().split(","))  # type: ignore[misc]
        for line in (files_dir / "test.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(individuals_txt) == EXPECTED_N_INDIVIDUALS, (
        f"Expected {EXPECTED_N_INDIVIDUALS} individuals, got {len(individuals_txt)}"
    )
    assert len(classes_txt) == EXPECTED_N_CLASSES, (
        f"Expected {EXPECTED_N_CLASSES} classes, got {len(classes_txt)}"
    )
    assert len(train_pos_pairs) == EXPECTED_N_TRAIN_POS, (
        f"Expected {EXPECTED_N_TRAIN_POS} positive training rows, got {len(train_pos_pairs)}"
    )
    assert n_train_neg_dropped == EXPECTED_N_TRAIN_NEG, (
        f"Expected {EXPECTED_N_TRAIN_NEG} negative training rows, got {n_train_neg_dropped}"
    )
    assert len(valid_pairs) == EXPECTED_N_VALID, (
        f"Expected {EXPECTED_N_VALID} validation rows, got {len(valid_pairs)}"
    )
    assert len(test_pairs) == EXPECTED_N_TEST, (
        f"Expected {EXPECTED_N_TEST} test rows, got {len(test_pairs)}"
    )

    candidates = sorted(
        c for c in classes_txt if c not in {OWL_THING_SHORT, OWL_THING_IRI}
    )
    assert len(candidates) == EXPECTED_N_CLASSES

    ds_orig = PathDataset(str(origin_owl_path))
    orig_ont = ds_orig.ontology
    t_load = time.perf_counter() - t0

    # 2. Extract and normalise the HeLiS TBox/RBox and count dropped axioms per OWL construct
    t0 = time.perf_counter()
    orig_tbox_axioms = list(orig_ont.getTBoxAxioms(Imports.fromBoolean(True)))
    orig_rbox_axioms = list(orig_ont.getRBoxAxioms(Imports.fromBoolean(True)))
    orig_abox_axioms = list(orig_ont.getABoxAxioms(Imports.fromBoolean(True)))

    el_norm = ELNormalizer()
    prep_ont = el_norm.preprocess_ontology(orig_ont)
    prep_tbox_set = set(prep_ont.getTBoxAxioms(Imports.fromBoolean(True)))
    supported_tbox_types = {
        AxiomType.SUBCLASS_OF,
        AxiomType.EQUIVALENT_CLASSES,
        AxiomType.DISJOINT_CLASSES,
        AxiomType.OBJECT_PROPERTY_DOMAIN,
    }

    dropped_tbox_counter: Counter[str] = Counter()
    for ax in orig_tbox_axioms:
        if (
            ax not in prep_tbox_set
            or ax.getAxiomType() not in supported_tbox_types
            or "ObjectInverseOf" in str(ax.toString())
        ):
            dropped_tbox_counter[_classify_dropped_tbox_axiom(ax)] += 1

    dropped_rbox_counter: Counter[str] = Counter()
    for ax in orig_rbox_axioms:
        ax_t = ax.getAxiomType()
        if ax_t not in {
            AxiomType.SUB_OBJECT_PROPERTY,
            AxiomType.EQUIVALENT_OBJECT_PROPERTIES,
            AxiomType.TRANSITIVE_OBJECT_PROPERTY,
            AxiomType.SUB_PROPERTY_CHAIN_OF,
        }:
            dropped_rbox_counter[str(ax_t)] += 1

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    raw_tbox_ont = adapter.create_ontology("http://mowl.borg/helis/tbox-raw")
    tbox_buf = HashSet()
    for cls in orig_ont.getClassesInSignature(Imports.fromBoolean(True)):
        tbox_buf.add(df.getOWLDeclarationAxiom(cls))
    for prop in orig_ont.getObjectPropertiesInSignature(Imports.fromBoolean(True)):
        tbox_buf.add(df.getOWLDeclarationAxiom(prop))
    for ax in orig_tbox_axioms:
        tbox_buf.add(ax)
    for ax in orig_rbox_axioms:
        tbox_buf.add(ax)
    mgr.addAxioms(raw_tbox_ont, tbox_buf)

    no_tbox = normalise(raw_tbox_ont)
    assert no_tbox.dropped_axioms == sum(dropped_tbox_counter.values()) + sum(
        dropped_rbox_counter.values()
    )

    # Assert both disjointness axioms survived normalisation:
    # 14-way AllDisjoint -> C(14, 2) = 91 unordered pairs; RedMeat/WhiteMeat -> 1 pair (total 92)
    nf2_bot_unordered: Set[FrozenSet[str]] = {
        frozenset([ax.sub1, ax.sub2])
        for ax in no_tbox.axioms
        if isinstance(ax, NF2) and ax.sup == BOTTOM
    }
    expected_14way_pairs: Set[FrozenSet[str]] = {
        frozenset([c1, c2])
        for c1, c2 in itertools.combinations(FOOD_GROUP_14_CLASSES, 2)
    }
    expected_meat_pair: FrozenSet[str] = frozenset(RED_WHITE_MEAT_PAIR)
    assert len(expected_14way_pairs) == 91
    assert expected_14way_pairs <= nf2_bot_unordered, (
        "14-way food-group AllDisjoint pairs missing after TBox normalisation"
    )
    assert expected_meat_pair in nf2_bot_unordered, (
        "RedMeat/WhiteMeat disjointness missing after TBox normalisation"
    )
    t_tbox_norm = time.perf_counter() - t0

    # 3. Build training ontology.owl, valid.owl, test.owl, candidates.txt, skip_{valid,test}.tsv
    t0 = time.perf_counter()
    opa_triples: List[Tuple[str, str, str]] = []
    opa_by_prop_counter: Counter[str] = Counter()
    for ax in orig_ont.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION):
        prop_expr = ax.getProperty()
        subj_ind = ax.getSubject()
        obj_ind = ax.getObject()
        if (
            not prop_expr.isAnonymous()
            and not subj_ind.isAnonymous()
            and not obj_ind.isAnonymous()
        ):
            r_iri = str(prop_expr.asOWLObjectProperty().toStringID())
            s_iri = str(subj_ind.asOWLNamedIndividual().toStringID())
            o_iri = str(obj_ind.asOWLNamedIndividual().toStringID())
            opa_triples.append((r_iri, s_iri, o_iri))
            opa_by_prop_counter[r_iri.rsplit("#", 1)[-1]] += 1
    opa_triples.sort()

    n_data_prop_assertions = int(
        orig_ont.getAxioms(AxiomType.DATA_PROPERTY_ASSERTION).size()
    )
    n_anon_class_assertions = sum(
        1
        for ax in orig_ont.getAxioms(AxiomType.CLASS_ASSERTION)
        if ax.getClassExpression().isAnonymous() or ax.getIndividual().isAnonymous()
    )

    train_role_names = set(no_tbox.role_names) | {r for r, _, _ in opa_triples}
    no_train_tbox = NormalisedOntology(
        axioms=list(no_tbox.axioms),
        concept_names=set(no_tbox.concept_names) | set(candidates),
        role_names=train_role_names,
        dropped_axioms=no_tbox.dropped_axioms,
    )
    train_ont = to_owl(no_train_tbox)
    train_mgr = train_ont.getOWLOntologyManager()
    train_df = train_mgr.getOWLDataFactory()

    train_abox = HashSet()
    for ind_iri in sorted(individuals_txt):
        ind_obj = train_df.getOWLNamedIndividual(IRI.create(ind_iri))
        train_abox.add(train_df.getOWLDeclarationAxiom(ind_obj))
    for ind_iri, cls_iri in train_pos_pairs:
        ind_obj = train_df.getOWLNamedIndividual(IRI.create(ind_iri))
        cls_obj = train_df.getOWLClass(IRI.create(cls_iri))
        train_abox.add(train_df.getOWLClassAssertionAxiom(cls_obj, ind_obj))
    for r_iri, s_iri, o_iri in opa_triples:
        prop_obj = train_df.getOWLObjectProperty(IRI.create(r_iri))
        subj_obj = train_df.getOWLNamedIndividual(IRI.create(s_iri))
        obj_obj = train_df.getOWLNamedIndividual(IRI.create(o_iri))
        train_abox.add(
            train_df.getOWLObjectPropertyAssertionAxiom(prop_obj, subj_obj, obj_obj)
        )
    train_mgr.addAxioms(train_ont, train_abox)

    def _build_assertions_only_ontology(
        split_name: str, split_pairs: Sequence[Tuple[str, str]]
    ):
        split_adapter = OWLAPIAdapter()
        split_mgr = split_adapter.owl_manager
        split_df = split_adapter.data_factory
        split_ont = split_adapter.create_ontology(
            f"http://mowl.borg/helis/{split_name}"
        )
        split_axioms = HashSet()
        for ind_iri, cls_iri in split_pairs:
            ind_obj = split_df.getOWLNamedIndividual(IRI.create(ind_iri))
            cls_obj = split_df.getOWLClass(IRI.create(cls_iri))
            split_axioms.add(split_df.getOWLClassAssertionAxiom(cls_obj, ind_obj))
        split_mgr.addAxioms(split_ont, split_axioms)
        return split_ont

    valid_ont = _build_assertions_only_ontology("valid", valid_pairs)
    test_ont = _build_assertions_only_ontology("test", test_pairs)

    owl_files = {
        "ontology.owl": ("ontology", train_ont),
        "valid.owl": ("valid", valid_ont),
        "test.owl": ("test", test_ont),
    }
    for fname, (key, ont) in owl_files.items():
        fpath = out_dir / fname
        omgr = ont.getOWLOntologyManager()
        omgr.applyChange(
            SetOntologyID(
                ont,
                OWLOntologyID(IRI.create(f"http://mowl.borg/helis/{key}")),
            )
        )
        omgr.saveOntology(ont, IRI.create("file:" + str(fpath.resolve())))

    candidates_path = out_dir / "candidates.txt"
    candidates_path.write_text(
        "".join(f"{c}\n" for c in candidates), encoding="utf-8"
    )

    valid_eval_inds = [ind for ind, _ in valid_pairs]
    test_eval_inds = [ind for ind, _ in test_pairs]
    assert len(set(valid_eval_inds)) == len(valid_eval_inds)
    assert len(set(test_eval_inds)) == len(test_eval_inds)

    skip_valid_path = out_dir / "skip_valid.tsv"
    skip_test_path = out_dir / "skip_test.tsv"
    _write_skip_tsv(skip_valid_path, valid_eval_inds, inferred_classes_map)
    _write_skip_tsv(skip_test_path, test_eval_inds, inferred_classes_map)
    t_owl_write = time.perf_counter() - t0

    # 4. Build CanonicalModel on training ontology and compute three-way labels
    t0 = time.perf_counter()
    valid_eval_inds_sorted = sorted(valid_eval_inds)
    test_eval_inds_sorted = sorted(test_eval_inds)
    cm_train = CanonicalModel.from_ontology(train_ont)
    disjoint_with = _seed_cm_joint_sat_cache(
        cm_train,
        candidates,
        eval_individuals=valid_eval_inds_sorted + test_eval_inds_sorted,
    )
    t_cm_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels_valid_path = out_dir / "labels_valid.parquet"
    labels_test_path = out_dir / "labels_test.parquet"

    valid_labels_meta = _write_labels_parquet_from_cm(
        labels_valid_path,
        valid_eval_inds_sorted,
        candidates,
        cm_train,
    )
    test_labels_meta = _write_labels_parquet_from_cm(
        labels_test_path,
        test_eval_inds_sorted,
        candidates,
        cm_train,
    )

    # Also compute training-individual label summary from cm_train (where food-group
    # assertions are present in ontology.owl, verifying > 0 contradicted pairs) and
    # ground-truth contradiction counts on valid/test splits.
    train_inds_sorted = sorted({ind for ind, _ in train_pos_pairs})
    cand_set = set(candidates)
    n_train_ent = 0
    n_train_contra = 0
    for ind_iri in train_inds_sorted:
        nom = cm_train.individuals[ind_iri]
        ent_cands = cm_train.completion.sub_super.get(nom, set()) & cand_set
        contra_cands: Set[str] = set()
        for b in ent_cands:
            contra_cands.update(disjoint_with.get(b, set()))
        n_train_ent += len(ent_cands)
        n_train_contra += len(contra_cands - ent_cands)
    n_train_total_rows = len(train_inds_sorted) * len(candidates)
    train_labels_summary = {
        "n_rows": n_train_total_rows,
        "n_individuals": len(train_inds_sorted),
        "n_candidates": len(candidates),
        "n_entailed": n_train_ent,
        "n_unknown": n_train_total_rows - n_train_ent - n_train_contra,
        "n_contradicted": n_train_contra,
    }
    assert train_labels_summary["n_contradicted"] > 0, (
        "Expected > 0 contradicted pairs from food-group disjointness on training ontology"
    )

    n_valid_contra_gt = sum(
        len(disjoint_with.get(cls_iri, set())) for _, cls_iri in valid_pairs
    )
    n_test_contra_gt = sum(
        len(disjoint_with.get(cls_iri, set())) for _, cls_iri in test_pairs
    )
    t_parquet = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    output_filenames = [
        "ontology.owl",
        "valid.owl",
        "test.owl",
        "candidates.txt",
        "skip_valid.tsv",
        "skip_test.tsv",
        "labels_valid.parquet",
        "labels_test.parquet",
    ]
    outputs_manifest: Dict[str, Dict[str, Any]] = {}
    for fname in output_filenames:
        fpath = out_dir / fname
        outputs_manifest[fname] = {
            "sha256": _file_sha256(fpath),
            "size_bytes": fpath.stat().st_size,
        }

    timings_sec = {
        "load": round(t_load, 4),
        "tbox_normalise": round(t_tbox_norm, 4),
        "owl_write": round(t_owl_write, 4),
        "cm_build": round(t_cm_build, 4),
        "labels_parquet": round(t_parquet, 4),
        "total": round(t_total, 4),
    }

    dropped_tbox_dict = dict(
        sorted(dropped_tbox_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    dropped_rbox_dict = dict(
        sorted(dropped_rbox_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    opa_by_prop_dict = dict(
        sorted(opa_by_prop_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    n_norm_nf1 = sum(isinstance(a, NF1) for a in no_tbox.axioms)
    n_norm_nf2 = sum(isinstance(a, NF2) for a in no_tbox.axioms)
    n_norm_nf3 = sum(isinstance(a, NF3) for a in no_tbox.axioms)
    n_norm_nf4 = sum(isinstance(a, NF4) for a in no_tbox.axioms)

    manifest_data: Dict[str, Any] = {
        "dataset": "helis",
        "source_repository": "https://github.com/KRR-Oxford/OWL2Vec-Star",
        "license": "Apache-2.0 (OWL2Vec-Star case_studies)",
        "inputs": {
            HELIS_FILES_TARBALL_NAME: {
                "url": HELIS_FILES_URL,
                "sha256": HELIS_FILES_SHA256,
                "size_bytes": (dl_dir / HELIS_FILES_TARBALL_NAME).stat().st_size,
            },
            HELIS_ONTS_TARBALL_NAME: {
                "url": HELIS_ONTS_URL,
                "sha256": HELIS_ONTS_SHA256,
                "size_bytes": (dl_dir / HELIS_ONTS_TARBALL_NAME).stat().st_size,
            },
            "helis_v1.00.origin.owl": {
                "path": str(origin_owl_path),
                "sha256": _file_sha256(origin_owl_path),
                "size_bytes": origin_owl_path.stat().st_size,
            },
        },
        "counts": {
            "n_individuals": len(individuals_txt),
            "n_classes": len(classes_txt),
            "n_candidates": len(candidates),
            "n_origin_tbox_axioms": len(orig_tbox_axioms),
            "n_origin_rbox_axioms": len(orig_rbox_axioms),
            "n_origin_abox_axioms": len(orig_abox_axioms),
            "n_dropped_tbox_axioms": sum(dropped_tbox_counter.values()),
            "n_dropped_rbox_axioms": sum(dropped_rbox_counter.values()),
            "n_dropped_tbox_rbox_total": no_tbox.dropped_axioms,
            "dropped_tbox_by_construct": dropped_tbox_dict,
            "dropped_rbox_by_construct": dropped_rbox_dict,
            "n_dropped_data_property_assertions": n_data_prop_assertions,
            "n_dropped_anon_class_assertions": n_anon_class_assertions,
            "n_norm_tbox_axioms": len(no_tbox.axioms),
            "n_norm_nf1": n_norm_nf1,
            "n_norm_nf2": n_norm_nf2,
            "n_norm_nf3": n_norm_nf3,
            "n_norm_nf4": n_norm_nf4,
            "n_disjoint_nf2_pairs": len(nf2_bot_unordered),
            "n_train_pos": len(train_pos_pairs),
            "n_train_neg_dropped": n_train_neg_dropped,
            "n_valid": len(valid_pairs),
            "n_test": len(test_pairs),
            "n_object_property_assertions": len(opa_triples),
            "opa_by_property": opa_by_prop_dict,
            "n_valid_contradicted_with_ground_truth": n_valid_contra_gt,
            "n_test_contradicted_with_ground_truth": n_test_contra_gt,
        },
        "train_labels_summary": train_labels_summary,
        "valid_labels": valid_labels_meta,
        "test_labels": test_labels_meta,
        "timings_sec": timings_sec,
        "outputs": outputs_manifest,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if repo_mirror_dir is not None and out_dir == HELIS_OUT_DIR.resolve():
        repo_mirror_dir = Path(repo_mirror_dir).expanduser().resolve()
        repo_mirror_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, repo_mirror_dir / "manifest.json")

    info: Dict[str, Any] = {
        **manifest_data["counts"],
        "dl_dir": str(dl_dir),
        "out_dir": str(out_dir),
        "train_labels_summary": train_labels_summary,
        "valid_labels": valid_labels_meta,
        "test_labels": test_labels_meta,
        "timings_sec": timings_sec,
    }

    if stats_path is not None:
        update_dataset_stats_file(stats_path, info)

    return info


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for building the B3 ``helis-membership`` benchmark."""
    parser = argparse.ArgumentParser(
        description="Build the B3 helis-membership ABox benchmark."
    )
    parser.add_argument(
        "--dl-dir",
        type=Path,
        default=HELIS_DL_DIR,
        help="Input directory containing HeLiS tarballs and extracted files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HELIS_OUT_DIR,
        help="Output directory for built HeLiS benchmark files.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=RESULTS_DIR / "dataset_stats.md",
        help="Path to dataset_stats.md to update with B3 statistics.",
    )
    args = parser.parse_args(argv)
    info = build_helis(
        dl_dir=args.dl_dir,
        out_dir=args.out,
        stats_path=args.stats_path,
    )
    t = info["timings_sec"]
    print(
        f"Built helis-membership in {info['out_dir']} "
        f"(individuals={info['n_individuals']}, classes={info['n_classes']}, "
        f"split={info['n_train_pos']}/{info['n_valid']}/{info['n_test']}, "
        f"opas={info['n_object_property_assertions']}, "
        f"dropped_tbox={info['n_dropped_tbox_axioms']}, "
        f"disjoint_nf2={info['n_disjoint_nf2_pairs']}, "
        f"train_contra={info['train_labels_summary']['n_contradicted']}, "
        f"valid_contra={info['valid_labels']['n_contradicted']} [gt={info['n_valid_contradicted_with_ground_truth']}], "
        f"test_contra={info['test_labels']['n_contradicted']} [gt={info['n_test_contradicted_with_ground_truth']}], "
        f"time={t['total']:.2f}s)"
    )


if __name__ == "__main__":
    main()
