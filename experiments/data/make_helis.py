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
  ``individuals.txt``, adds training class memberships from ``train.csv`` rows
  with label `1` (`14,222` positive ``ClassAssertion`` axioms; label-`0` sampled
  negatives and ``DataPropertyAssertion`` axioms are dropped), and materialises
  ``ObjectPropertyRange`` axioms with a named range class `R` by adding `R(b)` for
  every kept `r(a, b)` (`17,460` unique `R(b)` assertions, `16,458` net new
  ``ClassAssertion`` axioms, total `30,680` ``ClassAssertion`` axioms in
  ``ontology.owl``; sound because HeLiS has no role inclusions).
- Writes held-out ``valid.owl`` (`2,031` ``ClassAssertion`` axioms from
  ``valid.csv``) and ``test.owl`` (`4,065` ``ClassAssertion`` axioms from
  ``test.csv``), ``candidates.txt`` (`277` named classes from ``classes.txt``
  excluding ``owl:Thing``), and ``skip_{valid,test}.tsv`` (mapping each held-out
  individual to its comma-separated inferred classes from ``inferred_classes.txt``).
- Computes both strictly KB-based three-way labels (``label``: ``entailed``,
  ``unknown``, ``contradicted``) and gold-conditional three-way labels
  (``label_gold``: ``entailed_gold``, ``unknown``, ``contradicted_gold``) for every
  held-out `(individual, candidate)` pair via ``mowl.geometric.CanonicalModel``
  public methods (``instance_of``, ``entails``, and a precomputed 277 × 277
  pairwise table via ``joint_satisfiable``), writing
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
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
)

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
from experiments.data.stats_md import update_stats_section

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

_PAIRWISE_DISJOINT_MEMO: Dict[
    Tuple[FrozenSet[Any], Tuple[str, ...]], Dict[str, Set[str]]
] = {}


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


def precompute_pairwise_disjoint_table(
    cm_tbox: Any,
    candidates: Sequence[str],
) -> Dict[str, Set[str]]:
    """Precompute the 277 x 277 pairwise unsatisfiability table via public ``cm_tbox.joint_satisfiable``."""
    cand_tuple = tuple(candidates)
    key = (frozenset(cm_tbox.axioms), cand_tuple)
    cached = _PAIRWISE_DISJOINT_MEMO.get(key)
    if cached is not None:
        return {k: set(v) for k, v in cached.items()}

    disjoint_with: Dict[str, Set[str]] = {c: set() for c in cand_tuple}
    for i, c1 in enumerate(cand_tuple):
        for c2 in cand_tuple[i:]:
            if not cm_tbox.joint_satisfiable([c1, c2]):
                disjoint_with[c1].add(c2)
                disjoint_with[c2].add(c1)

    _PAIRWISE_DISJOINT_MEMO[key] = {k: set(v) for k, v in disjoint_with.items()}
    return disjoint_with


def precompute_tbox_superclasses(
    cm_tbox: Any,
    candidates: Sequence[str],
) -> Dict[str, Set[str]]:
    """Precompute candidate superclasses in the TBox closure via public ``cm_tbox.entails``."""
    return {
        c1: {c2 for c2 in candidates if cm_tbox.entails(c1, c2)}
        for c1 in candidates
    }


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
    split_pairs_sorted: Sequence[Tuple[str, str]],
    candidates: Sequence[str],
    cm_train: Any,
    tbox_superclasses: Mapping[str, Set[str]],
    disjoint_with: Mapping[str, Set[str]],
) -> Dict[str, int]:
    """Compute KB labels (``label``) and gold-conditional labels (``label_gold``) and write Parquet."""
    eval_individuals = [ind for ind, _ in split_pairs_sorted]
    n_ind = len(eval_individuals)
    n_cand = len(candidates)
    cand_to_idx = {c: j for j, c in enumerate(candidates)}

    ind_indices = np.repeat(np.arange(n_ind, dtype=np.int32), n_cand)
    cand_indices = np.tile(np.arange(n_cand, dtype=np.int32), n_ind)
    # 0 -> "unknown", 1 -> "entailed", 2 -> "contradicted"
    label_codes = np.zeros(n_ind * n_cand, dtype=np.int8)
    # 0 -> "unknown", 1 -> "entailed_gold", 2 -> "contradicted_gold"
    label_gold_codes = np.zeros(n_ind * n_cand, dtype=np.int8)

    for i, (ind_iri, gold_cls) in enumerate(split_pairs_sorted):
        offset = i * n_cand

        # Strictly KB-based label from cm_train and the 277x277 pairwise table
        ent_cands = [
            cls_iri
            for cls_iri in candidates
            if cm_train.instance_of(ind_iri, cls_iri)
        ]
        if ent_cands:
            ent_set = set(ent_cands)
            for cls_iri in ent_cands:
                label_codes[offset + cand_to_idx[cls_iri]] = 1
            contra_cands: Set[str] = set()
            for b in ent_cands:
                contra_cands.update(disjoint_with.get(b, set()))
            for cls_iri in contra_cands - ent_set:
                label_codes[offset + cand_to_idx[cls_iri]] = 2

        # Gold-conditional label from the TBox closure and 277x277 pairwise table
        gold_ent_set = tbox_superclasses.get(gold_cls, set())
        for cls_iri in gold_ent_set:
            label_gold_codes[offset + cand_to_idx[cls_iri]] = 1
        gold_contra_set = disjoint_with.get(gold_cls, set()) - gold_ent_set
        for cls_iri in gold_contra_set:
            label_gold_codes[offset + cand_to_idx[cls_iri]] = 2

    n_entailed = int(np.sum(label_codes == 1))
    n_contradicted = int(np.sum(label_codes == 2))
    n_unknown = int(len(label_codes) - n_entailed - n_contradicted)

    n_entailed_gold = int(np.sum(label_gold_codes == 1))
    n_contradicted_gold = int(np.sum(label_gold_codes == 2))
    n_unknown_gold = int(
        len(label_gold_codes) - n_entailed_gold - n_contradicted_gold
    )

    labels_vocab = pa.array(["unknown", "entailed", "contradicted"])
    labels_gold_vocab = pa.array(["unknown", "entailed_gold", "contradicted_gold"])
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
            pa.DictionaryArray.from_arrays(
                pa.array(label_gold_codes), labels_gold_vocab
            ).cast(pa.string()),
        ],
        names=["individual", "class", "label", "label_gold"],
    )
    pq.write_table(table, str(out_path))
    return {
        "n_rows": int(n_ind * n_cand),
        "n_individuals": n_ind,
        "n_candidates": n_cand,
        "n_entailed": n_entailed,
        "n_unknown": n_unknown,
        "n_contradicted": n_contradicted,
        "n_entailed_gold": n_entailed_gold,
        "n_unknown_gold": n_unknown_gold,
        "n_contradicted_gold": n_contradicted_gold,
    }


def render_helis_stats_section(info: Mapping[str, Any]) -> str:
    """Render the Markdown section for B3 ``helis-membership`` in ``dataset_stats.md``."""
    timings = info["timings_sec"]

    def _by_count_desc(d: Mapping[str, int]) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))

    dropped_tbox_str = ", ".join(
        f"`{k}`: {v}" for k, v in _by_count_desc(info["dropped_tbox_by_construct"])
    )
    dropped_rbox_str = ", ".join(
        f"`{k}`: {v}" for k, v in _by_count_desc(info["dropped_rbox_by_construct"])
    )
    opa_by_prop_str = ", ".join(
        f"`{k}`: {v:,}" for k, v in _by_count_desc(info["opa_by_property"])
    )
    range_unique_str = ", ".join(
        f"`{k}`: {v:,}"
        for k, v in _by_count_desc(info["range_assertions_unique_by_class"])
    )
    range_added_str = ", ".join(
        f"`{k}`: {v:,}"
        for k, v in _by_count_desc(info["range_assertions_added_by_class"])
    )
    newly_ent_str = ", ".join(
        f"`{k}`: {v:,}"
        for k, v in _by_count_desc(info["newly_entailed_heldout_by_class"])
    )
    tr_lbl = info["train_labels_summary"]
    va_lbl = info["valid_labels"]
    te_lbl = info["test_labels"]

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
        f"| Range → ABox Materialisation (`{info['n_object_property_ranges_named']}` named `ObjectPropertyRange`s; `0` role inclusions) | {info['n_range_assertions_raw']:,} raw / {info['n_range_assertions_unique']:,} unique ({range_unique_str}) / {info['n_range_assertions_added']:,} net new added ({range_added_str}; total `ClassAssertion`s in `ontology.owl`: {info['n_total_train_class_assertions']:,}) |",
        f"| Previously-Unknown Held-Out Pairs Entailed by Range Materialisation (`valid + test`) | {info['n_heldout_newly_entailed_by_ranges']:,} (`valid`: +{info['n_valid_newly_entailed_by_ranges']:,} [{info['n_valid_entailed_before_ranges']:,} → {va_lbl['n_entailed']:,}], `test`: +{info['n_test_newly_entailed_by_ranges']:,} [{info['n_test_entailed_before_ranges']:,} → {te_lbl['n_entailed']:,}]; by class: {newly_ent_str}; gold rows newly entailed: {info['n_heldout_gold_newly_entailed_by_ranges']:,} [`valid`: {info['n_valid_gold_newly_entailed_by_ranges']:,}, `test`: {info['n_test_gold_newly_entailed_by_ranges']:,}]) |",
        f"| Training Label Pairs (`train_individual × candidate`, KB vs Gold) | {tr_lbl['n_rows']:,} — KB (`label`): `entailed`: {tr_lbl['n_entailed']:,}, `unknown`: {tr_lbl['n_unknown']:,}, `contradicted`: {tr_lbl['n_contradicted']:,}; Gold (`label_gold`): `entailed_gold`: {tr_lbl['n_entailed_gold']:,}, `unknown`: {tr_lbl['n_unknown_gold']:,}, `contradicted_gold`: {tr_lbl['n_contradicted_gold']:,} |",
        f"| Validation Label Pairs (`labels_valid.parquet`, KB `label` vs Gold `label_gold`) | {va_lbl['n_rows']:,} — KB (`label`): `entailed`: {va_lbl['n_entailed']:,}, `unknown`: {va_lbl['n_unknown']:,}, `contradicted`: {va_lbl['n_contradicted']:,}; Gold (`label_gold`): `entailed_gold`: {va_lbl['n_entailed_gold']:,}, `unknown`: {va_lbl['n_unknown_gold']:,}, `contradicted_gold`: {va_lbl['n_contradicted_gold']:,} |",
        f"| Test Label Pairs (`labels_test.parquet`, KB `label` vs Gold `label_gold`) | {te_lbl['n_rows']:,} — KB (`label`): `entailed`: {te_lbl['n_entailed']:,}, `unknown`: {te_lbl['n_unknown']:,}, `contradicted`: {te_lbl['n_contradicted']:,}; Gold (`label_gold`): `entailed_gold`: {te_lbl['n_entailed_gold']:,}, `unknown`: {te_lbl['n_unknown_gold']:,}, `contradicted_gold`: {te_lbl['n_contradicted_gold']:,} |",
        f"| Build Time (`total`: `load / tbox_norm / owl_write / cm_build / parquet`) | {timings['total']:.2f} s ({timings['load']:.2f} / {timings['tbox_normalise']:.2f} / {timings['owl_write']:.2f} / {timings['cm_build']:.2f} / {timings['labels_parquet']:.2f} s) |",
        "",
        f"- **Protocol Note (Range Materialisation Soundness & Impact)**: All `{info['n_object_property_ranges_named']}` `ObjectPropertyRange` axioms in `helis_v1.00.origin.owl` have a named range class, and HeLiS has no role inclusions (`0` `SubObjectPropertyOf` and `0` `SubPropertyChainOf` axioms). Materialising `R(b)` for every kept `r(a, b)` with named range `R` is therefore sound in EL++ (Baader et al., 2005/2008). Across `{info['n_object_property_assertions']:,}` kept object-property assertions, materialisation produces `{info['n_range_assertions_unique']:,}` distinct `R(b)` assertions (`{info['n_range_assertions_added']:,}` net new `ClassAssertion` axioms not already in `train.csv` positive memberships, bringing `ontology.owl` to `{info['n_total_train_class_assertions']:,}` `ClassAssertion` axioms). On the held-out splits (`valid + test`), range materialisation turns `{info['n_heldout_newly_entailed_by_ranges']:,}` previously-unknown `(individual, candidate)` pairs into `entailed` (`+{info['n_valid_newly_entailed_by_ranges']:,}` on `valid` and `+{info['n_test_newly_entailed_by_ranges']:,}` on `test`; {newly_ent_str}), of which `4,813` are in `skip_{{valid,test}}.tsv` (`Nutrient` / `Food`) and `{info['n_heldout_gold_newly_entailed_by_ranges']:,}` are held-out gold `RecipeFood` rows (`{info['n_valid_gold_newly_entailed_by_ranges']:,}` on `valid`, `{info['n_test_gold_newly_entailed_by_ranges']:,}` on `test`, bringing total KB-entailed held-out gold rows from `{info['n_heldout_gold_entailed_before_ranges']:,}` to `{info['n_heldout_gold_entailed_after_ranges']:,}`).",
        f"- **Protocol Note (KB Labels `label` vs Gold-Conditional Labels `label_gold`, Spec §3.2 Amendment)**: Every HeLiS individual has a single atomic class assertion in `helis_v1.00.origin.owl`, partitioned into `train.csv` (`{info['n_train_pos']:,}`), `valid.csv` (`{info['n_valid']:,}`), and `test.csv` (`{info['n_test']:,}`). Held-out individuals can receive KB-entailed types `Recipe ⊑ Food`, `RecipeFood`, or `Nutrient` via domain/range materialisation on `hasRecipeFood`, `hasFood`, and `hasNutrient`, all of which are compatible with every food group; consequently, the strictly KB-based `label` column has `contradicted = {va_lbl['n_contradicted']:,}` on `valid` (`entailed = {va_lbl['n_entailed']:,}`, `unknown = {va_lbl['n_unknown']:,}`) and `contradicted = {te_lbl['n_contradicted']:,}` on `test` (`entailed = {te_lbl['n_entailed']:,}`, `unknown = {te_lbl['n_unknown']:,}`), while on `train` `639` food-group individuals yield `{tr_lbl['n_contradicted']:,}` contradicted pairs. Per the Spec §3.2 amendment, `labels_valid.parquet` and `labels_test.parquet` also carry `label_gold` (`entailed_gold` = TBox superclasses of the individual's gold class including itself; `contradicted_gold` = candidates whose conjunction with the gold class is unsatisfiable in the TBox; `unknown` otherwise), giving `entailed_gold = {va_lbl['n_entailed_gold']:,}`, `unknown = {va_lbl['n_unknown_gold']:,}`, `contradicted_gold = {va_lbl['n_contradicted_gold']:,}` on `valid` (`92` food-group individuals) and `entailed_gold = {te_lbl['n_entailed_gold']:,}`, `unknown = {te_lbl['n_unknown_gold']:,}`, `contradicted_gold = {te_lbl['n_contradicted_gold']:,}` on `test` (`185` food-group individuals).",
        "- **Protocol Note (Constructs Dropped by Normalisation)**: Beyond data-property axioms, `ObjectUnionOf` domains, and `InverseObjectProperties(hasActivity, isActivityOf)`, three axioms with partial EL content are dropped whole by `ELNormalizer` without manual rewriting: (1) `EquivalentClasses(ComposedFood, ObjectIntersectionOf(Food, ObjectSomeValuesFrom(contains, ObjectUnionOf(BasicFood, Dressing))))` (dropped because of the nested `ObjectUnionOf`, so `ComposedFood ⊑ Food` is not in the normalised TBox; `1` training `Soups ⊑ ComposedFood` individual (`FOOD-701358`) instead receives `Food` via `ObjectPropertyRange(hasFood, Food)`), (2) `SubClassOf(Violation, ObjectIntersectionOf(..., ObjectExactCardinality(1, hasViolationRule, MonitoringRule), ObjectExactCardinality(1, hasViolationUser, User)))` (dropped because of `ExactCardinality`), and (3) `1` anonymous ABox assertion `ClassAssertion(ObjectSomeValuesFrom(contains, VegetalOils), FOOD-1514)` (dropped because `ELNormalizer` retains only atomic `ClassAssertion(OWLClass, OWLNamedIndividual)` axioms).",
        "",
    ]
    return "\n".join(lines)


def update_dataset_stats_file(stats_path: Path, info: Mapping[str, Any]) -> None:
    """Append or replace only the B3 ``helis-membership`` section in ``stats_path``."""
    update_stats_section(
        stats_path, STATS_SECTION_HEADER, render_helis_stats_section(info)
    )


def build_helis(
    dl_dir: Path = HELIS_DL_DIR,
    out_dir: Path = HELIS_OUT_DIR,
    stats_path: Path | None = RESULTS_DIR / "dataset_stats.md",
    repo_mirror_dir: Path | None = HELIS_REPO_DIR,
    max_train_individuals: int | None = None,
    max_valid_individuals: int | None = None,
    max_test_individuals: int | None = None,
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

    is_subset = (
        max_train_individuals is not None
        or max_valid_individuals is not None
        or max_test_individuals is not None
    )

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
            if (
                max_train_individuals is None
                or len(train_pos_pairs) < max_train_individuals
            ):
                train_pos_pairs.append((ind_iri, cls_iri))
        elif label_str == "0":
            if (
                max_train_individuals is None
                or n_train_neg_dropped < max_train_individuals
            ):
                n_train_neg_dropped += 1
        else:
            raise ValueError(f"Unexpected label {label_str!r} in train.csv line: {s!r}")

    valid_pairs_all: List[Tuple[str, str]] = [
        tuple(p.strip() for p in line.strip().split(","))  # type: ignore[misc]
        for line in (files_dir / "valid.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_pairs_all: List[Tuple[str, str]] = [
        tuple(p.strip() for p in line.strip().split(","))  # type: ignore[misc]
        for line in (files_dir / "test.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if is_subset:
        n_val_lim = (
            max_valid_individuals
            if max_valid_individuals is not None
            else max(50, (max_train_individuals or 300) // 6)
        )
        n_test_lim = (
            max_test_individuals
            if max_test_individuals is not None
            else max(100, (max_train_individuals or 300) // 3)
        )
        valid_pairs = valid_pairs_all[:n_val_lim]
        test_pairs = test_pairs_all[:n_test_lim]
    else:
        valid_pairs = valid_pairs_all
        test_pairs = test_pairs_all
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

    # Verify HeLiS has no role inclusions (so range materialisation is sound in EL++)
    assert orig_ont.getAxioms(AxiomType.SUB_OBJECT_PROPERTY).size() == 0
    assert orig_ont.getAxioms(AxiomType.SUB_PROPERTY_CHAIN_OF).size() == 0

    prop_ranges: Dict[str, List[str]] = {}
    for ax in orig_ont.getAxioms(AxiomType.OBJECT_PROPERTY_RANGE):
        prop_expr = ax.getProperty()
        rng_expr = ax.getRange()
        if not prop_expr.isAnonymous() and not rng_expr.isAnonymous():
            r_iri = str(prop_expr.asOWLObjectProperty().toStringID())
            c_iri = str(rng_expr.asOWLClass().toStringID())
            prop_ranges.setdefault(r_iri, []).append(c_iri)
    for r_iri in prop_ranges:
        prop_ranges[r_iri].sort()
    n_object_property_ranges_named = sum(len(v) for v in prop_ranges.values())

    prop_domains: Dict[str, List[str]] = {}
    for ax in orig_ont.getAxioms(AxiomType.OBJECT_PROPERTY_DOMAIN):
        prop_expr = ax.getProperty()
        dom_expr = ax.getDomain()
        if not prop_expr.isAnonymous() and not dom_expr.isAnonymous():
            r_iri = str(prop_expr.asOWLObjectProperty().toStringID())
            c_iri = str(dom_expr.asOWLClass().toStringID())
            prop_domains.setdefault(r_iri, []).append(c_iri)

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
    split_inds: Set[str] = (
        {i for i, _ in train_pos_pairs}
        | {i for i, _ in valid_pairs}
        | {i for i, _ in test_pairs}
    )

    opa_triples: List[Tuple[str, str, str]] = []
    opa_by_prop_counter: Counter[str] = Counter()
    opa_endpoints: Set[str] = set()
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
            if is_subset and s_iri not in split_inds and o_iri not in split_inds:
                continue
            opa_triples.append((r_iri, s_iri, o_iri))
            opa_by_prop_counter[r_iri.rsplit("#", 1)[-1]] += 1
            opa_endpoints.add(s_iri)
            opa_endpoints.add(o_iri)
    opa_triples.sort()

    if is_subset:
        kept_ind_set = split_inds | opa_endpoints
        individuals_txt = [i for i in individuals_txt if i in kept_ind_set]

    # Materialise ObjectPropertyRange axioms with a named range class:
    # for every kept r(a, b) with range R, add ClassAssertion R(b) to ontology.owl
    train_pos_set: Set[Tuple[str, str]] = set(train_pos_pairs)
    n_range_assertions_raw = 0
    unique_range_assertions: Set[Tuple[str, str]] = set()
    for r_iri, _s_iri, o_iri in opa_triples:
        for rng_cls_iri in prop_ranges.get(r_iri, ()):
            n_range_assertions_raw += 1
            unique_range_assertions.add((o_iri, rng_cls_iri))

    range_unique_counter: Counter[str] = Counter()
    range_added_counter: Counter[str] = Counter()
    for o_iri, rng_cls_iri in unique_range_assertions:
        short_cls = rng_cls_iri.rsplit("#", 1)[-1]
        range_unique_counter[short_cls] += 1
        if (o_iri, rng_cls_iri) not in train_pos_set:
            range_added_counter[short_cls] += 1

    n_range_assertions_unique = len(unique_range_assertions)
    n_range_assertions_added = sum(range_added_counter.values())
    n_total_train_class_assertions = len(train_pos_set | unique_range_assertions)

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
        for rng_cls_iri in prop_ranges.get(r_iri, ()):
            rng_cls_obj = train_df.getOWLClass(IRI.create(rng_cls_iri))
            train_abox.add(
                train_df.getOWLClassAssertionAxiom(rng_cls_obj, obj_obj)
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

    # 4. Build CanonicalModel on TBox and training ontology using public API only
    t0 = time.perf_counter()
    cm_tbox = CanonicalModel.from_normalised(no_train_tbox)
    disjoint_with = precompute_pairwise_disjoint_table(cm_tbox, candidates)
    tbox_superclasses = precompute_tbox_superclasses(cm_tbox, candidates)
    cm_train = CanonicalModel.from_ontology(train_ont)
    t_cm_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    valid_pairs_sorted = sorted(valid_pairs, key=lambda kv: kv[0])
    test_pairs_sorted = sorted(test_pairs, key=lambda kv: kv[0])
    train_pairs_sorted = sorted(train_pos_pairs, key=lambda kv: kv[0])

    labels_valid_path = out_dir / "labels_valid.parquet"
    labels_test_path = out_dir / "labels_test.parquet"

    valid_labels_meta = _write_labels_parquet_from_cm(
        labels_valid_path,
        valid_pairs_sorted,
        candidates,
        cm_train,
        tbox_superclasses,
        disjoint_with,
    )
    test_labels_meta = _write_labels_parquet_from_cm(
        labels_test_path,
        test_pairs_sorted,
        candidates,
        cm_train,
        tbox_superclasses,
        disjoint_with,
    )

    # Measure held-out pairs that were unknown before range materialisation and became entailed
    subj_domain_classes: Dict[str, Set[str]] = {}
    for r_iri, s_iri, _o_iri in opa_triples:
        for dom_cls_iri in prop_domains.get(r_iri, ()):
            subj_domain_classes.setdefault(s_iri, set()).add(dom_cls_iri)

    def _pre_range_entailed_for_holdout(ind_iri: str) -> Set[str]:
        ent: Set[str] = set()
        for dom_cls in subj_domain_classes.get(ind_iri, ()):
            ent.update(tbox_superclasses.get(dom_cls, set()))
        return ent

    newly_entailed_counter: Counter[str] = Counter()
    n_valid_ent_before = 0
    n_valid_gold_before = 0
    n_valid_gold_after = 0
    for ind_iri, gold_cls in valid_pairs_sorted:
        before_ent = _pre_range_entailed_for_holdout(ind_iri)
        after_ent = {c for c in candidates if cm_train.instance_of(ind_iri, c)}
        n_valid_ent_before += len(before_ent)
        if gold_cls in before_ent:
            n_valid_gold_before += 1
        if gold_cls in after_ent:
            n_valid_gold_after += 1
        for c in after_ent - before_ent:
            newly_entailed_counter[c.rsplit("#", 1)[-1]] += 1

    n_test_ent_before = 0
    n_test_gold_before = 0
    n_test_gold_after = 0
    for ind_iri, gold_cls in test_pairs_sorted:
        before_ent = _pre_range_entailed_for_holdout(ind_iri)
        after_ent = {c for c in candidates if cm_train.instance_of(ind_iri, c)}
        n_test_ent_before += len(before_ent)
        if gold_cls in before_ent:
            n_test_gold_before += 1
        if gold_cls in after_ent:
            n_test_gold_after += 1
        for c in after_ent - before_ent:
            newly_entailed_counter[c.rsplit("#", 1)[-1]] += 1

    n_valid_newly_entailed = valid_labels_meta["n_entailed"] - n_valid_ent_before
    n_test_newly_entailed = test_labels_meta["n_entailed"] - n_test_ent_before
    n_heldout_newly_entailed = n_valid_newly_entailed + n_test_newly_entailed
    n_valid_gold_newly_entailed = n_valid_gold_after - n_valid_gold_before
    n_test_gold_newly_entailed = n_test_gold_after - n_test_gold_before
    n_heldout_gold_newly_entailed = (
        n_valid_gold_newly_entailed + n_test_gold_newly_entailed
    )

    # Also compute training-individual label summary (KB and gold-conditional)
    n_train_ent = 0
    n_train_contra = 0
    n_train_ent_gold = 0
    n_train_contra_gold = 0
    for ind_iri, gold_cls in train_pairs_sorted:
        ent_cands = [c for c in candidates if cm_train.instance_of(ind_iri, c)]
        ent_set = set(ent_cands)
        contra_cands: Set[str] = set()
        for b in ent_cands:
            contra_cands.update(disjoint_with.get(b, set()))
        n_train_ent += len(ent_set)
        n_train_contra += len(contra_cands - ent_set)

        gold_ent_set = tbox_superclasses.get(gold_cls, set())
        gold_contra_set = disjoint_with.get(gold_cls, set()) - gold_ent_set
        n_train_ent_gold += len(gold_ent_set)
        n_train_contra_gold += len(gold_contra_set)

    n_train_total_rows = len(train_pairs_sorted) * len(candidates)
    train_labels_summary = {
        "n_rows": n_train_total_rows,
        "n_individuals": len(train_pairs_sorted),
        "n_candidates": len(candidates),
        "n_entailed": n_train_ent,
        "n_unknown": n_train_total_rows - n_train_ent - n_train_contra,
        "n_contradicted": n_train_contra,
        "n_entailed_gold": n_train_ent_gold,
        "n_unknown_gold": (
            n_train_total_rows - n_train_ent_gold - n_train_contra_gold
        ),
        "n_contradicted_gold": n_train_contra_gold,
    }
    assert train_labels_summary["n_contradicted"] > 0, (
        "Expected > 0 contradicted pairs from food-group disjointness on training ontology"
    )

    n_valid_contra_gt = valid_labels_meta["n_contradicted_gold"]
    n_test_contra_gt = test_labels_meta["n_contradicted_gold"]
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
    range_unique_dict = dict(
        sorted(range_unique_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    range_added_dict = dict(
        sorted(range_added_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    newly_entailed_dict = dict(
        sorted(newly_entailed_counter.items(), key=lambda kv: (-kv[1], kv[0]))
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
            "n_object_property_ranges_named": n_object_property_ranges_named,
            "n_range_assertions_raw": n_range_assertions_raw,
            "n_range_assertions_unique": n_range_assertions_unique,
            "n_range_assertions_added": n_range_assertions_added,
            "n_total_train_class_assertions": n_total_train_class_assertions,
            "range_assertions_unique_by_class": range_unique_dict,
            "range_assertions_added_by_class": range_added_dict,
            "n_valid_entailed_before_ranges": n_valid_ent_before,
            "n_test_entailed_before_ranges": n_test_ent_before,
            "n_valid_newly_entailed_by_ranges": n_valid_newly_entailed,
            "n_test_newly_entailed_by_ranges": n_test_newly_entailed,
            "n_heldout_newly_entailed_by_ranges": n_heldout_newly_entailed,
            "newly_entailed_heldout_by_class": newly_entailed_dict,
            "n_valid_gold_newly_entailed_by_ranges": n_valid_gold_newly_entailed,
            "n_test_gold_newly_entailed_by_ranges": n_test_gold_newly_entailed,
            "n_heldout_gold_newly_entailed_by_ranges": (
                n_heldout_gold_newly_entailed
            ),
            "n_heldout_gold_entailed_before_ranges": (
                n_valid_gold_before + n_test_gold_before
            ),
            "n_heldout_gold_entailed_after_ranges": (
                n_valid_gold_after + n_test_gold_after
            ),
            "n_valid_entailed_gold": valid_labels_meta["n_entailed_gold"],
            "n_valid_unknown_gold": valid_labels_meta["n_unknown_gold"],
            "n_valid_contradicted_gold": valid_labels_meta["n_contradicted_gold"],
            "n_test_entailed_gold": test_labels_meta["n_entailed_gold"],
            "n_test_unknown_gold": test_labels_meta["n_unknown_gold"],
            "n_test_contradicted_gold": test_labels_meta["n_contradicted_gold"],
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

    if (
        repo_mirror_dir is not None
        and out_dir == HELIS_OUT_DIR.resolve()
        and not is_subset
    ):
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
        f"range_added={info['n_range_assertions_added']}, "
        f"newly_ent={info['n_heldout_newly_entailed_by_ranges']}, "
        f"train_contra={info['train_labels_summary']['n_contradicted']}, "
        f"valid_kb_ent={info['valid_labels']['n_entailed']}, "
        f"valid_gold=({info['valid_labels']['n_entailed_gold']}/{info['valid_labels']['n_unknown_gold']}/{info['valid_labels']['n_contradicted_gold']}), "
        f"test_kb_ent={info['test_labels']['n_entailed']}, "
        f"test_gold=({info['test_labels']['n_entailed_gold']}/{info['test_labels']['n_unknown_gold']}/{info['test_labels']['n_contradicted_gold']}), "
        f"time={t['total']:.2f}s)"
    )


if __name__ == "__main__":
    main()
