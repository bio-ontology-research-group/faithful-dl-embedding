#!/usr/bin/env python3
"""Build the B1 yeast-function ABox benchmark from mOWL PPIYeastDataset (Spec §3.1).

Transforms ``ontology.owl`` of mOWL's ``PPIYeastDataset`` into an ABox function-prediction
benchmark constrained by the Gene Ontology (GO) EL++ TBox:
- Rewrites every protein ``http://4932.*`` from an ``OWLClass`` into an
  ``OWLNamedIndividual``.
- Rewrites each function annotation ``p ⊑ ∃has_function.G`` into an atomic class
  assertion ``Aux_G(p)`` paired with the two TBox axioms
  ``Aux_G ⊑ ∃has_function.G`` and ``∃has_function.G ⊑ Aux_G``, where
  ``Aux_G`` has IRI ``http://mowl.borg/yeast-function/has_function/GO_XXXXXXX``.
- Rewrites training protein-protein interaction axioms ``p ⊑ ∃interacts_with.q``
  from ``ontology.owl`` into ``ObjectPropertyAssertion(interacts_with, p, q)``
  (validation/test PPI splits are unused).
- Extracts the GO EL++ fragment via ``mowl.geometric.normalise`` and ``to_owl``
  (as in ``make_pizza.py``) and ensures the three pairwise root disjointness axioms
  between ``biological_process`` (``GO_0008150``), ``molecular_function``
  (``GO_0003674``), and ``cellular_component`` (``GO_0005575``) are present.
- Computes the GO atomic subsumption closure with ELK on the GO TBox, reduces
  each protein's annotations to the most-specific (leaf) GO terms, splits leaf
  ``(protein, G)`` pairs 80/10/10 using ``numpy.random.RandomState(seed)`` while
  keeping single-leaf proteins in training, and drops any held-out leaf entailed
  by the same protein's training annotations.
- Writes ``ontology.owl``, ``valid.owl``, ``test.owl``,
  ``{train,valid,test}_annotations.tsv``, ``candidates.txt``,
  ``labels_{valid,test}.parquet``, ``manifest.json``, and appends the B1 section
  to ``experiments/results/dataset_stats.md``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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
from experiments.data.stats_md import update_stats_section

YEAST_RAW_ONTOLOGY_PATH: Path = (
    RAW_DATA_ROOT / "yeast" / "ppi_yeast" / "ontology.owl"
)
YEAST_FUNCTION_DIR: Path = RAW_DATA_ROOT / "yeast-function"
YEAST_FUNCTION_REPO_DIR: Path = DATA_DIR / "yeast-function"

PROTEIN_PREFIX: str = "http://4932."
GO_PREFIX: str = "http://purl.obolibrary.org/obo/GO_"
EC_LABEL_PREFIX: str = "http://mowl/label/"
AUX_PREFIX: str = "http://mowl.borg/yeast-function/has_function/"
HAS_FUNCTION_IRI: str = "http://has_function"
INTERACTS_WITH_IRI: str = "http://interacts_with"
HAS_LABEL_IRI: str = "http://has_label"

# Programmatically verified against rdfs:label annotations in both slim and full GO ontologies
GO_ROOT_TERMS: Dict[str, str] = {
    "biological_process": "http://purl.obolibrary.org/obo/GO_0008150",
    "molecular_function": "http://purl.obolibrary.org/obo/GO_0003674",
    "cellular_component": "http://purl.obolibrary.org/obo/GO_0005575",
}

STATS_SECTION_HEADER: str = "## 4. ABox Benchmark B1: `yeast-function` Statistics"


def aux_iri_for_go(go_iri: str) -> str:
    """Return the auxiliary concept IRI ``Aux_G`` for a GO class IRI ``go_iri``."""
    local_id = go_iri.rsplit("/", 1)[-1]
    if not local_id.startswith("GO_"):
        raise ValueError(f"Expected GO class IRI ending with GO_*, got {go_iri!r}")
    return f"{AUX_PREFIX}{local_id}"


def go_iri_for_aux(aux_iri: str) -> str:
    """Invert :func:`aux_iri_for_go` from an ``Aux_G`` IRI back to its GO class IRI."""
    if not aux_iri.startswith(AUX_PREFIX):
        raise ValueError(f"Expected Aux_G IRI starting with {AUX_PREFIX!r}, got {aux_iri!r}")
    local_id = aux_iri[len(AUX_PREFIX) :]
    return f"http://purl.obolibrary.org/obo/{local_id}"


def _file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_entailed_holdouts(
    train_pairs: Sequence[Tuple[str, str]],
    valid_pairs: Sequence[Tuple[str, str]],
    test_pairs: Sequence[Tuple[str, str]],
    go_superclasses: Mapping[str, Set[str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]:
    """Remove held-out ``(protein, G)`` pairs entailed by ``train_pairs`` via ``go_superclasses``.

    Parameters
    ----------
    train_pairs : Sequence[Tuple[str, str]]
        Training ``(protein_iri, go_iri)`` annotations.
    valid_pairs : Sequence[Tuple[str, str]]
        Candidate validation ``(protein_iri, go_iri)`` annotations.
    test_pairs : Sequence[Tuple[str, str]]
        Candidate test ``(protein_iri, go_iri)`` annotations.
    go_superclasses : Mapping[str, Set[str]]
        Mapping from each GO class ``G'`` to its reflexive superclass closure
        ``{G : G' ⊑ G}`` in the GO EL++ TBox.

    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]
        Cleaned ``(valid_pairs, test_pairs, n_dropped)``.
    """
    entailed_by_protein: Dict[str, Set[str]] = defaultdict(set)
    for p, g_train in train_pairs:
        entailed_by_protein[p].update(go_superclasses.get(g_train, {g_train}))

    clean_valid: List[Tuple[str, str]] = []
    clean_test: List[Tuple[str, str]] = []
    dropped = 0

    for p, g in valid_pairs:
        if g in entailed_by_protein.get(p, set()):
            dropped += 1
        else:
            clean_valid.append((p, g))

    for p, g in test_pairs:
        if g in entailed_by_protein.get(p, set()):
            dropped += 1
        else:
            clean_test.append((p, g))

    return clean_valid, clean_test, dropped


def _write_tsv(path: Path, pairs: Iterable[Tuple[str, str]]) -> None:
    """Write ``(protein, go_iri)`` pairs as tab-separated lines."""
    content = "".join(f"{p}\t{g}\n" for p, g in pairs)
    path.write_text(content, encoding="utf-8")


def _write_labels_parquet(
    out_path: Path,
    eval_individuals: Sequence[str],
    candidates_aux: Sequence[str],
    candidates_go: Sequence[str],
    train_entailed_go_by_prot: Mapping[str, Set[str]],
) -> Dict[str, int]:
    """Build and write ``labels_{valid,test}.parquet`` for ``eval_individuals × candidates``."""
    n_ind = len(eval_individuals)
    n_cand = len(candidates_aux)
    cand_idx_map = {g: idx for idx, g in enumerate(candidates_go)}

    ind_indices = np.repeat(np.arange(n_ind, dtype=np.int32), n_cand)
    cand_indices = np.tile(np.arange(n_cand, dtype=np.int32), n_ind)
    # 0 -> "unknown", 1 -> "entailed" (contradicted is empty by construction)
    label_codes = np.zeros(n_ind * n_cand, dtype=np.int8)

    for i, p in enumerate(eval_individuals):
        offset = i * n_cand
        for g_ent in train_entailed_go_by_prot.get(p, set()):
            c_idx = cand_idx_map.get(g_ent)
            if c_idx is not None:
                label_codes[offset + c_idx] = 1

    n_entailed = int(label_codes.sum())
    n_unknown = int(len(label_codes) - n_entailed)

    labels_vocab = pa.array(["unknown", "entailed", "contradicted"])
    table = pa.Table.from_arrays(
        [
            pa.DictionaryArray.from_arrays(
                pa.array(ind_indices), pa.array(eval_individuals, type=pa.string())
            ).cast(pa.string()),
            pa.DictionaryArray.from_arrays(
                pa.array(cand_indices), pa.array(candidates_aux, type=pa.string())
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
        "n_contradicted": 0,
    }


def render_yeast_function_stats_section(info: Mapping[str, Any]) -> str:
    """Render the Markdown section for B1 ``yeast-function`` in ``dataset_stats.md``."""
    timings = info["timings_sec"]
    hp = info["holdout_positives_outside_candidates"]
    hp_val = hp["valid"]
    hp_test = hp["test"]
    lines = [
        STATS_SECTION_HEADER,
        "",
        "Generated by `experiments/data/make_yeast_function.py` from `PPIYeastDataset` (`ontology.owl`).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Proteins (`OWLNamedIndividual`, `http://4932.*`) | {info['n_proteins']:,} (annotated: {info['n_annotated_proteins']:,}; in PPI: {info['n_ppi_proteins']:,}) |",
        f"| GO Classes in Signature (`http://purl.obolibrary.org/obo/GO_*`) | {info['n_go_classes_total']:,} (annotated across splits: {info['n_go_classes_annotated']:,}) |",
        f"| Normalised GO Concept Names (incl. `⊤`, `⊥`, aux, `Aux_G`) | {info['n_train_concept_names']:,} (`GO` EL TBox dropped axioms: {info['n_go_dropped_axioms']:,}) |",
        f"| Raw Asserted Annotations (`p ⊑ ∃has_function.G`) | {info['n_raw_annotations']:,} |",
        f"| Ancestor Annotations Dropped (Closure Leaf Reduction) | {info['n_ancestor_annotations_dropped']:,} |",
        f"| Remaining Leaf Annotations | {info['n_leaf_annotations']:,} |",
        f"| Mean Leaves per Annotated Protein (All Proteins) | {info['mean_leaves_per_annotated_protein']:.2f} ({info['mean_leaves_per_protein']:.2f}) |",
        f"| Single-Leaf Proteins Kept in Training | {info['n_single_leaf_proteins']:,} |",
        f"| Annotations per Split (`train / valid / test`, seed `{info['seed']}`) | {info['n_train_annotations']:,} / {info['n_valid_annotations']:,} / {info['n_test_annotations']:,} |",
        f"| Entailed Holdouts Dropped (`valid + test`) | {info['n_entailed_holdouts_dropped']:,} (`0` by construction after per-protein leaf reduction) |",
        f"| Candidate Classes (`Aux_G` with `>= 1` training annotation) | {info['n_candidates']:,} |",
        f"| Held-Out Positives Outside Candidates (`valid` / `test`) | {hp_val['count']:,} / {hp_val['total']:,} ({100.0 * hp_val['fraction']:.2f}%, frac `{hp_val['fraction']:.6f}`) / {hp_test['count']:,} / {hp_test['total']:,} ({100.0 * hp_test['fraction']:.2f}%, frac `{hp_test['fraction']:.6f}`) |",
        f"| Training PPI Assertions (`interacts_with(p, q)`) | {info['n_interactions']:,} |",
        f"| Validation Label Pairs (`individual × candidate`) | {info['valid_labels']['n_rows']:,} (`entailed`: {info['valid_labels']['n_entailed']:,}, `unknown`: {info['valid_labels']['n_unknown']:,}, `contradicted`: {info['valid_labels']['n_contradicted']:,} — `contradicted = 0` by design) |",
        f"| Test Label Pairs (`individual × candidate`) | {info['test_labels']['n_rows']:,} (`entailed`: {info['test_labels']['n_entailed']:,}, `unknown`: {info['test_labels']['n_unknown']:,}, `contradicted`: {info['test_labels']['n_contradicted']:,} — `contradicted = 0` by design) |",
        f"| Build Time (`total`: `load / normalise / elk_closure / owl_write / parquet`) | {timings['total']:.2f} s ({timings['load']:.2f} / {timings['go_normalise']:.2f} / {timings['elk_closure']:.2f} / {timings['owl_write']:.2f} / {timings['labels_parquet']:.2f} s) |",
        "",
        "- **Protocol Note (`contradicted = 0` by design)**: Each annotation is encoded via `Aux_G(p)` with `Aux_G ⊑ ∃has_function.G` and `∃has_function.G ⊑ Aux_G`. Because existential restrictions `∃has_function.G₁ ⊓ ∃has_function.G₂` remain satisfiable even when `G₁ ⊓ G₂ ⊑ ⊥` in GO, `contradicted = 0` by design for all candidate pairs (`labels_valid.parquet` and `labels_test.parquet` retain the three-way schema with `0` `'contradicted'` entries).",
        f"- **Protocol Note (Candidate Coverage of Held-Out Positives)**: Candidates are defined by Spec §3.1 as `Aux_G` classes with `>= 1` training annotation (`{info['n_candidates']:,}` of `{info['n_go_classes_annotated']:,}` annotated GO classes). Consequently, `{hp_val['count']:,} / {hp_val['total']:,}` (`{100.0 * hp_val['fraction']:.2f}%`, fraction `{hp_val['fraction']:.6f}`) held-out validation positives and `{hp_test['count']:,} / {hp_test['total']:,}` (`{100.0 * hp_test['fraction']:.2f}%`, fraction `{hp_test['fraction']:.6f}`) held-out test positives have a class `Aux_G` outside `candidates.txt` (leaving `{hp_val['in_candidates']:,}` validation and `{hp_test['in_candidates']:,}` test positives inside `candidates.txt`). Both `{{valid,test}}_annotations.tsv` and `{{valid,test}}.owl` retain all held-out positives so the evaluator can report both candidate-filtered and unfiltered metrics.",
        "",
    ]
    return "\n".join(lines)


def update_dataset_stats_file(stats_path: Path, info: Mapping[str, Any]) -> None:
    """Append or replace only the B1 ``yeast-function`` section in ``stats_path``.

    Replaces from ``STATS_SECTION_HEADER`` up to (not including) the next
    ``## `` header or EOF, preserving all preceding and following sections
    byte-for-byte.
    """
    update_stats_section(
        stats_path, STATS_SECTION_HEADER, render_yeast_function_stats_section(info)
    )


def build_yeast_function(
    input_path: Path = YEAST_RAW_ONTOLOGY_PATH,
    out_dir: Path = YEAST_FUNCTION_DIR,
    seed: int = 0,
    stats_path: Path | None = RESULTS_DIR / "dataset_stats.md",
    repo_mirror_dir: Path | None = YEAST_FUNCTION_REPO_DIR,
) -> Dict[str, Any]:
    """Build the B1 ``yeast-function`` ABox benchmark files and return metadata."""
    t_start = time.perf_counter()
    mowl.init_jvm("8g")

    from java.util import HashSet
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.model import (
        AxiomType,
        ClassExpressionType,
        IRI,
        OWLOntologyID,
        SetOntologyID,
    )
    from org.semanticweb.owlapi.model.parameters import Imports
    from org.semanticweb.owlapi.reasoner import InferenceType
    from mowl.datasets import PathDataset
    from mowl.geometric import (
        BOTTOM,
        NF2,
        NF3,
        NF4,
        NormalisedOntology,
        normalise,
        to_owl,
    )
    from mowl.owlapi import OWLAPIAdapter

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input yeast ontology not found at {input_path}")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load input ontology and separate proteins, annotations, interactions, and GO TBox/RBox
    t0 = time.perf_counter()
    ds_raw = PathDataset(str(input_path))
    raw_ont = ds_raw.ontology

    all_sig_classes = sorted(
        str(c.toStringID()) for c in raw_ont.getClassesInSignature()
    )
    proteins = sorted(c for c in all_sig_classes if c.startswith(PROTEIN_PREFIX))
    go_classes_total = sorted(c for c in all_sig_classes if c.startswith(GO_PREFIX))

    raw_annotations_by_prot: Dict[str, Set[str]] = defaultdict(set)
    interactions_set: Set[Tuple[str, str]] = set()
    ppi_proteins: Set[str] = set()

    for ax in raw_ont.getAxioms(AxiomType.SUBCLASS_OF):
        sub = ax.getSubClass()
        sup = ax.getSuperClass()
        if (
            sub.getClassExpressionType() == ClassExpressionType.OWL_CLASS
            and sup.getClassExpressionType()
            == ClassExpressionType.OBJECT_SOME_VALUES_FROM
        ):
            s_iri = str(sub.asOWLClass().toStringID())
            if s_iri.startswith(PROTEIN_PREFIX):
                prop_iri = str(sup.getProperty().toStringID())
                filler = sup.getFiller()
                if filler.getClassExpressionType() == ClassExpressionType.OWL_CLASS:
                    f_iri = str(filler.asOWLClass().toStringID())
                    if prop_iri == HAS_FUNCTION_IRI and f_iri.startswith(GO_PREFIX):
                        raw_annotations_by_prot[s_iri].add(f_iri)
                    elif prop_iri == INTERACTS_WITH_IRI and f_iri.startswith(
                        PROTEIN_PREFIX
                    ):
                        interactions_set.add((s_iri, f_iri))
                        ppi_proteins.add(s_iri)
                        ppi_proteins.add(f_iri)

    interactions = sorted(interactions_set)
    n_raw_annotations = sum(len(gs) for gs in raw_annotations_by_prot.values())

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    go_raw_ont = adapter.create_ontology("http://mowl.borg/yeast-function/go-raw")
    go_axioms_buf = HashSet()
    for g_iri in go_classes_total:
        go_axioms_buf.add(df.getOWLDeclarationAxiom(adapter.create_class(g_iri)))
    for ax in raw_ont.getTBoxAxioms(Imports.fromBoolean(True)):
        s = str(ax.toString())
        if (
            PROTEIN_PREFIX not in s
            and EC_LABEL_PREFIX not in s
            and HAS_LABEL_IRI not in s
        ):
            go_axioms_buf.add(ax)
    for ax in raw_ont.getRBoxAxioms(Imports.fromBoolean(True)):
        go_axioms_buf.add(ax)
    mgr.addAxioms(go_raw_ont, go_axioms_buf)
    t_load = time.perf_counter() - t0

    # 2. Normalise GO TBox/RBox into EL++ fragment and add 3 root disjointness axioms
    t0 = time.perf_counter()
    no_go = normalise(go_raw_ont)
    roots = list(GO_ROOT_TERMS.values())
    existing_axioms = set(no_go.axioms)
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            r1, r2 = roots[i], roots[j]
            nf_a = NF2(r1, r2, BOTTOM)
            nf_b = NF2(r2, r1, BOTTOM)
            if nf_a not in existing_axioms and nf_b not in existing_axioms:
                no_go.axioms.append(nf_a)
                existing_axioms.add(nf_a)
            no_go.concept_names.add(r1)
            no_go.concept_names.add(r2)
    t_go_norm = time.perf_counter() - t0

    # 3. Compute GO atomic subsumption closure once with ELK on the normalised GO TBox
    t0 = time.perf_counter()
    owl_go_tbox = to_owl(no_go)
    rf = ElkReasonerFactory()
    reasoner = rf.createReasoner(owl_go_tbox)
    try:
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
        all_annotated_go = sorted(
            {g for gs in raw_annotations_by_prot.values() for g in gs}
        )
        go_superclasses: Dict[str, Set[str]] = {}
        go_strict_superclasses: Dict[str, Set[str]] = {}
        for g in all_annotated_go:
            owl_cls = adapter.create_class(g)
            sups = {
                str(s.toStringID())
                for s in reasoner.getSuperClasses(owl_cls, False).getFlattened()
                if str(s.toStringID()).startswith(GO_PREFIX)
            }
            eqs = {
                str(s.toStringID())
                for s in reasoner.getEquivalentClasses(owl_cls).getEntities()
                if str(s.toStringID()).startswith(GO_PREFIX)
            }
            refl = sups | eqs | {g}
            go_superclasses[g] = refl
            # Note on GO equivalent-class edge case: go_strict_superclasses[g] includes
            # any named GO class equivalent to g (eqs - {g}). If a protein were annotated
            # with two distinct named GO classes that are mutually equivalent in GO, each
            # would appear in the other's go_strict_superclasses and both would be dropped
            # during leaf reduction below. In the GO release bundled with PPIYeastDataset
            # (2021-10), no two distinct named GO_* classes are mutually equivalent
            # (equivalences are only between a named GO_* class and a complex expression),
            # so eqs == {g} for every annotated GO class and no annotations are lost.
            go_strict_superclasses[g] = refl - {g}
    finally:
        reasoner.dispose()
    t_elk_closure = time.perf_counter() - t0

    # 4. Reduce per protein to most-specific (leaf) annotations and split 80/10/10
    t0 = time.perf_counter()
    leaves_by_prot: Dict[str, List[str]] = {}
    for p in sorted(raw_annotations_by_prot.keys()):
        gs = raw_annotations_by_prot[p]
        ancestors_of_others: Set[str] = set()
        for g_prime in gs:
            ancestors_of_others.update(go_strict_superclasses.get(g_prime, set()))
        leaves_by_prot[p] = sorted(gs - ancestors_of_others)

    all_leaf_pairs: List[Tuple[str, str]] = sorted(
        (p, g) for p, gs in leaves_by_prot.items() for g in gs
    )
    n_leaf_annotations = len(all_leaf_pairs)
    n_ancestor_dropped = n_raw_annotations - n_leaf_annotations
    single_leaf_prots = {p for p, gs in leaves_by_prot.items() if len(gs) == 1}

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_leaf_annotations)
    n_val_raw = int(round(0.10 * n_leaf_annotations))
    n_test_raw = int(round(0.10 * n_leaf_annotations))
    n_train_raw = n_leaf_annotations - n_val_raw - n_test_raw

    train_initial = [all_leaf_pairs[i] for i in perm[:n_train_raw]]
    valid_initial = [
        all_leaf_pairs[i] for i in perm[n_train_raw : n_train_raw + n_val_raw]
    ]
    test_initial = [all_leaf_pairs[i] for i in perm[n_train_raw + n_val_raw :]]

    # Proteins with a single leaf keep it in training
    train_pairs_list = list(train_initial)
    valid_kept: List[Tuple[str, str]] = []
    test_kept: List[Tuple[str, str]] = []
    for p, g in valid_initial:
        if p in single_leaf_prots:
            train_pairs_list.append((p, g))
        else:
            valid_kept.append((p, g))
    for p, g in test_initial:
        if p in single_leaf_prots:
            train_pairs_list.append((p, g))
        else:
            test_kept.append((p, g))

    train_pairs = sorted(train_pairs_list)
    valid_kept = sorted(valid_kept)
    test_kept = sorted(test_kept)

    # Drop any held-out leaf entailed by training leaves of the same protein
    # (0 by construction after per-protein leaf reduction when no two leaves are equivalent)
    valid_pairs, test_pairs, n_entailed_holdouts_dropped = filter_entailed_holdouts(
        train_pairs, valid_kept, test_kept, go_superclasses
    )

    used_go_classes = sorted(
        {g for _, g in train_pairs}
        | {g for _, g in valid_pairs}
        | {g for _, g in test_pairs}
    )
    candidates_go = sorted({g for _, g in train_pairs})
    candidates_go_set = set(candidates_go)
    candidates_aux = [aux_iri_for_go(g) for g in candidates_go]

    # Per split (valid, test), count held-out positives whose class is NOT in candidates.txt.
    # The TSVs and {valid,test}.owl retain all held-out positives so the evaluator decides
    # whether to report filtered or unfiltered metrics.
    n_valid_total = len(valid_pairs)
    n_valid_outside = sum(1 for _, g in valid_pairs if g not in candidates_go_set)
    n_valid_in_cands = n_valid_total - n_valid_outside
    frac_valid_outside = (
        round(float(n_valid_outside) / float(n_valid_total), 6)
        if n_valid_total > 0
        else 0.0
    )

    n_test_total = len(test_pairs)
    n_test_outside = sum(1 for _, g in test_pairs if g not in candidates_go_set)
    n_test_in_cands = n_test_total - n_test_outside
    frac_test_outside = (
        round(float(n_test_outside) / float(n_test_total), 6)
        if n_test_total > 0
        else 0.0
    )

    holdout_positives_outside_candidates: Dict[str, Dict[str, int | float]] = {
        "valid": {
            "count": n_valid_outside,
            "total": n_valid_total,
            "in_candidates": n_valid_in_cands,
            "fraction": frac_valid_outside,
        },
        "test": {
            "count": n_test_outside,
            "total": n_test_total,
            "in_candidates": n_test_in_cands,
            "fraction": frac_test_outside,
        },
    }
    t_split = time.perf_counter() - t0

    # 5. Write TSVs, candidates.txt, and OWL files (ontology.owl, valid.owl, test.owl)
    t0 = time.perf_counter()
    train_tsv_path = out_dir / "train_annotations.tsv"
    valid_tsv_path = out_dir / "valid_annotations.tsv"
    test_tsv_path = out_dir / "test_annotations.tsv"
    candidates_path = out_dir / "candidates.txt"

    _write_tsv(train_tsv_path, train_pairs)
    _write_tsv(valid_tsv_path, valid_pairs)
    _write_tsv(test_tsv_path, test_pairs)
    candidates_path.write_text(
        "".join(f"{c}\n" for c in candidates_aux), encoding="utf-8"
    )

    train_concept_names = set(no_go.concept_names)
    train_role_names = set(no_go.role_names) | {HAS_FUNCTION_IRI, INTERACTS_WITH_IRI}
    train_tbox_axioms = list(no_go.axioms)
    for g in used_go_classes:
        aux = aux_iri_for_go(g)
        train_concept_names.add(aux)
        train_concept_names.add(g)
        train_tbox_axioms.append(NF3(aux, HAS_FUNCTION_IRI, g))
        train_tbox_axioms.append(NF4(HAS_FUNCTION_IRI, g, aux))

    no_train_tbox = NormalisedOntology(
        axioms=train_tbox_axioms,
        concept_names=train_concept_names,
        role_names=train_role_names,
        dropped_axioms=no_go.dropped_axioms,
    )
    train_ont = to_owl(no_train_tbox)
    train_mgr = train_ont.getOWLOntologyManager()
    train_df = train_mgr.getOWLDataFactory()

    train_abox = HashSet()
    for p in proteins:
        ind_p = train_df.getOWLNamedIndividual(IRI.create(p))
        train_abox.add(train_df.getOWLDeclarationAxiom(ind_p))
    for p, g in train_pairs:
        ind_p = train_df.getOWLNamedIndividual(IRI.create(p))
        cls_aux = train_df.getOWLClass(IRI.create(aux_iri_for_go(g)))
        train_abox.add(train_df.getOWLClassAssertionAxiom(cls_aux, ind_p))
    iw_prop = train_df.getOWLObjectProperty(IRI.create(INTERACTS_WITH_IRI))
    for p, q in interactions:
        ind_p = train_df.getOWLNamedIndividual(IRI.create(p))
        ind_q = train_df.getOWLNamedIndividual(IRI.create(q))
        train_abox.add(
            train_df.getOWLObjectPropertyAssertionAxiom(iw_prop, ind_p, ind_q)
        )
    train_mgr.addAxioms(train_ont, train_abox)

    def _build_assertions_only_ontology(
        split_name: str, split_pairs: Sequence[Tuple[str, str]]
    ):
        split_adapter = OWLAPIAdapter()
        split_mgr = split_adapter.owl_manager
        split_df = split_adapter.data_factory
        split_ont = split_adapter.create_ontology(
            f"http://mowl.borg/yeast-function/{split_name}"
        )
        split_axioms = HashSet()
        for p, g in split_pairs:
            ind_p = split_df.getOWLNamedIndividual(IRI.create(p))
            cls_aux = split_df.getOWLClass(IRI.create(aux_iri_for_go(g)))
            split_axioms.add(split_df.getOWLClassAssertionAxiom(cls_aux, ind_p))
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
                OWLOntologyID(IRI.create(f"http://mowl.borg/yeast-function/{key}")),
            )
        )
        omgr.saveOntology(ont, IRI.create("file:" + str(fpath.resolve())))
    t_owl_write = time.perf_counter() - t0

    # 6. Compute three-way labels via GO closure and write labels_{valid,test}.parquet
    t0 = time.perf_counter()
    train_entailed_go_by_prot: Dict[str, Set[str]] = defaultdict(set)
    for p, g_train in train_pairs:
        train_entailed_go_by_prot[p].update(go_superclasses.get(g_train, {g_train}))

    valid_eval_inds = sorted({p for p, _ in valid_pairs})
    test_eval_inds = sorted({p for p, _ in test_pairs})

    labels_valid_path = out_dir / "labels_valid.parquet"
    labels_test_path = out_dir / "labels_test.parquet"

    valid_labels_meta = _write_labels_parquet(
        labels_valid_path,
        valid_eval_inds,
        candidates_aux,
        candidates_go,
        train_entailed_go_by_prot,
    )
    test_labels_meta = _write_labels_parquet(
        labels_test_path,
        test_eval_inds,
        candidates_aux,
        candidates_go,
        train_entailed_go_by_prot,
    )
    t_parquet = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    n_annotated_proteins = len(leaves_by_prot)
    mean_leaves_annotated = (
        float(n_leaf_annotations) / float(n_annotated_proteins)
        if n_annotated_proteins > 0
        else 0.0
    )
    mean_leaves_all = (
        float(n_leaf_annotations) / float(len(proteins))
        if len(proteins) > 0
        else 0.0
    )

    output_filenames = [
        "ontology.owl",
        "valid.owl",
        "test.owl",
        "train_annotations.tsv",
        "valid_annotations.tsv",
        "test_annotations.tsv",
        "labels_valid.parquet",
        "labels_test.parquet",
        "candidates.txt",
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
        "go_normalise": round(t_go_norm, 4),
        "elk_closure": round(t_elk_closure, 4),
        "split_and_filter": round(t_split, 4),
        "owl_write": round(t_owl_write, 4),
        "labels_parquet": round(t_parquet, 4),
        "total": round(t_total, 4),
    }

    manifest_data: Dict[str, Any] = {
        "dataset": "yeast-function",
        "seed": seed,
        "input": {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "size_bytes": input_path.stat().st_size,
        },
        "counts": {
            "n_proteins": len(proteins),
            "n_annotated_proteins": n_annotated_proteins,
            "n_ppi_proteins": len(ppi_proteins),
            "n_go_classes_total": len(go_classes_total),
            "n_go_classes_annotated": len(used_go_classes),
            "n_train_concept_names": len(train_concept_names),
            "n_go_dropped_axioms": no_go.dropped_axioms,
            "n_raw_annotations": n_raw_annotations,
            "n_ancestor_annotations_dropped": n_ancestor_dropped,
            "n_leaf_annotations": n_leaf_annotations,
            "mean_leaves_per_annotated_protein": round(mean_leaves_annotated, 4),
            "mean_leaves_per_protein": round(mean_leaves_all, 4),
            "n_single_leaf_proteins": len(single_leaf_prots),
            "n_train_annotations": len(train_pairs),
            "n_valid_annotations": len(valid_pairs),
            "n_test_annotations": len(test_pairs),
            "n_entailed_holdouts_dropped": n_entailed_holdouts_dropped,
            "n_candidates": len(candidates_aux),
            "holdout_positives_outside_candidates": holdout_positives_outside_candidates,
            "n_interactions": len(interactions),
            "contradicted": 0,
        },
        "holdout_positives_outside_candidates": holdout_positives_outside_candidates,
        "valid_labels": valid_labels_meta,
        "test_labels": test_labels_meta,
        "timings_sec": timings_sec,
        "outputs": outputs_manifest,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if repo_mirror_dir is not None and out_dir == YEAST_FUNCTION_DIR.resolve():
        repo_mirror_dir = Path(repo_mirror_dir).expanduser().resolve()
        repo_mirror_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, repo_mirror_dir / "manifest.json")

    info: Dict[str, Any] = {
        **manifest_data["counts"],
        "seed": seed,
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "valid_labels": valid_labels_meta,
        "test_labels": test_labels_meta,
        "timings_sec": timings_sec,
        "go_superclasses": go_superclasses,
    }

    if stats_path is not None:
        update_dataset_stats_file(stats_path, info)

    return info


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for building the B1 ``yeast-function`` benchmark."""
    parser = argparse.ArgumentParser(
        description="Build the B1 yeast-function ABox benchmark."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=YEAST_RAW_ONTOLOGY_PATH,
        help="Path to input PPIYeastDataset ontology.owl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=YEAST_FUNCTION_DIR,
        help="Output directory for built yeast-function files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for 80/10/10 leaf annotation split (default: 0).",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=RESULTS_DIR / "dataset_stats.md",
        help="Path to dataset_stats.md to update with B1 statistics.",
    )
    args = parser.parse_args(argv)
    info = build_yeast_function(
        input_path=args.input,
        out_dir=args.out,
        seed=args.seed,
        stats_path=args.stats_path,
    )
    t = info["timings_sec"]
    print(
        f"Built yeast-function in {info['out_dir']} "
        f"(proteins={info['n_proteins']}, go_classes={info['n_go_classes_total']}, "
        f"annotated_go={info['n_go_classes_annotated']}, "
        f"leaves={info['n_leaf_annotations']} [mean={info['mean_leaves_per_annotated_protein']:.2f}], "
        f"split={info['n_train_annotations']}/{info['n_valid_annotations']}/{info['n_test_annotations']}, "
        f"candidates={info['n_candidates']}, "
        f"entailed_holdouts_dropped={info['n_entailed_holdouts_dropped']}, "
        f"contradicted=0, "
        f"time={t['total']:.2f}s [load={t['load']:.2f}s, norm={t['go_normalise']:.2f}s, "
        f"elk={t['elk_closure']:.2f}s, owl={t['owl_write']:.2f}s, parquet={t['labels_parquet']:.2f}s])"
    )


if __name__ == "__main__":
    main()
