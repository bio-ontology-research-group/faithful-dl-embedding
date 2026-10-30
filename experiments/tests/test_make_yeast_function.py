"""Tests for the B1 yeast-function ABox benchmark builder (WP-B Task 1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mowl

mowl.init_jvm("8g")

import numpy as np
import pandas as pd
import pytest

from experiments.data.make_yeast_function import (
    GO_ROOT_TERMS,
    HAS_FUNCTION_IRI,
    INTERACTS_WITH_IRI,
    STATS_SECTION_HEADER,
    aux_iri_for_go,
    build_yeast_function,
    filter_entailed_holdouts,
    go_iri_for_aux,
    render_yeast_function_stats_section,
    update_dataset_stats_file,
)

SLIM_ONTOLOGY_PATH = Path(
    "~/.cache/mowl/datasets/ppi_yeast_slim/ontology.owl"
).expanduser()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p, g = line.split("\t")
            pairs.append((p, g))
    return pairs


@pytest.fixture(scope="module")
def slim_yeast_build(tmp_path_factory: pytest.TempPathFactory):
    """Build yeast-function once on the slim yeast ontology into a ~/tmp directory."""
    if not SLIM_ONTOLOGY_PATH.exists():
        pytest.skip(f"Slim yeast ontology not found at {SLIM_ONTOLOGY_PATH}")
    out_dir = tmp_path_factory.mktemp("yeast_function_slim")
    assert not str(out_dir.resolve()).startswith("/tmp/"), (
        f"tmp_path resolved under /tmp: {out_dir}"
    )
    stats_md = out_dir / "dataset_stats.md"
    info = build_yeast_function(
        input_path=SLIM_ONTOLOGY_PATH,
        out_dir=out_dir,
        seed=0,
        stats_path=stats_md,
    )
    return out_dir, info, stats_md


def test_split_sizes_add_up(slim_yeast_build):
    """Verify split sizes, TSV/OWL assertion counts, single-leaf invariant, and manifest sha256s."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AxiomType
    from org.semanticweb.owlapi.model.parameters import Imports
    import java.io

    out_dir, info, stats_md = slim_yeast_build

    expected_files = [
        "ontology.owl",
        "valid.owl",
        "test.owl",
        "train_annotations.tsv",
        "valid_annotations.tsv",
        "test_annotations.tsv",
        "labels_valid.parquet",
        "labels_test.parquet",
        "candidates.txt",
        "manifest.json",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing output file {fname}"
    assert stats_md.exists()

    train_pairs = _read_tsv_pairs(out_dir / "train_annotations.tsv")
    valid_pairs = _read_tsv_pairs(out_dir / "valid_annotations.tsv")
    test_pairs = _read_tsv_pairs(out_dir / "test_annotations.tsv")

    assert len(train_pairs) == info["n_train_annotations"]
    assert len(valid_pairs) == info["n_valid_annotations"]
    assert len(test_pairs) == info["n_test_annotations"]

    # Asserted annotations = leaf annotations + ancestors dropped
    assert (
        info["n_leaf_annotations"] + info["n_ancestor_annotations_dropped"]
        == info["n_raw_annotations"]
    )
    # Leaf annotations = train + valid + test + entailed holdouts dropped
    assert (
        len(train_pairs)
        + len(valid_pairs)
        + len(test_pairs)
        + info["n_entailed_holdouts_dropped"]
        == info["n_leaf_annotations"]
    )

    train_set = set(train_pairs)
    valid_set = set(valid_pairs)
    test_set = set(test_pairs)
    assert len(train_set) == len(train_pairs)
    assert len(valid_set) == len(valid_pairs)
    assert len(test_set) == len(test_pairs)
    assert train_set.isdisjoint(valid_set)
    assert train_set.isdisjoint(test_set)
    assert valid_set.isdisjoint(test_set)

    # Every protein with only 1 total leaf across splits must keep it in training
    leaves_by_prot: dict[str, int] = {}
    for p, _ in train_pairs + valid_pairs + test_pairs:
        leaves_by_prot[p] = leaves_by_prot.get(p, 0) + 1
    train_by_prot: dict[str, int] = {}
    for p, _ in train_pairs:
        train_by_prot[p] = train_by_prot.get(p, 0) + 1
    for p, total_c in leaves_by_prot.items():
        if total_c == 1:
            assert train_by_prot.get(p, 0) == 1, f"Single-leaf protein {p} not in train"

    # Candidates = Aux_G with >= 1 training annotation
    candidates = [
        line.strip()
        for line in (out_dir / "candidates.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_candidates = sorted({aux_iri_for_go(g) for _, g in train_pairs})
    assert candidates == expected_candidates
    assert len(candidates) == info["n_candidates"]

    # Check OWL files match TSV splits and signature rules
    mgr = OWLManager.createOWLOntologyManager()
    ont_train = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "ontology.owl"))
    )
    ont_valid = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "valid.owl"))
    )
    ont_test = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "test.owl"))
    )

    # Proteins must be NamedIndividuals, never OWLClasses, in ontology.owl
    train_class_iris = {
        str(c.toStringID()) for c in ont_train.getClassesInSignature()
    }
    assert not any(c.startswith("http://4932.") for c in train_class_iris)
    train_ind_iris = {
        str(ind.toStringID()) for ind in ont_train.getIndividualsInSignature()
    }
    assert len(train_ind_iris) == info["n_proteins"]
    assert all(p.startswith("http://4932.") for p in train_ind_iris)

    def _extract_class_assertions(ont) -> set[tuple[str, str]]:
        res = set()
        for ax in ont.getAxioms(AxiomType.CLASS_ASSERTION):
            ind_iri = str(ax.getIndividual().asOWLNamedIndividual().toStringID())
            cls_iri = str(ax.getClassExpression().asOWLClass().toStringID())
            res.add((ind_iri, go_iri_for_aux(cls_iri)))
        return res

    assert _extract_class_assertions(ont_train) == train_set
    assert _extract_class_assertions(ont_valid) == valid_set
    assert _extract_class_assertions(ont_test) == test_set

    # valid.owl and test.owl must have class assertions only (no TBox/RBox/OPA axioms)
    assert ont_valid.getTBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_valid.getRBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_valid.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size() == 0
    assert ont_test.getTBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_test.getRBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_test.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size() == 0

    # Training PPI object property assertions in ontology.owl
    assert (
        ont_train.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size()
        == info["n_interactions"]
    )

    # Manifest sha256 verification
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["sha256"] == _sha256(SLIM_ONTOLOGY_PATH)
    for fname, meta in manifest["outputs"].items():
        fpath = out_dir / fname
        assert fpath.exists()
        assert meta["sha256"] == _sha256(fpath)
        assert meta["size_bytes"] == fpath.stat().st_size

    # Held-out positives outside candidates verification in manifest.json and dataset_stats.md
    cand_go_set = {go_iri_for_aux(c) for c in candidates}
    exp_val_out = sum(1 for _, g in valid_pairs if g not in cand_go_set)
    exp_test_out = sum(1 for _, g in test_pairs if g not in cand_go_set)
    exp_val_frac = round(exp_val_out / len(valid_pairs), 6)
    exp_test_frac = round(exp_test_out / len(test_pairs), 6)

    for container in (manifest, manifest["counts"], info):
        assert "holdout_positives_outside_candidates" in container
        hp = container["holdout_positives_outside_candidates"]
        assert hp["valid"] == {
            "count": exp_val_out,
            "total": len(valid_pairs),
            "in_candidates": len(valid_pairs) - exp_val_out,
            "fraction": exp_val_frac,
        }
        assert hp["test"] == {
            "count": exp_test_out,
            "total": len(test_pairs),
            "in_candidates": len(test_pairs) - exp_test_out,
            "fraction": exp_test_frac,
        }

    stats_text = stats_md.read_text(encoding="utf-8")
    assert "Held-Out Positives Outside Candidates" in stats_text
    assert f"{exp_val_out:,} / {len(valid_pairs):,}" in stats_text
    assert f"{exp_test_out:,} / {len(test_pairs):,}" in stats_text
    assert f"{exp_val_frac:.6f}" in stats_text
    assert f"{exp_test_frac:.6f}" in stats_text


def test_split_drops_entailed_holdouts(slim_yeast_build):
    """Verify held-out annotations entailed by training annotations are dropped and counted."""
    out_dir, info, _ = slim_yeast_build

    train_pairs = _read_tsv_pairs(out_dir / "train_annotations.tsv")
    valid_pairs = _read_tsv_pairs(out_dir / "valid_annotations.tsv")
    test_pairs = _read_tsv_pairs(out_dir / "test_annotations.tsv")

    # On the built slim dataset, re-running filter_entailed_holdouts with the GO closure
    # must drop 0 additional pairs (all entailed holdouts already dropped).
    go_closure = info["go_superclasses"]
    valid_after, test_after, n_dropped = filter_entailed_holdouts(
        train_pairs, valid_pairs, test_pairs, go_closure
    )
    assert n_dropped == 0
    assert valid_after == valid_pairs
    assert test_after == test_pairs

    # Directly verify that when a held-out pair is an ancestor of a training annotation
    # of the same protein, filter_entailed_holdouts removes it and counts it.
    fake_closure = {
        "http://purl.obolibrary.org/obo/GO_0001": {
            "http://purl.obolibrary.org/obo/GO_0001",
            "http://purl.obolibrary.org/obo/GO_0002",
            "http://purl.obolibrary.org/obo/GO_0003",
        },
        "http://purl.obolibrary.org/obo/GO_0002": {
            "http://purl.obolibrary.org/obo/GO_0002",
            "http://purl.obolibrary.org/obo/GO_0003",
        },
        "http://purl.obolibrary.org/obo/GO_0004": {
            "http://purl.obolibrary.org/obo/GO_0004",
        },
    }
    tr = [("http://4932.P1", "http://purl.obolibrary.org/obo/GO_0001")]
    va = [
        ("http://4932.P1", "http://purl.obolibrary.org/obo/GO_0002"),  # entailed -> drop
        ("http://4932.P1", "http://purl.obolibrary.org/obo/GO_0004"),  # unknown -> keep
    ]
    te = [
        ("http://4932.P1", "http://purl.obolibrary.org/obo/GO_0003"),  # entailed -> drop
    ]
    va_clean, te_clean, dropped_count = filter_entailed_holdouts(
        tr, va, te, fake_closure
    )
    assert dropped_count == 2
    assert va_clean == [("http://4932.P1", "http://purl.obolibrary.org/obo/GO_0004")]
    assert te_clean == []


def test_every_aux_pair_present(slim_yeast_build):
    """Verify both Aux_G TBox axioms exist for every annotated GO class and 3 root disjointness axioms exist."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AxiomType, ClassExpressionType
    import java.io

    out_dir, _, _ = slim_yeast_build
    train_pairs = _read_tsv_pairs(out_dir / "train_annotations.tsv")
    valid_pairs = _read_tsv_pairs(out_dir / "valid_annotations.tsv")
    test_pairs = _read_tsv_pairs(out_dir / "test_annotations.tsv")

    all_go = {g for _, g in train_pairs + valid_pairs + test_pairs}
    assert len(all_go) > 0

    mgr = OWLManager.createOWLOntologyManager()
    ont = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "ontology.owl"))
    )

    aux_to_some: set[tuple[str, str, str]] = set()
    some_to_aux: set[tuple[str, str, str]] = set()
    disjoint_pairs: set[frozenset[str]] = set()

    for ax in ont.getAxioms(AxiomType.SUBCLASS_OF):
        sub = ax.getSubClass()
        sup = ax.getSuperClass()
        sub_type = sub.getClassExpressionType()
        sup_type = sup.getClassExpressionType()

        if (
            sub_type == ClassExpressionType.OWL_CLASS
            and sup_type == ClassExpressionType.OBJECT_SOME_VALUES_FROM
        ):
            s_iri = str(sub.asOWLClass().toStringID())
            prop_iri = str(sup.getProperty().toStringID())
            filler = sup.getFiller()
            if filler.getClassExpressionType() == ClassExpressionType.OWL_CLASS:
                f_iri = str(filler.asOWLClass().toStringID())
                aux_to_some.add((s_iri, prop_iri, f_iri))
        elif (
            sub_type == ClassExpressionType.OBJECT_SOME_VALUES_FROM
            and sup_type == ClassExpressionType.OWL_CLASS
        ):
            prop_iri = str(sub.getProperty().toStringID())
            filler = sub.getFiller()
            t_iri = str(sup.asOWLClass().toStringID())
            if filler.getClassExpressionType() == ClassExpressionType.OWL_CLASS:
                f_iri = str(filler.asOWLClass().toStringID())
                some_to_aux.add((prop_iri, f_iri, t_iri))
        elif (
            sub_type == ClassExpressionType.OBJECT_INTERSECTION_OF
            and sup.isOWLNothing()
        ):
            ops = [
                str(op.asOWLClass().toStringID())
                for op in sub.getOperands()
                if op.getClassExpressionType() == ClassExpressionType.OWL_CLASS
            ]
            if len(ops) == 2:
                disjoint_pairs.add(frozenset(ops))

    for g in all_go:
        aux = aux_iri_for_go(g)
        assert aux.startswith("http://mowl.borg/yeast-function/has_function/GO_")
        assert (aux, HAS_FUNCTION_IRI, g) in aux_to_some, (
            f"Missing Aux_G <= exists has_function.G for {g}"
        )
        assert (HAS_FUNCTION_IRI, g, aux) in some_to_aux, (
            f"Missing exists has_function.G <= Aux_G for {g}"
        )

    roots = list(GO_ROOT_TERMS.values())
    assert len(roots) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert frozenset([roots[i], roots[j]]) in disjoint_pairs, (
                f"Missing root disjointness between {roots[i]} and {roots[j]}"
            )


def test_labels_match_elk_sample(slim_yeast_build):
    """Verify parquet schemas, zero contradicted labels, and agreement with ELK getTypes on 200 samples."""
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import IRI
    from org.semanticweb.owlapi.reasoner import InferenceType
    import java.io

    out_dir, _, _ = slim_yeast_build

    candidates = [
        line.strip()
        for line in (out_dir / "candidates.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid_pairs = _read_tsv_pairs(out_dir / "valid_annotations.tsv")
    test_pairs = _read_tsv_pairs(out_dir / "test_annotations.tsv")
    valid_inds = sorted({p for p, _ in valid_pairs})
    test_inds = sorted({p for p, _ in test_pairs})

    df_valid = pd.read_parquet(out_dir / "labels_valid.parquet")
    df_test = pd.read_parquet(out_dir / "labels_test.parquet")

    assert list(df_valid.columns) == ["individual", "class", "label"]
    assert list(df_test.columns) == ["individual", "class", "label"]
    assert len(df_valid) == len(valid_inds) * len(candidates)
    assert len(df_test) == len(test_inds) * len(candidates)

    valid_labels = set(df_valid["label"].unique())
    test_labels = set(df_test["label"].unique())
    assert "contradicted" not in valid_labels
    assert "contradicted" not in test_labels
    assert valid_labels <= {"entailed", "unknown"}
    assert test_labels <= {"entailed", "unknown"}
    assert "entailed" in test_labels and "unknown" in test_labels

    # Classify and realize the built training ontology with ELK
    mgr = OWLManager.createOWLOntologyManager()
    df = mgr.getOWLDataFactory()
    ont_train = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "ontology.owl"))
    )
    rf = ElkReasonerFactory()
    reasoner = rf.createReasoner(ont_train)
    try:
        reasoner.precomputeInferences(
            InferenceType.CLASS_HIERARCHY, InferenceType.CLASS_ASSERTIONS
        )

        # Sample 200 rows from labels_test (100 entailed + 100 unknown)
        rng = np.random.RandomState(0)
        ent_rows = df_test[df_test["label"] == "entailed"]
        unk_rows = df_test[df_test["label"] == "unknown"]
        n_ent = min(100, len(ent_rows))
        n_unk = 200 - n_ent
        idx_ent = rng.choice(len(ent_rows), size=n_ent, replace=False)
        idx_unk = rng.choice(len(unk_rows), size=n_unk, replace=False)
        sample_df = pd.concat(
            [ent_rows.iloc[idx_ent], unk_rows.iloc[idx_unk]], ignore_index=True
        )
        assert len(sample_df) == 200

        types_cache: dict[str, set[str]] = {}
        for row in sample_df.itertuples(index=False):
            ind_iri = str(row.individual)
            cls_iri = str(row._1)
            lbl = str(row.label)
            if ind_iri not in types_cache:
                owl_ind = df.getOWLNamedIndividual(IRI.create(ind_iri))
                types_cache[ind_iri] = {
                    str(c.toStringID())
                    for c in reasoner.getTypes(owl_ind, False).getFlattened()
                }
            elk_entailed = cls_iri in types_cache[ind_iri]
            assert (lbl == "entailed") == elk_entailed, (
                f"Label mismatch for ({ind_iri}, {cls_iri}): label={lbl}, ELK={elk_entailed}"
            )
    finally:
        reasoner.dispose()


def test_update_dataset_stats_preserves_other_sections_byte_for_byte(
    tmp_path: Path, slim_yeast_build
):
    """Verify a file with sections A, B1, C has A and C preserved byte-for-byte after B1 update."""
    _, info, _ = slim_yeast_build

    section_a = (
        "# Experiment Dataset Statistics (`Task 6`)\n\n"
        "Introductory paragraph.\n\n"
        "## 1. Main Dataset Statistics Table\n\n"
        "| Dataset | Task |\n"
        "|---|---|\n"
        "| `pizza` | `subsumption` |\n\n"
        "## 3. Protocol Notes & Recorded Rulings\n\n"
        "1. First note.\n\n"
    )
    old_b1 = (
        f"{STATS_SECTION_HEADER}\n\n"
        "Old placeholder B1 content that must be replaced.\n"
        "- Old bullet.\n\n"
    )
    section_c = (
        "## 5. ABox Benchmark B3: `helis-membership` Statistics\n\n"
        "Generated by `experiments/data/make_helis.py`.\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        "| Individuals | 20,318 |\n\n"
        "- **Protocol Note**: Preserved byte-for-byte.\n"
    )

    stats_file = tmp_path / "dataset_stats.md"
    stats_file.write_bytes((section_a + old_b1 + section_c).encode("utf-8"))

    update_dataset_stats_file(stats_file, info)
    updated_bytes = stats_file.read_bytes()
    updated_text = updated_bytes.decode("utf-8")

    expected_b1 = render_yeast_function_stats_section(info)
    expected_full = (
        section_a + expected_b1.rstrip("\n") + "\n\n" + section_c
    ).encode("utf-8")

    assert updated_bytes == expected_full
    assert updated_text.startswith(section_a)
    assert updated_text.endswith(section_c)
    assert "Old placeholder B1 content" not in updated_text

    # Re-running update_dataset_stats_file must be idempotent
    update_dataset_stats_file(stats_file, info)
    assert stats_file.read_bytes() == expected_full

