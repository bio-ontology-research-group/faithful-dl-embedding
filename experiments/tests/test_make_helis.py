"""Tests for the B3 helis-membership ABox benchmark builder (WP-B Task 2)."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import mowl

mowl.init_jvm("8g")

import numpy as np
import pandas as pd
import pytest

from experiments.data.fetch_helis import (
    HELIS_DL_DIR,
    HELIS_FILES_SHA256,
    HELIS_FILES_TARBALL_NAME,
    HELIS_ONTS_SHA256,
    HELIS_ONTS_TARBALL_NAME,
    fetch_helis,
)
from experiments.data.make_helis import (
    EXPECTED_N_CLASSES,
    EXPECTED_N_INDIVIDUALS,
    EXPECTED_N_TEST,
    EXPECTED_N_TRAIN_NEG,
    EXPECTED_N_TRAIN_POS,
    EXPECTED_N_VALID,
    FOOD_GROUP_14_CLASSES,
    HELIS_OUT_DIR,
    OWL_THING_IRI,
    OWL_THING_SHORT,
    RED_WHITE_MEAT_PAIR,
    _seed_cm_joint_sat_cache,
    build_helis,
    candidate_set_for_individual,
    parse_inferred_classes_file,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def helis_build(tmp_path_factory: pytest.TempPathFactory):
    """Ensure HeLiS benchmark is built and return ``(out_dir, manifest)``."""
    if not HELIS_DL_DIR.exists():
        pytest.skip(f"HeLiS input directory not found at {HELIS_DL_DIR}")

    out_dir = HELIS_OUT_DIR
    required_files = [
        "ontology.owl",
        "valid.owl",
        "test.owl",
        "candidates.txt",
        "skip_valid.tsv",
        "skip_test.tsv",
        "labels_valid.parquet",
        "labels_test.parquet",
        "manifest.json",
    ]
    if not all((out_dir / f).exists() for f in required_files):
        tmp_stats = tmp_path_factory.mktemp("helis_stats") / "dataset_stats.md"
        build_helis(
            dl_dir=HELIS_DL_DIR,
            out_dir=out_dir,
            stats_path=tmp_stats,
            repo_mirror_dir=None,
        )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return out_dir, manifest


def test_fetch_helis_and_split_counts(helis_build):
    """Verify tarball SHA-256s, split counts (14,222 / 2,031 / 4,065), OWL files, and manifest digests."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AxiomType
    from org.semanticweb.owlapi.model.parameters import Imports
    import java.io

    fetch_meta = fetch_helis(dl_dir=HELIS_DL_DIR)
    assert (
        fetch_meta["tarballs"][HELIS_FILES_TARBALL_NAME]["sha256"]
        == HELIS_FILES_SHA256
    )
    assert (
        fetch_meta["tarballs"][HELIS_ONTS_TARBALL_NAME]["sha256"] == HELIS_ONTS_SHA256
    )
    assert fetch_meta["tarballs"][HELIS_FILES_TARBALL_NAME]["downloaded"] is False
    assert fetch_meta["tarballs"][HELIS_ONTS_TARBALL_NAME]["downloaded"] is False

    out_dir, manifest = helis_build
    counts = manifest["counts"]

    assert counts["n_individuals"] == EXPECTED_N_INDIVIDUALS == 20_318
    assert counts["n_classes"] == EXPECTED_N_CLASSES == 277
    assert counts["n_candidates"] == EXPECTED_N_CLASSES == 277
    assert counts["n_train_pos"] == EXPECTED_N_TRAIN_POS == 14_222
    assert counts["n_train_neg_dropped"] == EXPECTED_N_TRAIN_NEG == 14_222
    assert counts["n_valid"] == EXPECTED_N_VALID == 2_031
    assert counts["n_test"] == EXPECTED_N_TEST == 4_065
    assert counts["n_object_property_assertions"] == 83_210

    # Verify OWL file contents with OWLAPI
    mgr1 = OWLManager.createOWLOntologyManager()
    ont_train = mgr1.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "ontology.owl"))
    )
    mgr2 = OWLManager.createOWLOntologyManager()
    ont_valid = mgr2.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "valid.owl"))
    )
    mgr3 = OWLManager.createOWLOntologyManager()
    ont_test = mgr3.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "test.owl"))
    )

    assert ont_train.getIndividualsInSignature().size() == 20_318
    assert ont_train.getAxioms(AxiomType.CLASS_ASSERTION).size() == 14_222
    assert ont_train.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size() == 83_210
    assert ont_train.getAxioms(AxiomType.DATA_PROPERTY_ASSERTION).size() == 0

    assert ont_valid.getAxioms(AxiomType.CLASS_ASSERTION).size() == 2_031
    assert ont_valid.getTBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_valid.getRBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_valid.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size() == 0

    assert ont_test.getAxioms(AxiomType.CLASS_ASSERTION).size() == 4_065
    assert ont_test.getTBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_test.getRBoxAxioms(Imports.fromBoolean(True)).size() == 0
    assert ont_test.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size() == 0

    # Verify manifest SHA-256s match files on disk
    for fname, meta in manifest["outputs"].items():
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing output file {fname}"
        assert meta["sha256"] == _sha256(fpath)
        assert meta["size_bytes"] == fpath.stat().st_size


def test_helis_candidates_match_reference_protocol(helis_build):
    """For 20 sampled test rows, candidate set equals classes.txt minus owl:Thing minus inferred_classes[a]."""
    out_dir, _ = helis_build
    files_dir = HELIS_DL_DIR / "files"

    classes_txt = [
        line.strip()
        for line in (files_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inferred_map = parse_inferred_classes_file(files_dir / "inferred_classes.txt")
    test_rows = [
        tuple(p.strip() for p in line.strip().split(","))
        for line in (files_dir / "test.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    candidates = [
        line.strip()
        for line in (out_dir / "candidates.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert OWL_THING_SHORT not in candidates
    assert OWL_THING_IRI not in candidates
    assert set(candidates) == set(classes_txt) - {OWL_THING_SHORT, OWL_THING_IRI}

    skip_test_map: dict[str, list[str]] = {}
    for line in (out_dir / "skip_test.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ind_iri, rest = line.split("\t", 1)
        skip_test_map[ind_iri] = [
            c.strip() for c in rest.split(",") if c.strip()
        ]
    assert len(skip_test_map) == len(test_rows) == 4_065

    rng = np.random.RandomState(0)
    sampled_indices = rng.choice(len(test_rows), size=20, replace=False)
    for idx in sampled_indices:
        ind_iri, gt_cls = test_rows[idx]
        # Reference OWL2Vec* protocol: classes.txt minus owl:Thing minus inferred_classes[a]
        ref_candidates = (
            set(classes_txt)
            - {OWL_THING_SHORT, OWL_THING_IRI}
            - set(inferred_map[ind_iri])
        )
        # Built candidates.txt minus skip_test.tsv entry for ind_iri
        built_candidates = set(candidates) - set(skip_test_map[ind_iri])
        assert built_candidates == ref_candidates
        assert (
            candidate_set_for_individual(ind_iri, classes_txt, inferred_map)
            == ref_candidates
        )
        # Ground-truth class is never in the skip list and always remains a candidate
        assert gt_cls in built_candidates
        assert gt_cls not in skip_test_map[ind_iri]


def test_disjointness_survives_normalisation(helis_build):
    """Verify both the 14-way food-group AllDisjoint and RedMeat/WhiteMeat disjointness survive in ontology.owl."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AxiomType, ClassExpressionType
    import java.io

    out_dir, _ = helis_build
    mgr = OWLManager.createOWLOntologyManager()
    ont = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "ontology.owl"))
    )

    disjoint_pairs: set[frozenset[str]] = set()
    for ax in ont.getAxioms(AxiomType.SUBCLASS_OF):
        sub = ax.getSubClass()
        sup = ax.getSuperClass()
        if (
            sub.getClassExpressionType()
            == ClassExpressionType.OBJECT_INTERSECTION_OF
            and sup.isOWLNothing()
        ):
            ops = [
                str(op.asOWLClass().toStringID())
                for op in sub.getOperands()
                if op.getClassExpressionType() == ClassExpressionType.OWL_CLASS
            ]
            if len(ops) == 2:
                disjoint_pairs.add(frozenset(ops))

    expected_14way = {
        frozenset([c1, c2])
        for c1, c2 in itertools.combinations(FOOD_GROUP_14_CLASSES, 2)
    }
    expected_meat = frozenset(RED_WHITE_MEAT_PAIR)

    assert len(expected_14way) == 91
    assert expected_14way <= disjoint_pairs, (
        f"Missing 14-way AllDisjoint pairs: {expected_14way - disjoint_pairs}"
    )
    assert expected_meat in disjoint_pairs, (
        f"Missing RedMeat/WhiteMeat disjointness: {expected_meat}"
    )
    assert len(disjoint_pairs) == 92


def test_labels_match_elk_sample(helis_build):
    """Verify labels against ELK on 200 sampled (individual, class) pairs (entailed, unknown, and contradicted)."""
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AddAxiom, IRI, RemoveAxiom
    from org.semanticweb.owlapi.reasoner import InferenceType
    from mowl.geometric import CanonicalModel
    import java.io

    out_dir, _ = helis_build

    candidates = [
        line.strip()
        for line in (out_dir / "candidates.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df_valid = pd.read_parquet(out_dir / "labels_valid.parquet")
    df_test = pd.read_parquet(out_dir / "labels_test.parquet")

    assert list(df_valid.columns) == ["individual", "class", "label"]
    assert list(df_test.columns) == ["individual", "class", "label"]
    assert len(df_valid) == 2_031 * 277
    assert len(df_test) == 4_065 * 277

    mgr = OWLManager.createOWLOntologyManager()
    df = mgr.getOWLDataFactory()
    ont_train = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(out_dir / "ontology.owl"))
    )

    cm = CanonicalModel.from_ontology(ont_train)
    disjoint_with = _seed_cm_joint_sat_cache(cm, candidates, eval_individuals=[])

    # Build a 200-pair sample spanning:
    # - 140 rows from labels_test.parquet (70 entailed + 70 unknown)
    # - 60 training-ontology pairs (20 entailed + 20 unknown + 20 contradicted from food-group individuals)
    rng = np.random.RandomState(0)
    test_ent = df_test[df_test["label"] == "entailed"]
    test_unk = df_test[df_test["label"] == "unknown"]
    assert len(test_ent) >= 70
    idx_te_ent = rng.choice(len(test_ent), size=70, replace=False)
    idx_te_unk = rng.choice(len(test_unk), size=70, replace=False)

    sample_triples: list[tuple[str, str, str]] = []
    for row in test_ent.iloc[idx_te_ent].itertuples(index=False):
        sample_triples.append((str(row.individual), str(row._1), str(row.label)))
    for row in test_unk.iloc[idx_te_unk].itertuples(index=False):
        sample_triples.append((str(row.individual), str(row._1), str(row.label)))

    # Collect food-group training individuals that have contradicted candidates in ontology.owl
    cand_set = set(candidates)
    train_ent_pool: list[tuple[str, str, str]] = []
    train_unk_pool: list[tuple[str, str, str]] = []
    train_contra_pool: list[tuple[str, str, str]] = []

    for ind_iri in sorted(cm.individuals.keys()):
        nom = cm.individuals[ind_iri]
        ent_cands = cm.completion.sub_super.get(nom, set()) & cand_set
        contra_cands: set[str] = set()
        for b in ent_cands:
            contra_cands.update(disjoint_with.get(b, set()))
        contra_cands -= ent_cands
        if contra_cands and len(train_contra_pool) < 200:
            for c in sorted(ent_cands)[:2]:
                train_ent_pool.append((ind_iri, c, "entailed"))
            for c in sorted(contra_cands)[:3]:
                train_contra_pool.append((ind_iri, c, "contradicted"))
            unk_cands = sorted(cand_set - ent_cands - contra_cands)
            for c in unk_cands[:2]:
                train_unk_pool.append((ind_iri, c, "unknown"))

    assert len(train_contra_pool) >= 20
    for pool, k in [
        (train_ent_pool, 20),
        (train_unk_pool, 20),
        (train_contra_pool, 20),
    ]:
        chosen = rng.choice(len(pool), size=k, replace=False)
        for i in chosen:
            sample_triples.append(pool[i])

    assert len(sample_triples) == 200

    # Also verify un-seeded cm.assertion_contradicted and cm.instance_of match a 15-pair sample across all 3 label classes
    cm_unseeded = CanonicalModel.from_ontology(ont_train)
    for ind_iri, cls_iri, lbl in (
        sample_triples[140:145] + sample_triples[160:165] + sample_triples[180:185]
    ):
        assert cm_unseeded.instance_of(ind_iri, cls_iri) == (lbl == "entailed")
        assert cm_unseeded.assertion_contradicted(ind_iri, cls_iri) == (
            lbl == "contradicted"
        )

    rf = ElkReasonerFactory()
    reasoner = rf.createReasoner(ont_train)
    try:
        reasoner.precomputeInferences(
            InferenceType.CLASS_HIERARCHY, InferenceType.CLASS_ASSERTIONS
        )
        types_cache: dict[str, set[str]] = {}
        for ind_iri, cls_iri, lbl in sample_triples:
            if ind_iri not in types_cache:
                owl_ind = df.getOWLNamedIndividual(IRI.create(ind_iri))
                types_cache[ind_iri] = {
                    str(c.toStringID())
                    for c in reasoner.getTypes(owl_ind, False).getFlattened()
                }
            elk_entailed = cls_iri in types_cache[ind_iri]
            assert (lbl == "entailed") == elk_entailed, (
                f"Entailed mismatch for ({ind_iri}, {cls_iri}): label={lbl}, ELK={elk_entailed}"
            )

            owl_ind = df.getOWLNamedIndividual(IRI.create(ind_iri))
            owl_cls = df.getOWLClass(IRI.create(cls_iri))
            ca_ax = df.getOWLClassAssertionAxiom(owl_cls, owl_ind)
            already_present = ont_train.containsAxiom(ca_ax)
            if not already_present:
                mgr.applyChange(AddAxiom(ont_train, ca_ax))
                reasoner.flush()
            elk_contradicted = not reasoner.isConsistent()
            if not already_present:
                mgr.applyChange(RemoveAxiom(ont_train, ca_ax))
                reasoner.flush()
            assert (lbl == "contradicted") == elk_contradicted, (
                f"Contradicted mismatch for ({ind_iri}, {cls_iri}): label={lbl}, ELK={elk_contradicted}"
            )
    finally:
        reasoner.dispose()
