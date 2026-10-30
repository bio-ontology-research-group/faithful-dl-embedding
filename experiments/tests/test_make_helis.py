"""Tests for the B3 helis-membership ABox benchmark builder (WP-B Task 2)."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path
import tarfile

import mowl

mowl.init_jvm("8g")

import numpy as np
import pandas as pd
import pytest

from experiments.data.fetch_helis import (
    ARCHIVE_SHA256_STAMP,
    HELIS_DL_DIR,
    HELIS_FILES_SHA256,
    HELIS_FILES_TARBALL_NAME,
    HELIS_ONTS_SHA256,
    HELIS_ONTS_TARBALL_NAME,
    _safe_extract_tar,
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
    STATS_SECTION_HEADER,
    build_helis,
    candidate_set_for_individual,
    parse_inferred_classes_file,
    render_helis_stats_section,
    update_dataset_stats_file,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def subset_helis_build(tmp_path_factory: pytest.TempPathFactory):
    """Exercise ``build_helis`` on a 300-individual subset into a ``~/tmp`` directory."""
    if not HELIS_DL_DIR.exists():
        pytest.skip(f"HeLiS input directory not found at {HELIS_DL_DIR}")

    sub_out_dir = tmp_path_factory.mktemp("helis_subset_300")
    assert not str(sub_out_dir.resolve()).startswith("/tmp/"), (
        f"tmp_path resolved under /tmp: {sub_out_dir}"
    )
    sub_stats = sub_out_dir / "dataset_stats.md"
    sub_info = build_helis(
        dl_dir=HELIS_DL_DIR,
        out_dir=sub_out_dir,
        stats_path=sub_stats,
        repo_mirror_dir=None,
        max_train_individuals=300,
        max_valid_individuals=50,
        max_test_individuals=100,
    )
    sub_manifest = json.loads(
        (sub_out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return sub_out_dir, sub_info, sub_manifest, sub_stats


def test_subset_build_and_full_split_counts(subset_helis_build):
    """Verify subset build exercises build_helis and full build has exact counts and materialised ranges."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AxiomType
    from org.semanticweb.owlapi.model.parameters import Imports
    import java.io

    sub_out_dir, sub_info, sub_manifest, sub_stats = subset_helis_build

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
    for fname in required_files:
        fpath = sub_out_dir / fname
        assert fpath.exists(), f"Missing subset output file {fname}"
    for fname, meta in sub_manifest["outputs"].items():
        fpath = sub_out_dir / fname
        assert meta["sha256"] == _sha256(fpath)
        assert meta["size_bytes"] == fpath.stat().st_size
    assert sub_stats.exists()

    sub_counts = sub_manifest["counts"]
    assert sub_counts["n_train_pos"] == 300
    assert sub_counts["n_train_neg_dropped"] == 300
    assert sub_counts["n_valid"] == 50
    assert sub_counts["n_test"] == 100
    assert sub_counts["n_classes"] == EXPECTED_N_CLASSES == 277
    assert sub_counts["n_candidates"] == EXPECTED_N_CLASSES == 277
    assert sub_counts["n_object_property_assertions"] > 0
    assert sub_counts["n_range_assertions_added"] > 0
    assert (
        sub_counts["n_total_train_class_assertions"]
        == 300 + sub_counts["n_range_assertions_added"]
    )
    assert sub_info["train_labels_summary"]["n_contradicted"] > 0

    mgr_sub = OWLManager.createOWLOntologyManager()
    ont_sub_train = mgr_sub.loadOntologyFromOntologyDocument(
        java.io.File(str(sub_out_dir / "ontology.owl"))
    )
    assert (
        ont_sub_train.getAxioms(AxiomType.CLASS_ASSERTION).size()
        == sub_counts["n_total_train_class_assertions"]
    )
    assert (
        ont_sub_train.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size()
        == sub_counts["n_object_property_assertions"]
    )
    assert ont_sub_train.getAxioms(AxiomType.DATA_PROPERTY_ASSERTION).size() == 0

    # Also verify full build in HELIS_OUT_DIR
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

    out_dir = HELIS_OUT_DIR
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]

    assert counts["n_individuals"] == EXPECTED_N_INDIVIDUALS == 20_318
    assert counts["n_classes"] == EXPECTED_N_CLASSES == 277
    assert counts["n_candidates"] == EXPECTED_N_CLASSES == 277
    assert counts["n_train_pos"] == EXPECTED_N_TRAIN_POS == 14_222
    assert counts["n_train_neg_dropped"] == EXPECTED_N_TRAIN_NEG == 14_222
    assert counts["n_valid"] == EXPECTED_N_VALID == 2_031
    assert counts["n_test"] == EXPECTED_N_TEST == 4_065
    assert counts["n_object_property_assertions"] == 83_210

    # Verify range materialisation counts and newly entailed held-out pairs
    assert counts["n_object_property_ranges_named"] == 15
    assert counts["n_range_assertions_raw"] == 83_210
    assert counts["n_range_assertions_unique"] == 17_460
    assert counts["n_range_assertions_added"] == 16_458
    assert counts["n_total_train_class_assertions"] == 30_680
    assert counts["range_assertions_unique_by_class"] == {
        "Nutrient": 15780,
        "RecipeFood": 1418,
        "Food": 262,
    }
    assert counts["range_assertions_added_by_class"] == {
        "Nutrient": 15780,
        "RecipeFood": 416,
        "Food": 262,
    }
    assert counts["n_valid_entailed_before_ranges"] == 212
    assert counts["n_test_entailed_before_ranges"] == 420
    assert counts["n_valid_newly_entailed_by_ranges"] == 1_759
    assert counts["n_test_newly_entailed_by_ranges"] == 3_470
    assert counts["n_heldout_newly_entailed_by_ranges"] == 5_229
    assert counts["newly_entailed_heldout_by_class"] == {
        "Nutrient": 4732,
        "RecipeFood": 416,
        "Food": 81,
    }
    assert counts["n_valid_gold_newly_entailed_by_ranges"] == 133
    assert counts["n_test_gold_newly_entailed_by_ranges"] == 283
    assert counts["n_heldout_gold_newly_entailed_by_ranges"] == 416

    # Verify KB vs gold-conditional label counts per split
    assert manifest["valid_labels"] == {
        "n_rows": 562_587,
        "n_individuals": 2_031,
        "n_candidates": 277,
        "n_entailed": 1_971,
        "n_unknown": 560_616,
        "n_contradicted": 0,
        "n_entailed_gold": 6_332,
        "n_unknown_gold": 546_308,
        "n_contradicted_gold": 9_947,
    }
    assert manifest["test_labels"] == {
        "n_rows": 1_126_005,
        "n_individuals": 4_065,
        "n_candidates": 277,
        "n_entailed": 3_890,
        "n_unknown": 1_122_115,
        "n_contradicted": 0,
        "n_entailed_gold": 12_497,
        "n_unknown_gold": 1_093_339,
        "n_contradicted_gold": 20_169,
    }

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
    assert ont_train.getAxioms(AxiomType.CLASS_ASSERTION).size() == 30_680
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

    for fname, meta in manifest["outputs"].items():
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing output file {fname}"
        assert meta["sha256"] == _sha256(fpath)
        assert meta["size_bytes"] == fpath.stat().st_size


def test_fetch_helis_reextracts_on_sha_mismatch_and_blocks_traversal(tmp_path: Path):
    """Verify trailing-separator traversal guard and re-extraction when recorded sha256 differs."""
    # 1. Trailing-separator path traversal check: member escaping to sibling "files-evil"
    dest_dir = tmp_path / "files"
    evil_tar = tmp_path / "evil.tar.gz"
    with tarfile.open(evil_tar, "w:gz") as tf:
        info = tarfile.TarInfo(name="../files-evil/pwn.txt")
        payload = b"evil"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe tar member path"):
        _safe_extract_tar(evil_tar, dest_dir)

    # 2. Re-extraction when recorded archive sha256 differs even if expected files exist
    if not HELIS_DL_DIR.exists():
        pytest.skip(f"HeLiS input directory not found at {HELIS_DL_DIR}")

    dl_copy = tmp_path / "helis-dl-copy"
    dl_copy.mkdir(parents=True, exist_ok=True)
    for tar_name in (HELIS_FILES_TARBALL_NAME, HELIS_ONTS_TARBALL_NAME):
        (dl_copy / tar_name).symlink_to(HELIS_DL_DIR / tar_name)

    fetch_helis(dl_dir=dl_copy)
    files_dir = dl_copy / "files"
    stamp_path = files_dir / ARCHIVE_SHA256_STAMP
    assert stamp_path.exists()
    assert stamp_path.read_text(encoding="utf-8").strip() == HELIS_FILES_SHA256

    # Corrupt a file AND change the recorded sha256 stamp -> fetch_helis must re-extract
    classes_file = files_dir / "classes.txt"
    orig_classes_bytes = classes_file.read_bytes()
    classes_file.write_text("corrupted\n", encoding="utf-8")
    stamp_path.write_text("0" * 64 + "\n", encoding="utf-8")

    fetch_helis(dl_dir=dl_copy)
    assert classes_file.read_bytes() == orig_classes_bytes
    assert stamp_path.read_text(encoding="utf-8").strip() == HELIS_FILES_SHA256


def test_helis_candidates_match_reference_protocol(subset_helis_build):
    """For 20 sampled test rows, candidate set equals classes.txt minus owl:Thing minus inferred_classes[a]."""
    sub_out_dir, _, _, _ = subset_helis_build
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

    for check_dir, expected_skip_len, rows_pool in [
        (sub_out_dir, 100, test_rows[:100]),
        (HELIS_OUT_DIR, 4_065, test_rows),
    ]:
        candidates = [
            line.strip()
            for line in (check_dir / "candidates.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert OWL_THING_SHORT not in candidates
        assert OWL_THING_IRI not in candidates
        assert set(candidates) == set(classes_txt) - {OWL_THING_SHORT, OWL_THING_IRI}

        skip_test_map: dict[str, list[str]] = {}
        for line in (
            (check_dir / "skip_test.tsv").read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            ind_iri, rest = line.split("\t", 1)
            skip_test_map[ind_iri] = [
                c.strip() for c in rest.split(",") if c.strip()
            ]
        assert len(skip_test_map) == expected_skip_len

        rng = np.random.RandomState(0)
        sampled_indices = rng.choice(len(rows_pool), size=20, replace=False)
        for idx in sampled_indices:
            ind_iri, gt_cls = rows_pool[idx]
            ref_candidates = (
                set(classes_txt)
                - {OWL_THING_SHORT, OWL_THING_IRI}
                - set(inferred_map[ind_iri])
            )
            built_candidates = set(candidates) - set(skip_test_map[ind_iri])
            assert built_candidates == ref_candidates
            assert (
                candidate_set_for_individual(ind_iri, classes_txt, inferred_map)
                == ref_candidates
            )
            assert gt_cls in built_candidates
            assert gt_cls not in skip_test_map[ind_iri]


def test_disjointness_survives_normalisation(subset_helis_build):
    """Verify both the 14-way food-group AllDisjoint and RedMeat/WhiteMeat disjointness survive in ontology.owl."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AxiomType, ClassExpressionType
    import java.io

    sub_out_dir, _, _, _ = subset_helis_build
    mgr = OWLManager.createOWLOntologyManager()
    ont = mgr.loadOntologyFromOntologyDocument(
        java.io.File(str(sub_out_dir / "ontology.owl"))
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


def test_labels_match_elk_sample(subset_helis_build):
    """Verify KB `label` and `label_gold` against ELK on 200 pairs across both valid and test parquets."""
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import AddAxiom, IRI, RemoveAxiom
    from org.semanticweb.owlapi.reasoner import InferenceType
    from mowl.geometric import CanonicalModel
    import java.io

    sub_out_dir, _, _, _ = subset_helis_build
    out_dir = HELIS_OUT_DIR
    files_dir = HELIS_DL_DIR / "files"

    # Check subset parquet schema as well
    df_sub_valid = pd.read_parquet(sub_out_dir / "labels_valid.parquet")
    df_sub_test = pd.read_parquet(sub_out_dir / "labels_test.parquet")
    assert list(df_sub_valid.columns) == ["individual", "class", "label", "label_gold"]
    assert list(df_sub_test.columns) == ["individual", "class", "label", "label_gold"]
    assert len(df_sub_valid) == 50 * 277
    assert len(df_sub_test) == 100 * 277

    # Check full parquets against ELK across both labels_valid.parquet and labels_test.parquet
    candidates = [
        line.strip()
        for line in (out_dir / "candidates.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df_valid = pd.read_parquet(out_dir / "labels_valid.parquet")
    df_test = pd.read_parquet(out_dir / "labels_test.parquet")

    assert list(df_valid.columns) == ["individual", "class", "label", "label_gold"]
    assert list(df_test.columns) == ["individual", "class", "label", "label_gold"]
    assert len(df_valid) == 2_031 * 277
    assert len(df_test) == 4_065 * 277
    assert set(df_valid["label_gold"].unique()) == {
        "entailed_gold",
        "unknown",
        "contradicted_gold",
    }
    assert set(df_test["label_gold"].unique()) == {
        "entailed_gold",
        "unknown",
        "contradicted_gold",
    }

    gold_map: dict[str, str] = {}
    for split_csv in ("valid.csv", "test.csv"):
        for line in (files_dir / split_csv).read_text(encoding="utf-8").splitlines():
            if line.strip():
                i_str, c_str = [p.strip() for p in line.strip().split(",")]
                gold_map[i_str] = c_str

    # Sample 100 rows from labels_valid.parquet and 100 rows from labels_test.parquet
    # covering KB entailed, KB unknown, and all three label_gold classes.
    rng = np.random.RandomState(0)
    sample_rows: list[tuple[str, str, str, str]] = []
    for df_split in (df_valid, df_test):
        kb_ent = df_split[df_split["label"] == "entailed"]
        gold_ent = df_split[
            (df_split["label"] == "unknown")
            & (df_split["label_gold"] == "entailed_gold")
        ]
        gold_contra = df_split[df_split["label_gold"] == "contradicted_gold"]
        both_unk = df_split[
            (df_split["label"] == "unknown") & (df_split["label_gold"] == "unknown")
        ]

        for pool, k in [
            (kb_ent, 30),
            (gold_ent, 20),
            (gold_contra, 25),
            (both_unk, 25),
        ]:
            assert len(pool) >= k
            idx = rng.choice(len(pool), size=k, replace=False)
            for row in pool.iloc[idx].itertuples(index=False):
                sample_rows.append(
                    (
                        str(row.individual),
                        str(row._1),
                        str(row.label),
                        str(row.label_gold),
                    )
                )

    assert len(sample_rows) == 200

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
        types_cache: dict[str, set[str]] = {}
        gold_sups_cache: dict[str, set[str]] = {}
        joint_sat_cache: dict[frozenset[str], bool] = {}

        for ind_iri, cls_iri, lbl_kb, lbl_gold in sample_rows:
            # 1. Verify KB label (`label`) against ELK
            if ind_iri not in types_cache:
                owl_ind = df.getOWLNamedIndividual(IRI.create(ind_iri))
                types_cache[ind_iri] = {
                    str(c.toStringID())
                    for c in reasoner.getTypes(owl_ind, False).getFlattened()
                }
            elk_entailed = cls_iri in types_cache[ind_iri]
            assert (lbl_kb == "entailed") == elk_entailed, (
                f"KB entailed mismatch for ({ind_iri}, {cls_iri}): label={lbl_kb}, ELK={elk_entailed}"
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
            assert (lbl_kb == "contradicted") == elk_contradicted, (
                f"KB contradicted mismatch for ({ind_iri}, {cls_iri}): label={lbl_kb}, ELK={elk_contradicted}"
            )

            # 2. Verify gold-conditional label (`label_gold`) against ELK
            gold_cls = gold_map[ind_iri]
            if gold_cls not in gold_sups_cache:
                owl_gold = df.getOWLClass(IRI.create(gold_cls))
                sups = {
                    str(c.toStringID())
                    for c in reasoner.getSuperClasses(owl_gold, False).getFlattened()
                }
                eqs = {
                    str(c.toStringID())
                    for c in reasoner.getEquivalentClasses(owl_gold).getEntities()
                }
                gold_sups_cache[gold_cls] = sups | eqs | {gold_cls}

            elk_gold_entailed = cls_iri in gold_sups_cache[gold_cls]
            assert (lbl_gold == "entailed_gold") == elk_gold_entailed, (
                f"Gold entailed mismatch for ({ind_iri}, {cls_iri}): label_gold={lbl_gold}, ELK={elk_gold_entailed}"
            )

            pair_key = frozenset([gold_cls, cls_iri])
            if pair_key not in joint_sat_cache:
                conj = df.getOWLObjectIntersectionOf(
                    df.getOWLClass(IRI.create(gold_cls)),
                    df.getOWLClass(IRI.create(cls_iri)),
                )
                joint_sat_cache[pair_key] = bool(reasoner.isSatisfiable(conj))
            elk_gold_contradicted = not joint_sat_cache[pair_key]
            assert (lbl_gold == "contradicted_gold") == elk_gold_contradicted, (
                f"Gold contradicted mismatch for ({ind_iri}, {cls_iri}): label_gold={lbl_gold}, ELK={elk_gold_contradicted}"
            )

        # Also cross-check unseeded CanonicalModel public methods on subset training ontology
        mgr_sub = OWLManager.createOWLOntologyManager()
        ont_sub_train = mgr_sub.loadOntologyFromOntologyDocument(
            java.io.File(str(sub_out_dir / "ontology.owl"))
        )
        cm_sub = CanonicalModel.from_ontology(ont_sub_train)
        checked_contra = 0
        for ind_iri in sorted(cm_sub.individuals.keys()):
            ent = [c for c in candidates if cm_sub.instance_of(ind_iri, c)]
            if any(c in FOOD_GROUP_14_CLASSES for c in ent):
                ent_fg = [c for c in ent if c in FOOD_GROUP_14_CLASSES][0]
                other_fg = [c for c in FOOD_GROUP_14_CLASSES if c != ent_fg][0]
                assert cm_sub.instance_of(ind_iri, ent_fg) is True
                assert cm_sub.assertion_contradicted(ind_iri, other_fg) is True
                checked_contra += 1
                if checked_contra >= 5:
                    break
        assert checked_contra >= 5
    finally:
        reasoner.dispose()


def test_update_dataset_stats_preserves_other_sections_byte_for_byte(
    tmp_path: Path, subset_helis_build
):
    """Verify a file with sections B1, B3, and S6 has B1 and S6 preserved byte-for-byte after B3 update."""
    _, sub_info, _, _ = subset_helis_build

    preceding_sections = (
        "# Experiment Dataset Statistics (`Task 6`)\n\n"
        "## 4. ABox Benchmark B1: `yeast-function` Statistics\n\n"
        "Generated by `experiments/data/make_yeast_function.py`.\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        "| Proteins | 6,030 |\n\n"
        "- **Protocol Note**: Yeast section preserved byte-for-byte.\n\n"
    )
    old_b3 = (
        f"{STATS_SECTION_HEADER}\n\n"
        "Old B3 content to be replaced.\n\n"
    )
    following_section = (
        "## 6. Future Benchmark Section\n\n"
        "Trailing section content preserved byte-for-byte.\n"
    )

    stats_file = tmp_path / "dataset_stats.md"
    stats_file.write_bytes(
        (preceding_sections + old_b3 + following_section).encode("utf-8")
    )

    update_dataset_stats_file(stats_file, sub_info)
    updated_bytes = stats_file.read_bytes()
    updated_text = updated_bytes.decode("utf-8")

    expected_b3 = render_helis_stats_section(sub_info)
    expected_full = (
        preceding_sections + expected_b3.rstrip("\n") + "\n\n" + following_section
    ).encode("utf-8")

    assert updated_bytes == expected_full
    assert updated_text.startswith(preceding_sections)
    assert updated_text.endswith(following_section)
    assert "Old B3 content" not in updated_text

    # Re-running update_dataset_stats_file must be idempotent
    update_dataset_stats_file(stats_file, sub_info)
    assert stats_file.read_bytes() == expected_full
