"""Dataset bundle loader and PathDataset subsumption shim for experiments (Task 6).

Guarantees
----------
- ``load_dataset(name)`` accepts every identifier in ``DATASETS``
  (``tk3``, ``tk4``, ``tk5``, ``tk6``, ``pizza``, ``go``, ``foodon``) and returns
  a ``DatasetBundle(name, dataset, deductive_closure_ontology, task, paths)``.
- For ``task == "satisfaction"`` (``tk3..tk6``), ``deductive_closure_ontology`` is
  ``None``.
- For ``task == "subsumption"`` (``pizza``, ``go``, ``foodon``), ``dataset``
  exposes ``.validation``, ``.testing``, ``.deductive_closure_ontology``, and
  ``.evaluation_classes`` (compatible with ``mowl.evaluation.SubsumptionEvaluator``).

Non-guarantees
--------------
- Raises ``ValueError`` on unknown dataset identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Dict, Literal, Optional

import mowl

from experiments.config import (
    DATASETS,
    DATASET_TASKS,
    HELIS_DIR,
    MOWL_DATA_ROOT,
    PIZZA_EL_DIR,
    PIZZA_REPO_DIR,
    SYNTHETIC_DIR,
    YEAST_FUNCTION_DIR,
)
from experiments.data.make_pizza import is_auxiliary_iri

if TYPE_CHECKING:  # pragma: no cover
    from org.semanticweb.owlapi.model import OWLOntology
    from mowl.datasets import Dataset

TaskType = Literal["satisfaction", "subsumption", "abox"]


@dataclass
class DatasetBundle:
    """Container for a loaded experiment dataset and its evaluation metadata.

    Attributes
    ----------
    name : str
        Dataset identifier in ``DATASETS``.
    dataset : mowl.datasets.Dataset
        Loaded mOWL dataset object.
    deductive_closure_ontology : OWLOntology or None
        Deductive closure ontology for filtered ranking evaluation on subsumption
        benchmarks, or ``None`` for satisfaction-only synthetic benchmarks.
    task : {"satisfaction", "subsumption"}
        Evaluation task category.
    paths : Dict[str, str]
        Mapping of logical file keys (e.g., ``"ontology"``, ``"valid"``, ``"test"``,
        ``"deductive_closure"``) to resolved filesystem paths.

    Guarantees
    ----------
    - ``task == "subsumption"`` iff ``deductive_closure_ontology is not None``.

    Non-guarantees
    --------------
    - Does not eagerly normalise the ontology or build a ``CanonicalModel``.
    """

    name: str
    dataset: "Dataset"
    deductive_closure_ontology: Optional["OWLOntology"]
    task: TaskType
    paths: Dict[str, str]


def _make_path_subsumption_dataset(
    ontology_path: str,
    validation_path: str,
    testing_path: str,
    deductive_closure_path: str,
):
    """Create a ``PathDataset`` subclass instance with ``SubsumptionDataset`` properties.

    Parameters
    ----------
    ontology_path : str
        Path to ``ontology.owl``.
    validation_path : str
        Path to ``valid.owl``.
    testing_path : str
        Path to ``test.owl``.
    deductive_closure_path : str
        Path to ``ontology_deductive_closure.owl``.

    Returns
    -------
    PathDataset
        Dataset instance exposing ``.deductive_closure_ontology`` and
        ``.evaluation_classes``.
    """
    import java
    from org.semanticweb.owlapi.apibinding import OWLManager
    from mowl.datasets import OWLClasses, PathDataset

    class PathSubsumptionDataset(PathDataset):
        """Local-path subsumption dataset matching ``mowl.datasets.builtin.SubsumptionDataset``."""

        def __init__(
            self,
            ont_p: str,
            val_p: str,
            test_p: str,
            dc_p: str,
        ) -> None:
            super().__init__(ont_p, validation_path=val_p, testing_path=test_p)
            self.deductive_closure_ontology_path = dc_p
            self._deductive_closure_ontology = None

        @property
        def deductive_closure_ontology(self):
            if self._deductive_closure_ontology is None:
                ont_manager = OWLManager.createOWLOntologyManager()
                self._deductive_closure_ontology = (
                    ont_manager.loadOntologyFromOntologyDocument(
                        java.io.File(self.deductive_closure_ontology_path)
                    )
                )
            return self._deductive_closure_ontology

        @property
        def evaluation_classes(self):
            if self._evaluation_classes is None:
                train_classes = self.ontology.getClassesInSignature()
                valid_classes = self.validation.getClassesInSignature()
                test_classes = self.testing.getClassesInSignature()
                dc_classes = self.deductive_closure_ontology.getClassesInSignature()
                assert set(valid_classes) - set(train_classes) == set(), (
                    f"Valid classes not in train: {set(valid_classes) - set(train_classes)}"
                )
                assert set(test_classes) - set(train_classes) == set(), (
                    f"Test classes not in train: {set(test_classes) - set(train_classes)}"
                )
                assert set(dc_classes) - set(train_classes) == set(), (
                    f"Deductive closure classes not in train: {set(dc_classes) - set(train_classes)}"
                )

                # I1: Exclude auxiliary normaliser classes from evaluation_classes
                classes = {
                    cls
                    for cls in self.ontology.getClassesInSignature()
                    if not is_auxiliary_iri(str(cls.toStringID()))
                }
                bot_in_classes = any(cls.isOWLNothing() for cls in classes)
                top_in_classes = any(cls.isOWLThing() for cls in classes)
                df = self.ontology.getOWLOntologyManager().getOWLDataFactory()
                if not bot_in_classes:
                    classes.add(df.getOWLNothing())
                if not top_in_classes:
                    classes.add(df.getOWLThing())

                owl_classes = OWLClasses(classes)
                self._evaluation_classes = (owl_classes, owl_classes)
            return self._evaluation_classes

    return PathSubsumptionDataset(
        ontology_path, validation_path, testing_path, deductive_closure_path
    )


def _file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_pizza_files(pizza_dir: Path = PIZZA_EL_DIR) -> Dict[str, str]:
    """Ensure the four Pizza-EL OWL files exist and return their paths.

    Prefers the committed repository mirror ``experiments/data/pizza-el`` over any
    copy in ``pizza_dir`` (M4), syncing ``pizza_dir`` whenever a file is missing or
    its SHA-256 checksum differs from the committed mirror.
    """
    filenames = {
        "ontology": "ontology.owl",
        "valid": "valid.owl",
        "test": "test.owl",
        "deductive_closure": "ontology_deductive_closure.owl",
    }
    mirror_files = {k: PIZZA_REPO_DIR / fname for k, fname in filenames.items()}
    required = {k: pizza_dir / fname for k, fname in filenames.items()}

    if all(p.exists() for p in mirror_files.values()):
        pizza_dir.mkdir(parents=True, exist_ok=True)
        for k, src in mirror_files.items():
            dst = required[k]
            if src.resolve() != dst.resolve():
                if not dst.exists() or _file_sha256(src) != _file_sha256(dst):
                    shutil.copy2(src, dst)
        return {k: str(src.resolve()) for k, src in mirror_files.items()}

    if all(p.exists() for p in required.values()):
        return {k: str(p.resolve()) for k, p in required.items()}

    from experiments.data.make_pizza import build_pizza_el

    info = build_pizza_el(out_dir=pizza_dir)
    return info["paths"]


def load_dataset(name: str, data_root: Path | str = MOWL_DATA_ROOT) -> DatasetBundle:
    """Load an experiment dataset by identifier and return its ``DatasetBundle``.

    Parameters
    ----------
    name : str
        Dataset identifier in ``DATASETS``
        (``"tk3"``, ``"tk4"``, ``"tk5"``, ``"tk6"``, ``"pizza"``, ``"go"``, ``"foodon"``).
    data_root : Path or str, optional
        Cache directory for mOWL remote datasets ``go`` and ``foodon``
        (default: ``~/tmp/gfe-data/mowl``).

    Returns
    -------
    DatasetBundle
        Loaded dataset bundle with task and file paths.

    Guarantees
    ----------
    - Initialises the JVM if not already started.
    - Generates ``tk{k}.owl`` or Pizza-EL files on demand if not yet present on disk.

    Non-guarantees
    --------------
    - Raises ``ValueError`` if ``name`` is not in ``DATASETS``.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")

    mowl.init_jvm("8g")
    from mowl.datasets import PathDataset
    from mowl.datasets.builtin import FoodOnSubsumptionDataset, GOSubsumptionDataset

    task: TaskType = DATASET_TASKS[name]  # type: ignore[assignment]

    if name.startswith("tk"):
        k = int(name[2:])
        ont_path = SYNTHETIC_DIR / f"{name}.owl"
        if not ont_path.exists():
            from experiments.data.make_synthetic import generate_all_synthetic

            generate_all_synthetic(out_dir=SYNTHETIC_DIR, ks=(k,))
        ds = PathDataset(str(ont_path.resolve()))
        return DatasetBundle(
            name=name,
            dataset=ds,
            deductive_closure_ontology=None,
            task=task,
            paths={"ontology": str(ont_path.resolve())},
        )

    if name == "pizza":
        paths = _ensure_pizza_files(PIZZA_EL_DIR)
        ds = _make_path_subsumption_dataset(
            ontology_path=paths["ontology"],
            validation_path=paths["valid"],
            testing_path=paths["test"],
            deductive_closure_path=paths["deductive_closure"],
        )
        return DatasetBundle(
            name=name,
            dataset=ds,
            deductive_closure_ontology=ds.deductive_closure_ontology,
            task=task,
            paths=paths,
        )

    if name == "yeast-function":
        yf_dir = YEAST_FUNCTION_DIR
        required = {
            "ontology": yf_dir / "ontology.owl",
            "valid": yf_dir / "valid.owl",
            "test": yf_dir / "test.owl",
            "candidates": yf_dir / "candidates.txt",
            "labels_valid": yf_dir / "labels_valid.parquet",
            "labels_test": yf_dir / "labels_test.parquet",
            "train_annotations": yf_dir / "train_annotations.tsv",
            "valid_annotations": yf_dir / "valid_annotations.tsv",
            "test_annotations": yf_dir / "test_annotations.tsv",
            "manifest": yf_dir / "manifest.json",
        }
        if not all(p.exists() for p in required.values()):
            from experiments.data.make_yeast_function import build_yeast_function

            build_yeast_function(out_dir=yf_dir)
        paths = {k: str(p.resolve()) for k, p in required.items()}
        ds = PathDataset(
            paths["ontology"],
            validation_path=paths["valid"],
            testing_path=paths["test"],
        )
        return DatasetBundle(
            name=name,
            dataset=ds,
            deductive_closure_ontology=None,
            task=task,
            paths=paths,
        )

    if name == "helis":
        hl_dir = HELIS_DIR
        required = {
            "ontology": hl_dir / "ontology.owl",
            "valid": hl_dir / "valid.owl",
            "test": hl_dir / "test.owl",
            "candidates": hl_dir / "candidates.txt",
            "labels_valid": hl_dir / "labels_valid.parquet",
            "labels_test": hl_dir / "labels_test.parquet",
            "skip_valid": hl_dir / "skip_valid.tsv",
            "skip_test": hl_dir / "skip_test.tsv",
            "manifest": hl_dir / "manifest.json",
        }
        if not all(p.exists() for p in required.values()):
            from experiments.data.make_helis import build_helis

            build_helis(out_dir=hl_dir)
        paths = {k: str(p.resolve()) for k, p in required.items()}
        ds = PathDataset(
            paths["ontology"],
            validation_path=paths["valid"],
            testing_path=paths["test"],
        )
        return DatasetBundle(
            name=name,
            dataset=ds,
            deductive_closure_ontology=None,
            task=task,
            paths=paths,
        )

    resolved_root = str(Path(data_root).expanduser().resolve())
    if name == "go":
        ds_remote = GOSubsumptionDataset(data_root=resolved_root)
    elif name == "foodon":
        ds_remote = FoodOnSubsumptionDataset(data_root=resolved_root)
    else:  # pragma: no cover
        raise ValueError(f"Unhandled dataset {name!r}")

    dc_ont = ds_remote.deductive_closure_ontology
    paths = {
        "ontology": str(Path(ds_remote.ontology_path).resolve()),
        "valid": str(Path(ds_remote.validation_path).resolve()),
        "test": str(Path(ds_remote.testing_path).resolve()),
        "deductive_closure": str(
            Path(ds_remote.deductive_closure_ontology_path).resolve()
        ),
        "tarfile": str(Path(ds_remote.tarfile_path).resolve()),
    }
    return DatasetBundle(
        name=name,
        dataset=ds_remote,
        deductive_closure_ontology=dc_ont,
        task=task,
        paths=paths,
    )
