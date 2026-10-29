"""Disk-backed pickle cache for ``mowl.geometric.CanonicalModel`` (Task 7).

Building the :class:`~mowl.geometric.CanonicalModel` of large ontologies such as
GO or FoodOn requires normalising the TBox/RBox and running ELK worklist
saturation. Because :class:`~mowl.geometric.CanonicalModel` and its underlying
:class:`~mowl.geometric.completion.CompletionResult` and normal-form dataclasses
(:class:`~mowl.geometric.normal_forms.NF1`–:class:`~mowl.geometric.normal_forms.NF7`)
hold exclusively plain-Python state (``str``, ``int``, ``bool``, ``tuple``,
``list``, ``set``, ``frozenset``, ``dict``, ``defaultdict(set)``) and no JVM /
JPype objects, a built :class:`~mowl.geometric.CanonicalModel` can be pickled
directly with ``pickle`` protocol 5.

Guarantees
----------
- ``get_canonical_model(bundle)`` caches the built model at
  ``~/tmp/gfe-data/cm-cache/<dataset-id>-<sha1 of the training ontology file>.pkl``
  using ``pickle`` protocol 5 and atomic rename (``os.replace``), and reloads it
  on subsequent calls when the file is present.
- Also caches the loaded model in memory on ``bundle._canonical_model`` so
  repeated metric calls in the same process do not re-read disk.
- Provides a CLI ``python -m experiments.cm_cache --dataset <id>`` (or
  ``--dataset all``) to pre-build caches.

Non-guarantees
--------------
- Requires ``bundle.paths["ontology"]`` (or ``bundle.dataset.ontology_path``) to
  point to an existing ontology file on disk to compute the SHA-1 cache key.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import pickle
import time
from typing import TYPE_CHECKING, List, Optional

import mowl

from experiments.config import DATASETS, RAW_DATA_ROOT
from experiments.datasets import DatasetBundle, load_dataset

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric import CanonicalModel

CM_CACHE_DIR: Path = RAW_DATA_ROOT / "cm-cache"


def _sha1_file(path: Path) -> str:
    """Compute the full 40-character hexadecimal SHA-1 digest of ``path``."""
    digest = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_ontology_path(bundle: DatasetBundle) -> Path:
    """Resolve the training ontology file path from ``bundle``."""
    ont_str: Optional[str] = None
    if hasattr(bundle, "paths") and isinstance(bundle.paths, dict):
        ont_str = bundle.paths.get("ontology")
    if not ont_str and hasattr(bundle.dataset, "ontology_path"):
        ont_str = getattr(bundle.dataset, "ontology_path")
    if not ont_str:
        raise ValueError(
            f"Cannot determine training ontology path for bundle {bundle.name!r}"
        )
    p = Path(ont_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Training ontology file for {bundle.name!r} does not exist: {p}"
        )
    return p


def cache_path_for_bundle(
    bundle: DatasetBundle, cache_dir: Path | str = CM_CACHE_DIR
) -> Path:
    """Return ``<cache_dir>/<dataset-id>-<sha1>.pkl`` for ``bundle``."""
    ont_path = _resolve_ontology_path(bundle)
    sha1_hex = _sha1_file(ont_path)
    out_dir = Path(cache_dir).expanduser().resolve()
    return out_dir / f"{bundle.name}-{sha1_hex}.pkl"


def get_canonical_model(
    bundle: DatasetBundle,
    *,
    cache_dir: Path | str = CM_CACHE_DIR,
    force_rebuild: bool = False,
) -> "CanonicalModel":
    """Load or build and pickle-cache the ``CanonicalModel`` for ``bundle``.

    Parameters
    ----------
    bundle : DatasetBundle
        Loaded dataset bundle whose training ontology will be saturated.
    cache_dir : Path or str, optional
        Directory for ``.pkl`` files (default: ``~/tmp/gfe-data/cm-cache``).
    force_rebuild : bool, optional
        If ``True``, rebuild the ``CanonicalModel`` even if the pickle file exists.

    Returns
    -------
    mowl.geometric.CanonicalModel
        Saturated canonical model of ``bundle.dataset``.
    """
    mowl.init_jvm("8g")
    from mowl.geometric import CanonicalModel

    pkl_path = cache_path_for_bundle(bundle, cache_dir=cache_dir)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force_rebuild
        and pkl_path.exists()
        and getattr(bundle, "_canonical_model", None) is not None
        and getattr(bundle, "_canonical_model_path", None) == str(pkl_path)
    ):
        return getattr(bundle, "_canonical_model")

    if not force_rebuild and pkl_path.exists():
        with pkl_path.open("rb") as f:
            cm: CanonicalModel = pickle.load(f)
        setattr(bundle, "_canonical_model", cm)
        setattr(bundle, "_canonical_model_path", str(pkl_path))
        return cm

    cm = CanonicalModel.from_dataset(bundle.dataset)
    tmp_path = pkl_path.with_name(f"{pkl_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(cm, f, protocol=5)
        os.replace(tmp_path, pkl_path)
    except BaseException:  # pragma: no cover
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    setattr(bundle, "_canonical_model", cm)
    setattr(bundle, "_canonical_model_path", str(pkl_path))
    return cm


def main(argv: List[str] | None = None) -> None:
    """CLI entry point: ``python -m experiments.cm_cache --dataset <id|all>``."""
    parser = argparse.ArgumentParser(
        description="Pre-build and cache CanonicalModel pickles for experiment datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Dataset identifier in {DATASETS} or 'all'.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CM_CACHE_DIR,
        help="Directory for cached .pkl files (default: ~/tmp/gfe-data/cm-cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding the CanonicalModel even if a cache file exists.",
    )
    args = parser.parse_args(argv)

    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds_name in target_datasets:
        t0 = time.monotonic()
        bundle = load_dataset(ds_name)
        pkl_path = cache_path_for_bundle(bundle, cache_dir=args.cache_dir)
        was_cached = pkl_path.exists() and not args.force
        cm = get_canonical_model(
            bundle, cache_dir=args.cache_dir, force_rebuild=args.force
        )
        elapsed = time.monotonic() - t0
        size_mb = pkl_path.stat().st_size / (1024.0 * 1024.0)
        status = "reloaded" if was_cached else "built+saved"
        print(
            f"[{ds_name}] {status} CanonicalModel: {pkl_path} "
            f"({size_mb:.2f} MB, {len(cm.names)} satisfiable names, {elapsed:.2f}s)"
        )


if __name__ == "__main__":
    main()
