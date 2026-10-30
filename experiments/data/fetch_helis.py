#!/usr/bin/env python3
"""Download, verify SHA-256 checksums, and extract the OWL2Vec* HeLiS tarballs (Spec §3.2).

Verifies ``helis_membership_files.tar.gz`` and ``ontologies.tar.gz`` under
``~/tmp/gfe-data/helis-dl/`` against their pinned SHA-256 digests, downloads
either tarball from ``https://github.com/KRR-Oxford/OWL2Vec-Star`` (`case_studies`)
if missing, and extracts them into ``files/`` and ``onts/`` if not already extracted.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
from typing import Any, Dict, List
import urllib.request

from experiments.config import RAW_DATA_ROOT

HELIS_DL_DIR: Path = RAW_DATA_ROOT / "helis-dl"

HELIS_FILES_TARBALL_NAME: str = "helis_membership_files.tar.gz"
HELIS_FILES_URL: str = (
    "https://raw.githubusercontent.com/KRR-Oxford/OWL2Vec-Star/master/"
    "case_studies/helis_membership/files.tar.gz"
)
HELIS_FILES_SHA256: str = (
    "88dcc42994dde689dadeecf159562f873d30af0da63aeb3cf46aafbd29a67cd1"
)

HELIS_ONTS_TARBALL_NAME: str = "ontologies.tar.gz"
HELIS_ONTS_URL: str = (
    "https://raw.githubusercontent.com/KRR-Oxford/OWL2Vec-Star/master/"
    "case_studies/ontologies.tar.gz"
)
HELIS_ONTS_SHA256: str = (
    "44ddca245e1cd98cb59def740e454b55536ce6835180919f45689da97c92a8f1"
)

EXPECTED_EXTRACTED_FILES: List[str] = [
    "train.csv",
    "valid.csv",
    "test.csv",
    "classes.txt",
    "individuals.txt",
    "inferred_classes.txt",
]
EXPECTED_ORIGIN_RELPATH: str = "data/helis_v1.00.origin.owl"


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_tarball(path: Path, url: str, expected_sha256: str) -> Dict[str, Any]:
    """Ensure ``path`` exists and matches ``expected_sha256``, downloading from ``url`` only if missing."""
    downloaded = False
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.replace(path)
        downloaded = True

    actual_sha256 = compute_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": str(path),
        "url": url,
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
        "downloaded": downloaded,
    }


def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Safely extract ``tar_path`` into ``dest_dir`` preventing path traversal."""
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest_resolved / member.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe tar member path {member.name!r} in {tar_path}")
        tf.extractall(path=str(dest_resolved))


def fetch_helis(dl_dir: Path = HELIS_DL_DIR) -> Dict[str, Any]:
    """Verify (or download if missing) and extract the HeLiS membership and ontology tarballs.

    Parameters
    ----------
    dl_dir : Path, optional
        Directory storing the downloaded tarballs and extracted ``files/`` and
        ``onts/`` subdirectories (default: ``~/tmp/gfe-data/helis-dl``).

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary with tarball paths, URLs, SHA-256 checksums, and
        extracted file paths.
    """
    dl_dir = Path(dl_dir).expanduser().resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)

    files_tar = dl_dir / HELIS_FILES_TARBALL_NAME
    onts_tar = dl_dir / HELIS_ONTS_TARBALL_NAME

    files_meta = _ensure_tarball(files_tar, HELIS_FILES_URL, HELIS_FILES_SHA256)
    onts_meta = _ensure_tarball(onts_tar, HELIS_ONTS_URL, HELIS_ONTS_SHA256)

    files_dir = dl_dir / "files"
    onts_dir = dl_dir / "onts"

    if not all((files_dir / fname).exists() for fname in EXPECTED_EXTRACTED_FILES):
        _safe_extract_tar(files_tar, files_dir)

    origin_owl = onts_dir / EXPECTED_ORIGIN_RELPATH
    if not origin_owl.exists():
        _safe_extract_tar(onts_tar, onts_dir)

    for fname in EXPECTED_EXTRACTED_FILES:
        fpath = files_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected extracted file missing: {fpath}")
    if not origin_owl.exists():
        raise FileNotFoundError(f"Expected extracted ontology missing: {origin_owl}")

    return {
        "dl_dir": str(dl_dir),
        "files_dir": str(files_dir),
        "origin_owl": str(origin_owl),
        "tarballs": {
            HELIS_FILES_TARBALL_NAME: files_meta,
            HELIS_ONTS_TARBALL_NAME: onts_meta,
        },
    }


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for verifying/downloading and extracting HeLiS input tarballs."""
    parser = argparse.ArgumentParser(
        description="Verify or download OWL2Vec* HeLiS tarballs by SHA-256 and extract."
    )
    parser.add_argument(
        "--dl-dir",
        type=Path,
        default=HELIS_DL_DIR,
        help="Target directory for HeLiS tarballs and extracted files.",
    )
    args = parser.parse_args(argv)
    meta = fetch_helis(dl_dir=args.dl_dir)
    f_tar = meta["tarballs"][HELIS_FILES_TARBALL_NAME]
    o_tar = meta["tarballs"][HELIS_ONTS_TARBALL_NAME]
    print(
        f"Verified HeLiS inputs in {meta['dl_dir']} "
        f"({HELIS_FILES_TARBALL_NAME}: sha256={f_tar['sha256'][:12]}..., "
        f"{HELIS_ONTS_TARBALL_NAME}: sha256={o_tar['sha256'][:12]}...)"
    )


if __name__ == "__main__":
    main()
