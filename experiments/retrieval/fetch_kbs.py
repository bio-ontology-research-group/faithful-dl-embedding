#!/usr/bin/env python3
"""Fetch the six public retrieval-benchmark OWL knowledge bases.

Inputs are public source URLs; outputs are OWL files and a provenance manifest
(URL, archive member, sha256, size) under ``$GFE_RETRIEVAL_DATA/kbs``. Each download
is kept in its own directory under ``kbs/_downloads``; archives are only listed and
read member-wise with :mod:`zipfile`, never executed or unpickled.

Sources: the Ontolearn ``KGs.zip`` (carcinogenesis, mutagenesis, family, father); the
DL-Learner examples (semantic_bible); and for vicodi, which ``KGs.zip`` lacks, the member
``NCESData/vicodi/vicodi.owl`` of ``NCESData.zip``, the archive linked from the Ontolearn
README. Only that member is transferred, by HTTP range requests on the zip's central
directory and the member's bytes; if the server does not honour ranges the whole archive is
downloaded into ``kbs/_downloads/vicodi`` first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

from common import DATA_ROOT, NAMES

ZIP_URL = "https://files.dice-research.org/projects/Ontolearn/KGs.zip"
USER_AGENT = "faithful-dl-embedding-retrieval/1.0"
DIRECT_URLS = {"semantic_bible": "https://raw.githubusercontent.com/SmartDataAnalytics/DL-Learner/master/examples/semantic_bible/NTNcombined.owl"}
ZIP_MEMBER_SOURCES = {
    "vicodi": {
        "url": "https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCESData.zip",
        "member": "NCESData/vicodi/vicodi.owl",
        "referenced_by": "https://github.com/dice-group/Ontolearn README (NCESData.zip download instructions)",
    },
}
ALIASES = {"semantic_bible": ("semanticbible", "semantic_bible", "semantic bible"), "family": ("family",), "father": ("father",), "carcinogenesis": ("carcinogenesis",), "mutagenesis": ("mutagenesis",), "vicodi": ("vicodi",)}


def sha256(path: Path) -> str:
    """Compute SHA-256 without loading the whole file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file backed by HTTP ``Range`` requests.

    Parameters
    ----------
    url : str
        Remote file URL; the server must answer ranged GETs with status 206.
    """

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(head, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.last_modified = response.headers.get("Last-Modified")
        self.bytes_read = 0

    def readable(self) -> bool:  # noqa: D102
        return True

    def seekable(self) -> bool:  # noqa: D102
        return True

    def tell(self) -> int:  # noqa: D102
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:  # noqa: D102
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:  # noqa: D102
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.size, self.pos + len(buffer)) - 1
        request = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise OSError(f"server ignored Range header (status {response.status})")
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        self.bytes_read += len(data)
        return len(data)


def fetch_zip_member(name: str, root: Path) -> dict:
    """Extract one configured archive member into ``kbs/<name>/<name>.owl``; return its manifest entry."""
    source = ZIP_MEMBER_SOURCES[name]
    staging = root / "kbs" / "_downloads" / name
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / Path(source["member"]).name
    entry = {"url": source["url"], "archive_member": source["member"], "referenced_by": source["referenced_by"]}
    try:
        remote = HttpRangeFile(source["url"])
        entry.update(archive_size_bytes=remote.size, archive_last_modified=remote.last_modified)
        try:
            archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
            entry["method"] = "HTTP range requests (zip central directory + member only)"
        except OSError:
            local = staging / Path(source["url"]).name
            request = urllib.request.Request(source["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=600) as response, local.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            archive = zipfile.ZipFile(local)
            entry.update(method="full archive download", archive_sha256=sha256(local))
        with archive, archive.open(source["member"]) as src, staged.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        target = root / "kbs" / name / f"{name}.owl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(staged.read_bytes())
        entry.update(status="downloaded", sha256=sha256(target), size_bytes=target.stat().st_size, path=str(target))
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest
        entry.update(status="not_found", reason=f"{type(exc).__name__}: {exc}")
    return entry


def fetch(root: Path = DATA_ROOT, url: str = ZIP_URL, names: tuple[str, ...] = NAMES) -> dict:
    """Download/extract matching ontology members and update the manifest.

    Parameters
    ----------
    root : pathlib.Path
        Data directory.
    url : str
        Ontolearn ``KGs.zip`` URL.
    names : tuple of str
        KBs to (re)fetch; entries of other KBs in an existing manifest are kept.

    Returns
    -------
    dict
        The manifest.
    """
    root = root.expanduser().resolve()
    archive_dir = root / "kbs" / "_downloads" / "ontolearn-kgs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / "KGs.zip"
    manifest_path = root / "kbs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["source"] = url
    manifest.setdefault("datasets", {})
    manifest.pop("download_error", None)
    try:
        if not archive.exists():
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith((".owl", ".rdf", ".xml"))]
            for name in names:
                aliases = ALIASES[name]
                matches = [m for m in members if any(a in Path(m.filename).name.lower() for a in aliases)]
                # Prefer exact basename matches; family-benchmark is an intentional alias.
                matches.sort(key=lambda m: (0 if Path(m.filename).stem.lower() == name else 1, len(m.filename)))
                if not matches:
                    if name in ZIP_MEMBER_SOURCES:
                        manifest["datasets"][name] = fetch_zip_member(name, root)
                        continue
                    direct_url = DIRECT_URLS.get(name)
                    if direct_url:
                        staging = root / "kbs" / "_downloads" / name
                        staging.mkdir(parents=True, exist_ok=True)
                        staged = staging / "source.owl"
                        try:
                            request = urllib.request.Request(direct_url, headers={"User-Agent": USER_AGENT})
                            with urllib.request.urlopen(request, timeout=60) as response, staged.open("wb") as out:
                                while chunk := response.read(1024 * 1024): out.write(chunk)
                            target = root / "kbs" / name / f"{name}.owl"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(staged.read_bytes())
                            manifest["datasets"][name] = {"status": "downloaded", "url": direct_url, "sha256": sha256(target), "size_bytes": target.stat().st_size, "path": str(target)}
                            continue
                        except Exception as exc:
                            reason = f"direct source failed: {type(exc).__name__}: {exc}"
                    else:
                        reason = "No matching OWL/RDF/XML member in KGs.zip; no direct source configured"
                    manifest["datasets"][name] = {"status": "not_found", "url": direct_url or url, "reason": reason}
                    continue
                member = matches[0]
                suffix = Path(member.filename).suffix.lower()
                target = root / "kbs" / name / f"{name}.owl"
                target.parent.mkdir(parents=True, exist_ok=True)
                # Read only the matched member and write it in its dataset-specific directory.
                with zf.open(member) as src, target.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                manifest["datasets"][name] = {"status": "downloaded", "url": url, "archive_member": member.filename, "sha256": sha256(target), "size_bytes": target.stat().st_size, "original_suffix": suffix, "path": str(target)}
    except Exception as exc:  # keep a usable manifest for partial failures
        manifest["download_error"] = f"{type(exc).__name__}: {exc}"
        for name in names:
            manifest["datasets"].setdefault(name, {"status": "not_found", "url": url, "reason": manifest["download_error"]})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    """Run the fetcher CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--url", default=ZIP_URL)
    parser.add_argument("--kb", choices=NAMES, action="append", help="fetch only these KBs (repeatable); default all")
    args = parser.parse_args()
    names = tuple(args.kb) if args.kb else NAMES
    result = fetch(args.data_dir, args.url, names)
    for name in names:
        item = result["datasets"][name]
        print(f"{name}: {item['status']}" + (f" ({item.get('size_bytes', 0)} bytes, sha256 {item.get('sha256')})" if item["status"] == "downloaded" else f" — {item.get('reason', '')}"))

if __name__ == "__main__":
    main()
