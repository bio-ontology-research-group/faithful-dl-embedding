"""Read and write QTO "neural adjacency" cache files (``neural_adj/<key>_<fraction>_<thrshd>.pt``).

Format (QTO ``model.py``, ``KGReasoning.__init__``): a Python list over relations; element ``r`` is a list of
``fraction`` torch sparse COO tensors, the row chunks ``[s:t, :]`` of the ``nentity x nentity`` matrix of relation
``r`` (rows = head, columns = tail), with ``dim = nentity // fraction`` and the remainder added to the last chunk.
Values are float32 in (0, 1]; training edges are exactly 1.

Inside this package each relation is one ``scipy.sparse.csr_matrix`` (float32, canonical format).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch

PathLike = Union[str, os.PathLike]


def chunk_bounds(nentity: int, fraction: int) -> List[Tuple[int, int]]:
    """Row chunk boundaries exactly as in QTO ``model.py`` (remainder goes to the last chunk)."""
    dim = nentity // fraction
    rest = nentity - fraction * dim
    out = []
    for i in range(fraction):
        s, t = i * dim, (i + 1) * dim
        if i == fraction - 1:
            t += rest
        out.append((s, t))
    return out


def read_stats(data_dir: PathLike) -> Tuple[int, int]:
    """(nentity, nrelation) from a QTO data dir's ``stats.txt`` (parsed as QTO ``main.py`` does)."""
    with open(Path(data_dir) / "stats.txt") as fh:
        lines = fh.readlines()
    return int(lines[0].split(" ")[-1]), int(lines[1].split(" ")[-1])


def read_train(data_dir: PathLike) -> np.ndarray:
    """Training triples of a QTO data dir as an ``(n, 3)`` int64 array (head, rel, tail)."""
    arr = np.loadtxt(Path(data_dir) / "train.txt", dtype=np.int64, delimiter="\t", ndmin=2)
    return arr.reshape(-1, 3)


def _chunks_to_csr(chunks: Sequence[torch.Tensor], nentity: int) -> sp.csr_matrix:
    rows_l, cols_l, vals_l = [], [], []
    off = 0
    for ch in chunks:
        ch = ch.cpu()
        if not ch.is_coalesced():
            ch = ch.coalesce()
        idx = ch.indices().numpy()
        rows_l.append(idx[0].astype(np.int64) + off)
        cols_l.append(idx[1].astype(np.int64))
        vals_l.append(ch.values().numpy().astype(np.float32, copy=False))
        off += ch.shape[0]
    if off != nentity:
        raise ValueError(f"row chunks cover {off} rows, expected {nentity}")
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    del rows_l, cols_l, vals_l
    keep = vals != 0
    if not keep.all():
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
    # chunks are coalesced (row-major sorted, no duplicates) and consecutive -> build CSR directly
    indptr = np.zeros(nentity + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=nentity), out=indptr[1:])
    idx_dtype = np.int32 if max(len(vals), nentity) < 2**31 - 1 else np.int64
    m = sp.csr_matrix((vals, cols.astype(idx_dtype), indptr.astype(idx_dtype)), shape=(nentity, nentity))
    m.has_sorted_indices = True
    return m


def load_qto_adj(path: PathLike, nentity: int | None = None) -> List[sp.csr_matrix]:
    """Load a QTO adjacency cache as one CSR matrix per relation (frees the torch tensors as it goes)."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - very old torch
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, list) or not obj or not isinstance(obj[0], list):
        raise ValueError(f"{path}: not a QTO adjacency list")
    if nentity is None:
        nentity = int(obj[0][0].shape[1])
    mats: List[sp.csr_matrix] = []
    for r in range(len(obj)):
        mats.append(_chunks_to_csr(obj[r], nentity))
        obj[r] = None  # release memory
    return mats


def csr_to_chunks(m: sp.csr_matrix, fraction: int) -> List[torch.Tensor]:
    """Split a CSR matrix into QTO's row chunks as coalesced sparse COO float32 CPU tensors."""
    n_rows, n_cols = m.shape
    m = m.tocsr()
    if not m.has_canonical_format:
        m.sum_duplicates()
    out = []
    for s, t in chunk_bounds(n_rows, fraction):
        sub = m[s:t]
        sub.eliminate_zeros()
        rows = np.repeat(np.arange(t - s, dtype=np.int64), np.diff(sub.indptr))
        idx = torch.from_numpy(np.vstack([rows, sub.indices.astype(np.int64)]))
        val = torch.from_numpy(np.ascontiguousarray(sub.data, dtype=np.float32))
        try:
            ten = torch.sparse_coo_tensor(idx, val, size=(t - s, n_cols), is_coalesced=True)
        except TypeError:  # pragma: no cover - torch < 2.1
            ten = torch.sparse_coo_tensor(idx, val, size=(t - s, n_cols)).coalesce()
        out.append(ten)
    return out


def save_qto_adj(mats: Sequence[sp.csr_matrix], path: PathLike, fraction: int = 10) -> None:
    """Save in QTO's cache format (list over relations of ``fraction`` sparse COO row chunks, CPU tensors).

    Writes to ``<path>.partial`` and renames, so an interrupted run never leaves a truncated cache that QTO would
    silently load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = [csr_to_chunks(m, fraction) for m in mats]
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)
