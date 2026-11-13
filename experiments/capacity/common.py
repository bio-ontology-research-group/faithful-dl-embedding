"""Shared helpers for the capacity experiments (B1-B4).

Purpose
-------
Load every dataset of the capacity study under one name space, build the indexed
normalised EL++ tables exactly as ``GroundedEL(dataset, embed_dim=1,
n_components=1, n_role_components=1)`` builds them, compute the reference named
subsumptions from :class:`mowl.geometric.completion.CanonicalModel` (through the
pickle cache of :mod:`experiments.cm_cache`), and score predicted subsumption
matrices against it exactly and sparsely.

Inputs
------
Dataset identifiers (see :data:`CAPACITY_DATASETS`):

- the synthetic theories of :mod:`experiments.grounded.theories` (``THEORIES``);
- ``pizza``, ``go``, ``foodon`` through :func:`experiments.datasets.load_dataset`;
- ``galen_j24``, ``go_j24``, ``anatomy_j24``: the Box²EL benchmark ontologies through
  mOWL's ``*Jackermeier2024Dataset`` classes (training ontology only), data root
  ``$MOWL_DATA_ROOT`` (default ``~/tmp/gfe-data/mowl``).

Outputs
-------
Python objects only (:class:`KBTables`, :class:`Reference`, metric dictionaries);
:func:`write_json` writes result files atomically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse

from experiments.config import MOWL_DATA_ROOT, RESULTS_DIR
from experiments.grounded.theories import THEORIES

CAPACITY_RAW_DIR: Path = RESULTS_DIR / "raw-capacity"

HARNESS_DATASETS = ("pizza", "go", "foodon")
J24_DATASETS: Dict[str, str] = {
    "galen_j24": "GALENJackermeier2024Dataset",
    "go_j24": "GOJackermeier2024Dataset",
    "anatomy_j24": "AnatomyJackermeier2024Dataset",
}
CAPACITY_DATASETS = tuple(THEORIES) + HARNESS_DATASETS + tuple(J24_DATASETS)
# The Box²EL benchmark files are semantically corrupt in the upstream release: the helper
# classes of the normalisation share IRIs with real classes (mOWL issue #178). They stay
# loadable but are excluded from every default and from the aggregated tables.
CORRUPT_DATASETS = tuple(J24_DATASETS)
DEFAULT_DATASETS = tuple(d for d in CAPACITY_DATASETS if d not in CORRUPT_DATASETS)

TABLE_KEYS = (
    "gci0",
    "gci1",
    "gci2",
    "gci3",
    "gci0_bot",
    "gci1_bot",
    "gci3_bot",
    "role_inclusion",
    "role_chain",
    "class_assertion",
    "object_property_assertion",
)


def peak_rss_gb() -> float:
    """Peak resident set size of this process (JVM included) in GB.

    Returns
    -------
    float
        ``ru_maxrss`` of ``RUSAGE_SELF`` converted from KiB to GB.
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0 / 1e9


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples into JSON-serialisable values.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        JSON-serialisable object.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(payload: Dict[str, Any], path: Path) -> Path:
    """Atomically write ``payload`` as indented JSON to ``path``.

    Parameters
    ----------
    payload : dict
        Result dictionary (numpy values allowed).
    path : Path
        Destination file; parent directories are created.

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path)
    if str(path.resolve()).startswith("/tmp/"):
        raise ValueError(f"refusing to write under /tmp: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_capacity_dataset(name: str):
    """Load dataset ``name`` as a :class:`experiments.datasets.DatasetBundle`.

    Parameters
    ----------
    name : str
        One of :data:`CAPACITY_DATASETS`.

    Returns
    -------
    experiments.datasets.DatasetBundle
        Bundle whose ``dataset`` is a mOWL dataset and whose ``paths["ontology"]``
        is the training ontology (used as the CanonicalModel cache key).
    """
    import mowl

    from experiments.datasets import DatasetBundle, load_dataset

    if name not in CAPACITY_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {CAPACITY_DATASETS}")
    if name in HARNESS_DATASETS:
        return load_dataset(name)
    mowl.init_jvm("8g")
    if name in THEORIES:
        from experiments.grounded.theories import load_theory_dataset

        ds = load_theory_dataset(name)
        return DatasetBundle(
            name=name,
            dataset=ds,
            deductive_closure_ontology=None,
            task="satisfaction",
            paths={"ontology": str(Path(ds.ontology_path).resolve())},
        )
    import mowl.datasets.builtin as builtin

    cls = getattr(builtin, J24_DATASETS[name])
    ds = cls(data_root=str(Path(MOWL_DATA_ROOT).expanduser().resolve()))
    return DatasetBundle(
        name=name,
        dataset=ds,
        deductive_closure_ontology=None,
        task="subsumption",
        paths={
            "ontology": str(Path(ds.ontology_path).resolve()),
            "valid": str(Path(ds.validation_path).resolve()),
            "test": str(Path(ds.testing_path).resolve()),
        },
    )


@dataclass
class KBTables:
    """Indexed normalised EL++ tables of one ontology (the input of ``least_model``).

    Attributes
    ----------
    nb_classes, nb_rels, nb_individuals : int
        Sizes of the class, role and individual index dictionaries.
    tables : dict
        ``{name: (m, w) int64 array}`` for every key of :data:`TABLE_KEYS`.
    top_id, bot_id : int or None
        Indices of ``owl:Thing`` / ``owl:Nothing``.
    inverse_of : list or None
        Inverse-role partner list.
    class_iris : list of str
        IRI of every class index.
    named_ids : numpy.ndarray
        Sorted indices of named classes (no TOP/BOTTOM/normalisation auxiliaries).
    build_time_s : float
        Wall time of the table construction.
    """

    nb_classes: int
    nb_rels: int
    nb_individuals: int
    tables: Dict[str, np.ndarray]
    top_id: Optional[int]
    bot_id: Optional[int]
    inverse_of: Optional[List[int]]
    class_iris: List[str]
    named_ids: np.ndarray
    build_time_s: float = 0.0
    model: Any = field(default=None, repr=False)

    def least_model(self, **kwargs):
        """Run :func:`mowl.geometric.least_model` on these tables.

        Parameters
        ----------
        **kwargs
            Forwarded (``method``, ``extra_facts``, ``objective_weights``, ...).

        Returns
        -------
        mowl.geometric.least_model.LeastModel
        """
        from mowl.geometric.least_model import least_model

        return least_model(
            self.nb_classes,
            self.nb_rels,
            self.nb_individuals,
            top_id=self.top_id,
            bot_id=self.bot_id,
            inverse_of=self.inverse_of,
            **self.tables,
            **kwargs,
        )

    def check(self, lm) -> Dict[str, int]:
        """Count violated ground axiom instances of ``lm`` (``mowl.geometric.check_model``).

        Parameters
        ----------
        lm : LeastModel

        Returns
        -------
        dict
            Violation counts per table; all zeros iff ``lm`` is a model.
        """
        from mowl.geometric.least_model import check_model

        return {k: int(v) for k, v in check_model(lm, **self.tables).items()}

    def sizes(self) -> Dict[str, Any]:
        """Return the size summary recorded in the result files."""
        return {
            "n_classes": int(self.nb_classes),
            "n_named_classes": int(len(self.named_ids)),
            "n_aux_classes": int(
                self.nb_classes
                - len(self.named_ids)
                - (self.top_id is not None)
                - (self.bot_id is not None)
            ),
            "n_roles": int(self.nb_rels),
            "n_individuals": int(self.nb_individuals),
            "rows": {k: int(len(v)) for k, v in self.tables.items()},
        }


def tables_from_arrays(
    nb_classes: int,
    nb_rels: int = 0,
    nb_individuals: int = 0,
    *,
    top_id: Optional[int] = None,
    bot_id: Optional[int] = None,
    named_ids: Optional[Sequence[int]] = None,
    **tables,
) -> KBTables:
    """Build :class:`KBTables` from raw integer arrays (tests, hand-built theories).

    Parameters
    ----------
    nb_classes, nb_rels, nb_individuals : int
        Index sizes.
    top_id, bot_id : int, optional
        TOP / BOTTOM class ids.
    named_ids : sequence of int, optional
        Named classes (default: every class except TOP and BOTTOM).
    **tables
        Any of :data:`TABLE_KEYS` as ``(m, w)`` integer arrays.

    Returns
    -------
    KBTables
    """
    from mowl.geometric.least_model import _TABLE_WIDTHS

    t = {}
    for k in TABLE_KEYS:
        v = tables.get(k)
        t[k] = (
            np.empty((0, _TABLE_WIDTHS[k]), dtype=np.int64)
            if v is None or len(v) == 0
            else np.asarray(v, dtype=np.int64).reshape(-1, _TABLE_WIDTHS[k])
        )
    if named_ids is None:
        named_ids = [c for c in range(nb_classes) if c not in (top_id, bot_id)]
    return KBTables(
        int(nb_classes),
        int(nb_rels),
        int(nb_individuals),
        t,
        top_id,
        bot_id,
        None,
        [f"c{i}" for i in range(nb_classes)],
        np.asarray(sorted(named_ids), dtype=np.int64),
    )


def build_tables(bundle, keep_model: bool = False) -> KBTables:
    """Build the indexed tables exactly as ``GroundedEL(dataset, 1, 1, 1)`` does.

    Parameters
    ----------
    bundle : DatasetBundle
        Loaded dataset.
    keep_model : bool, default False
        Keep the constructed ``GroundedEL`` on the returned object.

    Returns
    -------
    KBTables
        Tables, index sizes and named-class indices (named classes are selected
        with ``experiments.metrics._named_class_indices``, i.e. the dataset's
        classes minus TOP/BOTTOM and ``http://mowl.borg/aux`` IRIs).
    """
    import mowl

    mowl.init_jvm("8g")
    from mowl.models import GroundedEL

    from experiments.metrics import _named_class_indices

    t0 = time.perf_counter()
    model = GroundedEL(bundle.dataset, embed_dim=1, n_components=1, n_role_components=1)
    td = model.training_datasets
    tables = {}
    for k in TABLE_KEYS:
        if k in td and td[k] is not None:
            v = getattr(td[k], "data", td[k])
            tables[k] = np.asarray(v.detach().cpu().numpy(), dtype=np.int64)
    cls_iris = [""] * int(model.nb_classes)
    for iri, idx in model.class_index_dict.items():
        cls_iris[int(idx)] = str(iri)
    named = np.asarray(_named_class_indices(model, int(model.nb_classes)), dtype=np.int64)
    kb = KBTables(
        int(model.nb_classes),
        int(model.nb_rels),
        int(model.nb_inds),
        {k: tables.get(k, _empty(k)) for k in TABLE_KEYS},
        model.top_id,
        model.bot_id,
        model.inverse_of,
        cls_iris,
        named,
        time.perf_counter() - t0,
        model if keep_model else None,
    )
    return kb


def _empty(key: str) -> np.ndarray:
    from mowl.geometric.least_model import _TABLE_WIDTHS

    return np.empty((0, _TABLE_WIDTHS[key]), dtype=np.int64)


@dataclass
class Reference:
    """Reference named-class entailments from the CanonicalModel.

    Attributes
    ----------
    iris : list of str
        Named class IRIs (order of the caller's ``named_ids``).
    satisfiable : numpy.ndarray
        Boolean mask, ``True`` iff the reference finds the class satisfiable.
    matrix : scipy.sparse.csr_matrix
        Boolean ``(n, n)``; ``(i, j)`` iff ``i != j``, both satisfiable and
        ``A_i ⊑ A_j`` is entailed.
    source : str
        ``"cm_cache"`` or ``"constructed"``.
    time_s : float
        Wall time to obtain the reference.
    """

    iris: List[str]
    satisfiable: np.ndarray
    matrix: sparse.csr_matrix
    source: str
    time_s: float

    @property
    def n_entailed(self) -> int:
        """Number of entailed pairs ``A ⊑ B`` (``A != B``, both satisfiable)."""
        return int(self.matrix.nnz)


def reference_entailments(bundle, iris: Sequence[str]) -> Reference:
    """Compute exact reference subsumptions among ``iris`` from the CanonicalModel.

    Parameters
    ----------
    bundle : DatasetBundle
        Dataset; the CanonicalModel of its training ontology is loaded from (or
        built into) the cache of :func:`experiments.cm_cache.get_canonical_model`.
    iris : sequence of str
        Named class IRIs.

    Returns
    -------
    Reference

    Notes
    -----
    Uses the saturation sets ``S(A)`` of the completion (``cm.completion.sub_super``),
    so the computation is sparse and exact; no ``n x n`` dense array is built.
    Classes absent from the training signature have ``S(A) = {A, TOP}``.
    """
    from experiments.cm_cache import get_canonical_model

    t0 = time.perf_counter()
    source = "cm_cache"
    try:
        cm = get_canonical_model(bundle)
    except (ValueError, FileNotFoundError):
        from mowl.geometric.completion import CanonicalModel

        cm = CanonicalModel.from_dataset(bundle.dataset)
        source = "constructed"
    sub_super = cm.completion.sub_super
    iris = [str(i) for i in iris]
    pos = {iri: j for j, iri in enumerate(iris)}
    sat = np.array([bool(cm.is_satisfiable(iri)) for iri in iris], dtype=bool)
    rows: List[int] = []
    cols: List[int] = []
    for j, iri in enumerate(iris):
        if not sat[j]:
            continue
        for s in sub_super.get(iri, ()):
            k = pos.get(s)
            if k is not None and k != j and sat[k]:
                rows.append(j)
                cols.append(k)
    n = len(iris)
    mat = bool_csr(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), (n, n))
    return Reference(iris, sat, mat, source, time.perf_counter() - t0)


def bool_csr(rows: np.ndarray, cols: np.ndarray, shape) -> sparse.csr_matrix:
    """Boolean CSR matrix with ones at ``(rows, cols)`` (duplicates collapsed).

    Parameters
    ----------
    rows, cols : numpy.ndarray
        Coordinates.
    shape : tuple of int
        Matrix shape.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    m = sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (np.asarray(rows), np.asarray(cols))), shape=shape
    ).tocsr()
    m.sum_duplicates()
    m.data[:] = True
    return m


def strip_diagonal(m: sparse.spmatrix) -> sparse.csr_matrix:
    """Return ``m`` as boolean CSR without diagonal entries."""
    coo = sparse.coo_matrix(m)
    keep = (coo.row != coo.col) & (coo.data != 0)
    return bool_csr(coo.row[keep], coo.col[keep], m.shape)


def restrict(m: sparse.csr_matrix, idx: np.ndarray) -> sparse.csr_matrix:
    """Principal submatrix ``m[idx][:, idx]`` as boolean CSR."""
    idx = np.asarray(idx, dtype=np.int64)
    return sparse.csr_matrix(m)[idx][:, idx].tocsr()


def pair_metrics(
    pred: sparse.csr_matrix,
    ref: sparse.csr_matrix,
    unsat_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Exact precision/recall of predicted subsumption pairs against a reference.

    Both matrices are over the same ``n`` classes; pairs are ordered ``(A, B)``
    with ``A != B`` (diagonals are ignored).

    Parameters
    ----------
    pred : scipy.sparse matrix
        Predicted holding subsumptions.
    ref : scipy.sparse matrix
        Reference entailments.
    unsat_pred : numpy.ndarray of bool, optional
        Rows predicted unsatisfiable: such a class counts as subsumed by every
        other class (its row in ``pred`` is replaced by all ones).

    Returns
    -------
    dict
        ``tp``, ``fp``, ``fn``, ``n_pred``, ``n_ref``, ``precision``, ``recall``
        (precision/recall are 1.0 when their denominator is zero).
    """
    n = int(ref.shape[0])
    p = strip_diagonal(pred)
    r = strip_diagonal(ref)
    ref_rows = np.asarray(r.getnnz(axis=1)).ravel()
    extra_tp = 0
    extra_pred = 0
    if unsat_pred is not None and np.any(unsat_pred):
        u = np.asarray(unsat_pred, dtype=bool)
        p = (sparse.diags((~u).astype(np.int8)) @ p).astype(bool).tocsr()
        p.eliminate_zeros()
        extra_tp = int(ref_rows[u].sum())
        extra_pred = int(u.sum()) * (n - 1)
    tp = int(p.multiply(r).nnz) + extra_tp
    n_pred = int(p.nnz) + extra_pred
    n_ref = int(r.nnz)
    return {
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_ref - tp,
        "n_pred": n_pred,
        "n_ref": n_ref,
        "precision": float(tp / n_pred) if n_pred else 1.0,
        "recall": float(tp / n_ref) if n_ref else 1.0,
    }


def runs_summary(runs: np.ndarray) -> Dict[str, Any]:
    """Summary statistics of runs (intervals) per class, over classes with >= 1 run.

    Parameters
    ----------
    runs : numpy.ndarray
        Runs per class (0 for unsatisfiable / bottom classes).

    Returns
    -------
    dict
        ``n``, ``max``, ``mean``, ``median``, ``p95``, ``share_one``, ``total``,
        ``hist`` (exact counts for 1..8, then power-of-two buckets ``"9-16"``, ...).
    """
    r = np.asarray(runs, dtype=np.int64)
    r = r[r > 0]
    if len(r) == 0:
        return {"n": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "share_one": 1.0,
                "total": 0, "hist": {}}
    hist: Dict[str, int] = {}
    for k in range(1, 9):
        c = int((r == k).sum())
        if c:
            hist[str(k)] = c
    lo = 9
    while lo <= r.max():
        hi = 2 * (lo - 1)
        c = int(((r >= lo) & (r <= hi)).sum())
        if c:
            hist[f"{lo}-{hi}"] = c
        lo = hi + 1
    return {
        "n": int(len(r)),
        "max": int(r.max()),
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "p95": float(np.percentile(r, 95)),
        "share_one": float((r == 1).mean()),
        "total": int(r.sum()),
        "hist": hist,
    }


def intervals_membership(lm, real) -> sparse.csr_matrix:
    """Rebuild the ``(n_blocks, nb_classes)`` membership matrix from an interval realisation.

    Parameters
    ----------
    lm : LeastModel
    real : IntervalRealisation

    Returns
    -------
    scipy.sparse.csr_matrix
        Boolean membership read off the intervals (positions mapped back to blocks).
    """
    order = np.asarray(real.order, dtype=np.int64)
    parts_c, parts_b = [], []
    for c, iv in real.intervals.items():
        if len(iv) == 0:
            continue
        lengths = iv[:, 1] - iv[:, 0] + 1
        starts = np.repeat(iv[:, 0], lengths)
        offs = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        parts_b.append(order[starts + offs])
        parts_c.append(np.full(int(lengths.sum()), c, dtype=np.int64))
    if not parts_b:
        return sparse.csr_matrix((lm.n_blocks, lm.nb_classes), dtype=bool)
    return bool_csr(np.concatenate(parts_b), np.concatenate(parts_c), (lm.n_blocks, lm.nb_classes))


def membership_equals_labels(lm, real) -> bool:
    """``True`` iff membership from the intervals equals ``lm.labels`` on live blocks."""
    m = intervals_membership(lm, real)
    live = (~lm.dead).astype(np.int8)
    lab = (sparse.diags(live) @ lm.labels).astype(bool).tocsr()
    lab.eliminate_zeros()
    diff = (m != lab)
    return int(diff.nnz) == 0
