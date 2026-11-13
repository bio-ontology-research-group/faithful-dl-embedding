"""B2 (total budget): sound lossy compression of the least model under a TOTAL interval budget.

Purpose
-------
The per-class budget of :mod:`experiments.capacity.budget` ("at most K intervals for
every class") collapses on real ontologies: a few cross-cutting classes need thousands
of intervals and forcing each of them under a uniform cap fills huge gaps. Here the
budget is the total number of intervals over all classes and the globally smallest
gaps are filled first::

    lm = exact least model; order fixed (dfs | identity | best-of-N)
    T_exact = total runs; n_live = classes with >= 1 run
    for f in fractions (descending):
        T = n_live + round(f * (T_exact - n_live))      # f = 1 exact, f = 0 one interval per class
        repeat:
            real = interval_realisation(lm, order restricted to live blocks)
            excess = total runs - T; if excess <= 0: stop
            gaps = all gaps of all classes (size = live blocks strictly between two runs)
            drop ineligible gaps (``--protect disjoint``)
            fill the ``excess`` smallest gaps by (size, class id, left position)
            lm = least_model(extra_facts = all facts so far)     # re-closure

Fractions run in descending order and every fraction starts from the facts of the
previous one, so the fact sets (and hence the models) are nested: facts only grow.

``--protect disjoint``: a gap of class ``A`` is ineligible if filling it kills one of its
blocks at once. ``U(A)`` is the label set of ``A``'s own block in the current model,
``D(A)`` the classes ``D`` with a disjointness row ``C ⊓ D ⊑ ⊥`` (table ``gci1_bot``, both
orders) for some ``C`` in ``U(A)``; the gap is ineligible if one of its blocks, or a block
labelled with the class of one of its blocks (a subclass or instance, which inherits the
new label), has a label in ``D(A)``. This is a one-step test (it does not foresee deaths that need a conjunction
or an existential rule), so some classes may still become unsatisfiable. With protection
the target may be unreachable; the achieved total is recorded.

Inputs
------
``--dataset`` (ids of :data:`experiments.capacity.common.CAPACITY_DATASETS` or ``all`` =
:data:`~experiments.capacity.common.DEFAULT_DATASETS`), ``--fractions``, ``--protect``
(``none`` | ``disjoint`` | ``both``), ``--order``, ``--seed``, ``--time-limit``,
``--skip-existing``.

Outputs
-------
One JSON per (dataset, fraction, protect):
``<out>/<dataset>__f<fraction>__<protect>[__<order>].json`` (default
``experiments/results/raw-capacity/budget-total/``). If ``check_model`` is not all
zeros the facts are saved next to it as ``.facts.npy``.

Usage
-----
``python -m experiments.capacity.budget_total --dataset pizza --protect both``
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from experiments.capacity.budget import budget_metrics, choose_order
from experiments.capacity.common import (
    CAPACITY_DATASETS,
    CAPACITY_RAW_DIR,
    DEFAULT_DATASETS,
    KBTables,
    Reference,
    build_tables,
    load_capacity_dataset,
    peak_rss_gb,
    reference_entailments,
    write_json,
)

DEFAULT_OUT = CAPACITY_RAW_DIR / "budget-total"
FRACTIONS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0)
PROTECT_MODES = ("none", "disjoint")
COSTS = ("size", "weight")


def block_weights(lm) -> np.ndarray:
    """Number of blocks that inherit a new label of block ``b``.

    A fact ``(b, A)`` reaches every block labelled with the class of ``b`` (its
    subclasses and instances), so the weight of a class block is the size of its
    class's column (the largest one if several classes share the block); other
    blocks have weight 1.
    """
    col = np.asarray(lm.labels.tocsc().getnnz(axis=0)).ravel().astype(np.int64)
    w = np.ones(lm.n_blocks, dtype=np.int64)
    cls = np.arange(lm.nb_classes)
    if lm.top_id is not None:
        cls = cls[cls != lm.top_id]
    np.maximum.at(w, lm.element_map[cls], col[cls])
    return w


def all_gaps(real) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All gaps between consecutive runs of every class.

    Parameters
    ----------
    real : IntervalRealisation
        ``intervals[a]`` is a ``(k, 2)`` array of inclusive position ranges.

    Returns
    -------
    cls, lo, hi : numpy.ndarray
        Class of every gap and its position range ``[lo, hi)`` (never empty).
    """
    multi = [(a, iv) for a, iv in real.intervals.items() if len(iv) > 1]
    if not multi:
        e = np.empty(0, dtype=np.int64)
        return e, e.copy(), e.copy()
    cls = np.concatenate([np.full(len(iv) - 1, int(a), dtype=np.int64) for a, iv in multi])
    lo = np.concatenate([np.asarray(iv[:-1, 1], dtype=np.int64) + 1 for _, iv in multi])
    hi = np.concatenate([np.asarray(iv[1:, 0], dtype=np.int64) for _, iv in multi])
    return cls, lo, hi


def disjoint_matrix(kb: KBTables) -> sparse.csr_matrix:
    """Symmetric class × class matrix of the asserted disjointness rows ``C ⊓ D ⊑ ⊥``."""
    nc = kb.nb_classes
    g = np.asarray(kb.tables["gci1_bot"]).reshape(-1, 3)
    g1 = np.asarray(kb.tables["gci1"]).reshape(-1, 3)
    if kb.bot_id is not None and len(g1):
        g = np.vstack([g, g1[g1[:, 2] == kb.bot_id]])
    if len(g) == 0:
        return sparse.csr_matrix((nc, nc), dtype=np.int32)
    r = np.concatenate([g[:, 0], g[:, 1]])
    c = np.concatenate([g[:, 1], g[:, 0]])
    m = sparse.csr_matrix((np.ones(len(r), dtype=np.int32), (r, c)), shape=(nc, nc))
    m.data[:] = 1
    return m


def eligible_mask(lm, dis: sparse.csr_matrix, order_r: np.ndarray, cls: np.ndarray,
                  lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """One-step death test of every gap (see the module docstring).

    Parameters
    ----------
    lm : LeastModel
        Current model.
    dis : scipy.sparse.csr_matrix
        :func:`disjoint_matrix`.
    order_r : numpy.ndarray
        Live blocks in position order.
    cls, lo, hi : numpy.ndarray
        Gaps as returned by :func:`all_gaps`.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` for gaps whose filling kills no block immediately.
    """
    ok = np.ones(len(cls), dtype=bool)
    if dis.nnz == 0 or len(cls) == 0:
        return ok
    L = lm.labels.tocsr().astype(np.int32)
    Lc = L.tocsc()
    ucls, inv = np.unique(cls, return_inverse=True)
    # D(A) for every class with a gap: rows of  L[block(A)] @ dis.
    DA = (L[lm.element_map[ucls]] @ dis).tocsr()
    DA.sort_indices()
    cache: Dict[bytes, Optional[np.ndarray]] = {}
    cums: List[Optional[np.ndarray]] = []
    for i in range(len(ucls)):
        d = DA.indices[DA.indptr[i]:DA.indptr[i + 1]]
        key = d.tobytes()
        if key not in cache:
            if len(d) == 0:
                cache[key] = None
            else:
                bad_block = np.zeros(lm.n_blocks, dtype=bool)
                sub = Lc[:, d]
                bad_block[np.unique(sub.indices)] = True
                # a fact (b, A) also reaches every block labelled with b's class
                # (its subclasses and instances): b is bad if one of those is.
                bad_cls = np.asarray(Lc.T @ bad_block.astype(np.int32)).ravel() > 0
                bad_block[lm.element_map[np.flatnonzero(bad_cls)]] = True
                bad_pos = bad_block[order_r]
                cache[key] = (np.concatenate(([0], np.cumsum(bad_pos))) if bad_pos.any() else None)
        cums.append(cache[key])
    # evaluate per distinct cumulative array
    by_id: Dict[int, List[int]] = {}
    for i, cu in enumerate(cums):
        if cu is not None:
            by_id.setdefault(id(cu), []).append(i)
    lookup = {id(cu): cu for cu in cums if cu is not None}
    for k, members in by_id.items():
        cu = lookup[k]
        sel = np.flatnonzero(np.isin(inv, np.asarray(members)))
        ok[sel] = (cu[hi[sel]] - cu[lo[sel]]) == 0
    return ok


def compress_total(
    kb: KBTables,
    lm_exact,
    order: np.ndarray,
    fractions: Sequence[float] = FRACTIONS,
    protect: str = "disjoint",
    max_iter: int = 200,
    deadline: Optional[float] = None,
    verbose: bool = False,
    cost: str = "size",
) -> Iterator[Dict[str, Any]]:
    """Yield the compressed model of every fraction (descending, warm-started).

    Parameters
    ----------
    kb : KBTables
    lm_exact : LeastModel
    order : numpy.ndarray
        Fixed order of the live blocks of ``lm_exact``.
    fractions : sequence of float
        Fractions in ``[0, 1]`` of the exact excess ``T_exact - n_live`` to keep; they are
        processed in descending order.
    protect : str
        ``"none"`` or ``"disjoint"``.
    max_iter : int
        Cap on fill/re-closure rounds per fraction.
    deadline : float, optional
        ``time.perf_counter()`` value after which the run stops between rounds.
    cost : str
        Gap cost that decides the filling order: ``"size"`` (number of blocks in the gap)
        or ``"weight"`` (sum of :func:`block_weights` over the gap, i.e. an estimate of
        the number of labels the filling adds once subclasses inherit them).

    Yields
    ------
    dict
        ``fraction``, ``target_total``, ``lm``, ``real``, ``facts`` (``(m, 2)`` array, all
        facts so far), ``iterations``, ``n_facts``, ``status``, ``iteration_log`` and the
        gap statistics of this fraction.
    """
    from mowl.geometric.least_model import interval_realisation

    if protect not in PROTECT_MODES:
        raise ValueError(f"protect must be one of {PROTECT_MODES}")
    dis = disjoint_matrix(kb) if protect == "disjoint" else None
    real0 = interval_realisation(lm_exact, order[~lm_exact.dead[order]])
    t_exact = int(real0.runs_per_class.sum())
    n_live = int((real0.runs_per_class > 0).sum())
    fb: List[np.ndarray] = []
    fc: List[np.ndarray] = []
    lm = lm_exact
    for f in sorted({float(x) for x in fractions}, reverse=True):
        if not 0.0 <= f <= 1.0:
            raise ValueError("fractions must lie in [0, 1]")
        target = n_live + int(round(f * (t_exact - n_live)))
        it = 0
        status = "ok"
        log: List[Dict[str, Any]] = []
        sizes_filled: List[np.ndarray] = []
        n_inelig = 0
        while True:
            order_r = order[~lm.dead[order]]
            real = interval_realisation(lm, order_r)
            excess = int(real.runs_per_class.sum()) - target
            if excess <= 0:
                break
            if it >= max_iter:
                status = f"aborted: max_iter {max_iter}"
                break
            if deadline is not None and time.perf_counter() > deadline:
                status = "aborted: time limit"
                break
            cls, lo, hi = all_gaps(real)
            if dis is not None:
                ok = eligible_mask(lm, dis, order_r, cls, lo, hi)
                n_inelig = int((~ok).sum())
                cls, lo, hi = cls[ok], lo[ok], hi[ok]
            if len(cls) == 0:
                status = "target not reachable (protected)"
                break
            size = hi - lo
            if cost == "weight":
                cw = np.concatenate(([0], np.cumsum(block_weights(lm)[order_r])))
                gap_cost = cw[hi] - cw[lo]
            elif cost == "size":
                gap_cost = size
            else:
                raise ValueError(f"cost must be one of {COSTS}")
            sel = np.lexsort((lo, cls, gap_cost))[:excess]
            lengths = size[sel]
            tot = int(lengths.sum())
            pos = np.repeat(lo[sel], lengths) + (
                np.arange(tot) - np.repeat(np.cumsum(lengths) - lengths, lengths))
            fb.append(order_r[pos])
            fc.append(np.repeat(cls[sel], lengths))
            sizes_filled.append(lengths)
            it += 1
            t0 = time.perf_counter()
            facts = np.stack([np.concatenate(fb), np.concatenate(fc)], axis=1)
            try:
                lm = kb.least_model(method="iteration", extra_facts=facts)
            except MemoryError:
                # the re-closure exploded (labels of a general class spread everywhere):
                # report the last model that could be computed and stop the curve
                status = "aborted: out of memory in re-closure"
                fb.pop(), fc.pop(), sizes_filled.pop()
                break
            log.append({
                "iteration": it,
                "excess": excess,
                "gaps_filled": int(len(sel)),
                "new_facts": tot,
                "least_model_time_s": time.perf_counter() - t0,
                "n_dead_blocks": int(lm.dead.sum()),
            })
            if verbose:
                e = log[-1]
                print(f"[budget-total]   f={f:g} it={it}: excess={excess} filled={e['gaps_filled']} "
                      f"new_facts={tot} dead={e['n_dead_blocks']} lm={e['least_model_time_s']:.1f}s",
                      flush=True)
        filled = np.concatenate(sizes_filled) if sizes_filled else np.empty(0, dtype=np.int64)
        facts = (np.stack([np.concatenate(fb), np.concatenate(fc)], axis=1)
                 if fb else np.empty((0, 2), dtype=np.int64))
        yield {
            "fraction": f,
            "target_total": target,
            "exact_total": t_exact,
            "n_live_classes": n_live,
            "lm": lm,
            "real": real,
            "facts": facts,
            "iterations": it,
            "n_facts": int(len(facts)),
            "status": status,
            "iteration_log": log,
            "n_gaps_filled": int(len(filled)),
            "n_gaps_ineligible": n_inelig,
            "gap_size_filled_max": int(filled.max()) if len(filled) else 0,
            "gap_size_filled_mean": float(filled.mean()) if len(filled) else 0.0,
        }
        if status.startswith("aborted"):
            return


def out_name(dataset: str, fraction: float, protect: str, order: str = "dfs",
             cost: str = "size") -> str:
    """File stem of one result."""
    tag = "" if order == "dfs" else f"__{order}"
    tag += "" if cost == "size" else f"__{cost}"
    return f"{dataset}__f{fraction:g}__{protect}{tag}"


def run_budget_total(
    dataset: str,
    fractions: Sequence[float] = FRACTIONS,
    protect: str = "disjoint",
    order: str = "dfs",
    seed: int = 0,
    out_dir: Path = DEFAULT_OUT,
    time_limit: Optional[float] = None,
    kb: Optional[KBTables] = None,
    ref: Optional[Reference] = None,
    bundle: Any = None,
    write: bool = True,
    skip_existing: bool = False,
    keep_facts: bool = False,
    cost: str = "size",
) -> List[Dict[str, Any]]:
    """Run the total-budget curve of one dataset and one protect setting.

    Parameters
    ----------
    dataset : str
    fractions : sequence of float
    protect : str
        ``"none"`` or ``"disjoint"``.
    order, seed : see :func:`experiments.capacity.budget.choose_order`.
    out_dir : Path
    time_limit : float, optional
        Seconds for the whole curve.
    kb, ref, bundle : optional
        Pre-built tables / reference / bundle (tests, reuse across protect settings).
    write : bool
    skip_existing : bool
        Skip the whole curve when every JSON exists (the curve is warm-started, so
        single fractions cannot be skipped).
    keep_facts : bool
        Keep the ``facts`` array in the returned payloads (tests).

    Returns
    -------
    list of dict
        Payload per fraction.
    """
    out_dir = Path(out_dir)
    fr = sorted({float(x) for x in fractions}, reverse=True)
    if skip_existing and write and all(
            (out_dir / f"{out_name(dataset, f, protect, order, cost)}.json").exists() for f in fr):
        print(f"[budget-total] {dataset} protect={protect}: all results exist, skipped", flush=True)
        return []
    t_start = time.perf_counter()
    deadline = None if time_limit is None else t_start + float(time_limit)
    if kb is None:
        bundle = load_capacity_dataset(dataset)
        kb = build_tables(bundle)
    lm_exact = kb.least_model(method="iteration")
    if ref is None and bundle is not None:
        from mowl.geometric.errors import InconsistentOntology

        try:
            ref = reference_entailments(bundle, [kb.class_iris[i] for i in kb.named_ids])
        except InconsistentOntology:
            ref = None
    o, info = choose_order(lm_exact, order, seed)
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for res in compress_total(kb, lm_exact, o, fr, protect=protect, deadline=deadline,
                              verbose=write, cost=cost):
        payload: Dict[str, Any] = {
            "dataset": dataset,
            "mode": "total",
            "protect": protect,
            "cost": cost,
            "order_info": info,
            "sizes": kb.sizes(),
            "reference_available": ref is not None,
        }
        payload.update({k: v for k, v in res.items() if k not in ("lm", "real", "facts")})
        payload.update(budget_metrics(kb, lm_exact, res, ref))
        payload["achieved_total"] = payload["total_intervals"]
        payload["wall_time_s"] = time.perf_counter() - t0
        payload["peak_rss_gb"] = peak_rss_gb()
        stem = out_name(dataset, res["fraction"], protect, order, cost)
        if write:
            write_json(payload, out_dir / f"{stem}.json")
            if not payload["check_model_all_zero"]:
                np.save(out_dir / f"{stem}.facts.npy", res["facts"])
        if keep_facts:
            payload["facts"] = res["facts"]
        results.append(payload)
        print(
            f"[budget-total] {dataset} {protect} {cost} f={res['fraction']:g}: target={res['target_total']} "
            f"achieved={payload['achieved_total']} it={res['iterations']} status={res['status']} "
            f"unsat+={payload['n_named_made_unsat']} P_live={payload.get('precision_live')} "
            f"R_live={payload.get('recall_live')} fp/class={payload.get('false_pairs_per_class_live')} "
            f"infl={payload['label_inflation']:.3f} check0={payload['check_model_all_zero']} "
            f"t={payload['wall_time_s']:.1f}s",
            flush=True,
        )
        t0 = time.perf_counter()
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", nargs="+", required=True,
                   help=f"Dataset ids ({', '.join(CAPACITY_DATASETS)}) or 'all'.")
    p.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    p.add_argument("--protect", default="disjoint", choices=PROTECT_MODES + ("both",))
    p.add_argument("--cost", default="size", choices=COSTS,
                   help="Gap cost: number of blocks, or labels added (subclasses inherit).")
    p.add_argument("--order", default="dfs", help="dfs | identity | best-of-N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--time-limit", type=float, default=None,
                   help="Time limit in seconds per (dataset, protect) curve.")
    p.add_argument("--skip-existing", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    names = list(DEFAULT_DATASETS) if args.dataset == ["all"] else args.dataset
    modes = PROTECT_MODES if args.protect == "both" else (args.protect,)
    failed = []
    for name in names:
        try:
            bundle = load_capacity_dataset(name)
            kb = build_tables(bundle)
            from mowl.geometric.errors import InconsistentOntology

            try:
                ref = reference_entailments(bundle, [kb.class_iris[i] for i in kb.named_ids])
            except InconsistentOntology:
                ref = None
            for mode in modes:
                try:
                    run_budget_total(name, fractions=args.fractions, protect=mode,
                                     order=args.order, seed=args.seed, out_dir=args.out,
                                     time_limit=args.time_limit, kb=kb, ref=ref,
                                     skip_existing=args.skip_existing, cost=args.cost)
                except Exception:  # noqa: BLE001 -- continue with the other protect setting
                    traceback.print_exc()
                    failed.append(f"{name}:{mode}")
        except Exception:  # noqa: BLE001 -- report and continue with the others
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"[budget-total] FAILED: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
