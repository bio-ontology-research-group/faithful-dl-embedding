"""B2: sound lossy compression of the least model under an interval budget.

Purpose
-------
Given a budget of ``K`` intervals per class in one dimension, coarsen the exact
least model until every class is a union of at most ``K`` runs of consecutive live
blocks, staying a model of the ontology (sound) and recording what is lost
(faithfulness). Deterministic algorithm::

    facts = {}                                  # extra (block, class) facts
    lm = least_model(...); order = <order of the exact model>   # fixed for the run
    repeat:
        real = interval_realisation(lm, order restricted to live blocks)
        over = classes with runs > K
        if none: stop
        for A in over: fill its (runs - K) smallest gaps (gap size = number of live
            blocks strictly between two consecutive runs; ties -> leftmost):
            add (b, A) to facts for every block b in the gap
        lm = least_model(..., extra_facts=facts)   # re-closure; may kill blocks

Orders: ``dfs`` (``interval_realisation(lm, "dfs").order``), ``identity``, or
``best-of-N``: N DFS orders with random child order (the roots, i.e. the children of
a virtual root, are shuffled as well), keeping the one with the fewest total runs on
the exact model.

Inputs
------
``--dataset`` (identifiers of :data:`experiments.capacity.common.CAPACITY_DATASETS`),
``--budgets`` (default ``1 2 3 4 6 8 12 16 24 32 48 64``, run in the given order; budgets
above the smallest budget that is >= the exact maximum number of runs are dropped, i.e.
an ascending list stops at the first lossless K), ``--order``, ``--seed``, ``--time-limit``,
``--max-facts``, ``--skip-existing``.

Outputs
-------
One JSON per (dataset, K): ``<out>/<dataset>__K<K>.json`` (default
``experiments/results/raw-capacity/budget/``); with ``--order`` other than ``dfs`` the
order tag is appended (``<dataset>__K<K>__<order>.json``). Fields: iterations,
``check_model`` (must be zeros), max runs, total intervals, ``n_parameters``, label
inflation, named classes made unsatisfiable, precision/recall of held named
subsumptions against the reference entailments.

Usage
-----
``python -m experiments.capacity.budget --dataset pizza --order dfs``
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from experiments.capacity.common import (
    CAPACITY_DATASETS,
    DEFAULT_DATASETS,
    CAPACITY_RAW_DIR,
    KBTables,
    Reference,
    build_tables,
    load_capacity_dataset,
    pair_metrics,
    peak_rss_gb,
    reference_entailments,
    restrict,
    write_json,
)

DEFAULT_OUT = CAPACITY_RAW_DIR / "budget"
BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def dfs_order(lm, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """DFS order of live blocks along the most-specific-superclass forest.

    With ``rng=None`` this reproduces ``mowl.geometric.least_model._dfs_order``
    (children by decreasing subtree size, then id; roots in id order). With a
    generator, roots and children are visited in a random order.

    Parameters
    ----------
    lm : LeastModel
    rng : numpy.random.Generator, optional

    Returns
    -------
    numpy.ndarray
        Permutation of the live block ids.
    """
    nc = lm.nb_classes
    em = lm.element_map
    live = ~lm.dead
    L = lm.labels.tocsr()
    sizes = np.asarray(L.getnnz(axis=0)).ravel()
    parent = np.full(nc, -1, dtype=np.int64)
    is_node = np.zeros(nc, dtype=bool)
    top = -1 if lm.top_id is None else lm.top_id
    for a in range(nc):
        b = em[a]
        if a == lm.bot_id or not live[b]:
            continue
        is_node[a] = True
        if a == lm.top_id:
            continue
        cand = L.indices[L.indptr[b]:L.indptr[b + 1]]
        cand = cand[(cand != a) & (cand != top)]
        if len(cand) == 0:
            continue
        cand = cand[live[em[cand]] & (em[cand] != b)]
        cand = cand[(sizes[cand] > sizes[a]) | ((sizes[cand] == sizes[a]) & (cand > a))]
        if len(cand):
            parent[a] = cand[np.lexsort((cand, sizes[cand]))[0]]
    nodes = np.flatnonzero(is_node)
    children: List[List[int]] = [[] for _ in range(nc)]
    for a in nodes:
        if parent[a] >= 0:
            children[parent[a]].append(int(a))
    subtree = np.ones(nc, dtype=np.int64)
    for a in nodes[np.lexsort((nodes, sizes[nodes]))]:
        if parent[a] >= 0:
            subtree[parent[a]] += subtree[a]
    seen = np.zeros(lm.n_blocks, dtype=bool)
    order: List[int] = []
    roots = [int(a) for a in nodes if parent[a] < 0]
    if rng is not None:
        roots = [roots[i] for i in rng.permutation(len(roots))]
    stack = roots[::-1]
    while stack:
        a = stack.pop()
        b = int(em[a])
        if not seen[b]:
            seen[b] = True
            order.append(b)
        ch = children[a]
        if rng is None:
            ch = sorted(ch, key=lambda c: (-subtree[c], c))
        elif len(ch) > 1:
            ch = [ch[i] for i in rng.permutation(len(ch))]
        stack.extend(ch[::-1])
    rest = np.flatnonzero(live & ~seen)
    return np.concatenate([np.asarray(order, dtype=np.int64), rest.astype(np.int64)])


def choose_order(lm, order: str, seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve an order specification on the exact model.

    Parameters
    ----------
    lm : LeastModel
        Exact least model.
    order : str
        ``"dfs"``, ``"identity"`` or ``"best-of-N"``.
    seed : int
        Seed of the random DFS orders.

    Returns
    -------
    order : numpy.ndarray
        Permutation of the live blocks of ``lm``.
    info : dict
        Which order was used and, for best-of-N, the total runs of every candidate.
    """
    from mowl.geometric.least_model import interval_realisation

    if order == "dfs":
        o = interval_realisation(lm, "dfs").order
        return np.asarray(o, dtype=np.int64), {"order": "dfs"}
    if order == "identity":
        return np.flatnonzero(~lm.dead).astype(np.int64), {"order": "identity"}
    if order.startswith("best-of-"):
        n = int(order[len("best-of-"):])
        rng = np.random.default_rng(seed)
        totals: List[int] = []
        best, best_total, best_i = None, None, -1
        for i in range(n):
            o = dfs_order(lm, rng)
            tot = int(interval_realisation(lm, o).runs_per_class.sum())
            totals.append(tot)
            if best_total is None or tot < best_total:
                best, best_total, best_i = o, tot, i
        dfs_total = int(interval_realisation(lm, "dfs").runs_per_class.sum())
        return best, {
            "order": order,
            "seed": int(seed),
            "chosen_candidate": int(best_i),
            "candidate_total_runs": totals,
            "default_dfs_total_runs": dfs_total,
        }
    raise ValueError("order must be 'dfs', 'identity' or 'best-of-N'")


def gap_facts(real, over: np.ndarray, K: int, order_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Facts that fill the ``runs - K`` smallest gaps of every over-budget class.

    Parameters
    ----------
    real : IntervalRealisation
        Realisation on ``order_r`` (positions index ``order_r``).
    over : numpy.ndarray
        Classes with more than ``K`` runs.
    K : int
        Budget.
    order_r : numpy.ndarray
        Live blocks in position order.

    Returns
    -------
    blocks, classes : numpy.ndarray
        New ``(block, class)`` facts; gap blocks are non-members, so no fact is
        already true in the current model.
    """
    bl: List[np.ndarray] = []
    cl: List[np.ndarray] = []
    for a in over:
        iv = real.intervals[int(a)]
        gaps = iv[1:, 0] - iv[:-1, 1] - 1
        n_fill = len(iv) - K
        sel = np.lexsort((np.arange(len(gaps)), gaps))[:n_fill]
        lo = iv[sel, 1] + 1
        hi = iv[sel + 1, 0]  # exclusive
        lengths = hi - lo
        tot = int(lengths.sum())
        if tot == 0:
            continue
        pos = np.repeat(lo, lengths) + (np.arange(tot) - np.repeat(np.cumsum(lengths) - lengths, lengths))
        bl.append(order_r[pos])
        cl.append(np.full(tot, int(a), dtype=np.int64))
    if not bl:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(bl), np.concatenate(cl)


def compress(
    kb: KBTables,
    lm_exact,
    order: np.ndarray,
    K: int,
    max_iter: int = 10_000,
    max_facts: Optional[int] = None,
    deadline: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the gap-filling compression for budget ``K``.

    Parameters
    ----------
    kb : KBTables
    lm_exact : LeastModel
        Exact least model.
    order : numpy.ndarray
        Fixed order of the live blocks of ``lm_exact``.
    K : int
        Interval budget per class (>= 1).
    max_iter : int
        Safety cap on iterations.
    max_facts : int, optional
        Abort when the number of extra facts exceeds this (memory guard).
    deadline : float, optional
        ``time.perf_counter()`` value after which the run is aborted.

    Returns
    -------
    dict
        ``lm`` (compressed model), ``real`` (its realisation), ``iterations``,
        ``n_facts``, ``status`` (``"ok"`` or an abort reason), ``iteration_log``.
    """
    from mowl.geometric.least_model import interval_realisation

    if K < 1:
        raise ValueError("K must be >= 1")
    fb: List[np.ndarray] = []
    fc: List[np.ndarray] = []
    n_facts = 0
    lm = lm_exact
    it = 0
    log: List[Dict[str, Any]] = []
    status = "ok"
    while True:
        order_r = order[~lm.dead[order]]
        real = interval_realisation(lm, order_r)
        over = np.flatnonzero(real.runs_per_class > K)
        if len(over) == 0:
            break
        if it >= max_iter:
            status = f"aborted: max_iter {max_iter}"
            break
        if deadline is not None and time.perf_counter() > deadline:
            status = "aborted: time limit"
            break
        b, c = gap_facts(real, over, K, order_r)
        fb.append(b)
        fc.append(c)
        n_facts += len(b)
        it += 1
        if max_facts is not None and n_facts > max_facts:
            status = f"aborted: n_facts {n_facts} > {max_facts}"
            break
        t0 = time.perf_counter()
        facts = np.stack([np.concatenate(fb), np.concatenate(fc)], axis=1)
        lm = kb.least_model(method="iteration", extra_facts=facts)
        log.append({
            "iteration": it,
            "n_over": int(len(over)),
            "max_runs_before": int(real.runs_per_class.max()),
            "new_facts": int(len(b)),
            "least_model_time_s": time.perf_counter() - t0,
            "n_dead_blocks": int(lm.dead.sum()),
        })
        if verbose:
            e = log[-1]
            print(f"[budget]   K={K} it={it}: over={e['n_over']} max_runs={e['max_runs_before']} "
                  f"new_facts={e['new_facts']} total_facts={n_facts} dead={e['n_dead_blocks']} "
                  f"lm={e['least_model_time_s']:.1f}s", flush=True)
    return {"lm": lm, "real": real, "iterations": it, "n_facts": n_facts, "status": status,
            "iteration_log": log}


def budget_metrics(kb: KBTables, lm_exact, res: Dict[str, Any], ref: Optional[Reference]) -> Dict[str, Any]:
    """Metrics of one compressed model.

    Parameters
    ----------
    kb : KBTables
    lm_exact : LeastModel
    res : dict
        Output of :func:`compress`.
    ref : Reference or None
        Reference over ``kb.named_ids``.

    Returns
    -------
    dict
    """
    lm, real = res["lm"], res["real"]
    named = kb.named_ids
    alive = (~lm.dead & ~lm_exact.dead).astype(np.int8)
    d = sparse.diags(alive)
    nnz_c = int((d @ lm.labels).astype(bool).nnz)
    nnz_e = int((d @ lm_exact.labels).astype(bool).nnz)
    exact_unsat = np.isin(named, lm_exact.unsatisfiable_classes())
    comp_unsat = np.isin(named, lm.unsatisfiable_classes())
    out: Dict[str, Any] = {
        "check_model": kb.check(lm),
        "max_runs": int(real.runs_per_class.max()) if len(real.runs_per_class) else 0,
        "total_intervals": int(real.runs_per_class.sum()),
        "role_products": int(real.role_products),
        "n_parameters": int(real.n_parameters()),
        "nnz_labels": int(lm.labels.nnz),
        "nnz_labels_exact": int(lm_exact.labels.nnz),
        "label_inflation": float(nnz_c / nnz_e) if nnz_e else 1.0,
        "n_dead_blocks": int(lm.dead.sum()),
        "n_dead_blocks_exact": int(lm_exact.dead.sum()),
        "n_unsat_named": int(comp_unsat.sum()),
        "n_unsat_named_exact": int(exact_unsat.sum()),
        "n_named_made_unsat": int((comp_unsat & ~exact_unsat).sum()),
        "consistent": bool(lm.consistent),
    }
    out["check_model_all_zero"] = bool(all(v == 0 for v in out["check_model"].values()))
    if ref is not None:
        idx = np.flatnonzero(~exact_unsat)
        pred = lm.subsumption_matrix(named[idx], empty_subsumed_by_all=False)
        pm = pair_metrics(pred, restrict(ref.matrix, idx), unsat_pred=comp_unsat[idx])
        out.update({
            "n_named_satisfiable_exact": int(len(idx)),
            "n_entailed_pairs": pm["n_ref"],
            "n_held_pairs": pm["n_pred"],
            "tp": pm["tp"],
            "fp": pm["fp"],
            "fn": pm["fn"],
            "precision": pm["precision"],
            "recall": pm["recall"],
        })
        # diagnostic: the same pairs restricted to classes still satisfiable after compression
        live = np.flatnonzero(~exact_unsat & ~comp_unsat)
        pl = pair_metrics(lm.subsumption_matrix(named[live], empty_subsumed_by_all=False),
                          restrict(ref.matrix, live))
        out.update({
            "n_named_live": int(len(live)),
            "precision_live": pl["precision"],
            "recall_live": pl["recall"],
            "n_false_pairs_live": pl["fp"],
            "false_pairs_per_class_live": float(pl["fp"] / len(live)) if len(live) else 0.0,
        })
    return out


def run_budget(
    dataset: str,
    budgets: Sequence[int] = BUDGETS,
    order: str = "dfs",
    seed: int = 0,
    out_dir: Path = DEFAULT_OUT,
    time_limit: Optional[float] = None,
    max_facts: Optional[int] = None,
    kb: Optional[KBTables] = None,
    ref: Optional[Reference] = None,
    bundle: Any = None,
    write: bool = True,
    skip_existing: bool = False,
) -> List[Dict[str, Any]]:
    """Run B2 for one dataset over ``budgets`` and write one JSON per K.

    Parameters
    ----------
    dataset : str
        Dataset identifier (used for loading unless ``kb`` is given, and for file names).
    budgets : sequence of int
        Budgets in the order to run; budgets larger than the smallest budget that
        is >= the exact maximum number of runs (the first lossless one) are dropped.
    order : str
        ``"dfs"``, ``"identity"`` or ``"best-of-N"``.
    seed : int
        Seed of ``best-of-N``.
    out_dir : Path
        Output directory.
    time_limit : float, optional
        Overall time limit in seconds (remaining budgets are skipped, a running K is
        aborted between iterations).
    max_facts : int, optional
        Memory guard passed to :func:`compress`.
    kb, ref, bundle : optional
        Pre-built tables / reference / bundle (tests); when ``kb`` is given and
        ``ref`` is ``None`` the reference is skipped unless ``bundle`` is given.
    write : bool
        Write JSON files.
    skip_existing : bool
        Skip budgets whose JSON already exists (resume).

    Returns
    -------
    list of dict
        Payload per budget.
    """
    from mowl.geometric.least_model import interval_realisation

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
    exact_real = interval_realisation(lm_exact, o)
    exact_max = int(exact_real.runs_per_class.max())
    base = {
        "dataset": dataset,
        "order_info": info,
        "exact_max_runs": exact_max,
        "exact_total_intervals": int(exact_real.runs_per_class.sum()),
        "exact_n_parameters": int(exact_real.n_parameters()),
        "sizes": kb.sizes(),
        "reference_available": ref is not None,
    }
    print(f"[budget] {dataset} order={info['order']} exact max runs {exact_max}", flush=True)
    lossless = [int(k) for k in budgets if int(k) >= exact_max]
    first_lossless = min(lossless) if lossless else None
    todo = [int(k) for k in budgets if int(k) < exact_max or int(k) == first_lossless]
    todo = list(dict.fromkeys(todo))
    tag = "" if order == "dfs" else f"__{order}"
    results: List[Dict[str, Any]] = []
    for K in todo:
        out_path = Path(out_dir) / f"{dataset}__K{int(K)}{tag}.json"
        if skip_existing and write and out_path.exists():
            print(f"[budget] {dataset} K={K}: exists, skipped", flush=True)
            continue
        if deadline is not None and time.perf_counter() > deadline:
            print(f"[budget] {dataset}: time limit reached before K={K}", flush=True)
            break
        t0 = time.perf_counter()
        res = compress(kb, lm_exact, o, int(K), max_facts=max_facts, deadline=deadline,
                       verbose=write)
        payload = dict(base)
        payload.update({
            "K": int(K),
            "lossless": bool(K >= exact_max),
            "iterations": res["iterations"],
            "n_extra_facts": res["n_facts"],
            "status": res["status"],
            "iteration_log": res["iteration_log"],
        })
        payload.update(budget_metrics(kb, lm_exact, res, ref))
        payload["wall_time_s"] = time.perf_counter() - t0
        payload["peak_rss_gb"] = peak_rss_gb()
        results.append(payload)
        print(
            f"[budget] {dataset} K={K}: it={res['iterations']} status={res['status']} "
            f"max_runs={payload['max_runs']} params={payload['n_parameters']} "
            f"infl={payload['label_inflation']:.3f} check0={payload['check_model_all_zero']} "
            f"P={payload.get('precision')} R={payload.get('recall')} "
            f"P_live={payload.get('precision_live')} "
            f"unsat+={payload['n_named_made_unsat']} t={payload['wall_time_s']:.1f}s",
            flush=True,
        )
        if write:
            write_json(payload, out_path)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser of B2."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", nargs="+", required=True,
                   help=f"Dataset ids ({', '.join(CAPACITY_DATASETS)}) or 'all'.")
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    p.add_argument("--order", default="dfs", help="dfs | identity | best-of-N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--time-limit", type=float, default=None,
                   help="Per-dataset time limit in seconds.")
    p.add_argument("--max-facts", type=int, default=200_000_000,
                   help="Abort a budget when the extra facts exceed this (0 disables).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip budgets whose JSON already exists (resume).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    names = list(DEFAULT_DATASETS) if args.dataset == ["all"] else args.dataset
    failed = []
    for name in names:
        try:
            run_budget(name, budgets=args.budgets, order=args.order, seed=args.seed,
                       out_dir=args.out, time_limit=args.time_limit,
                       max_facts=args.max_facts or None, skip_existing=args.skip_existing)
        except Exception:  # noqa: BLE001 -- report and continue with the others
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"[budget] FAILED: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
