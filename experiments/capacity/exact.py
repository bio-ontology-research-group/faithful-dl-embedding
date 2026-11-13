"""B1: exactness of the least-model solver and intervals per class in one dimension.

Purpose
-------
For one dataset's normalised ontology (tables built exactly as
``GroundedEL(dataset, embed_dim=1, n_components=1, n_role_components=1)`` builds
them) this script records

- sizes (classes, named classes, auxiliary classes, roles, individuals, table rows);
- ``least_model(method="iteration")``: wall time, peak RSS, sweeps, label nnz, edges,
  unsatisfiable classes, consistency, and ``check_model`` on the result;
- agreement with the reference reasoner (``CanonicalModel`` via
  :func:`experiments.cm_cache.get_canonical_model`) over NAMED classes: number of
  entailed pairs ``A ⊑ B`` (``A != B``, both satisfiable), exact precision/recall of
  ``LeastModel.subsumption_matrix`` and agreement on the unsatisfiable named classes;
- ``method="lp"`` when the LP has at most ``--lp-max-variables`` variables: equality
  with the iteration result, ``lp_max_fractionality`` and wall time, for the default
  objective and ``--lp-objectives`` random strictly positive objectives;
- ``interval_realisation`` for ``order="dfs"`` and ``order="identity"``: runs per
  class (max, mean, median, p95, histogram, share with one interval), parameters,
  and a check that membership read from the intervals equals the labels;
- for ``nb_elements <= --geom-max-elements``: the geometric realisation
  ``GroundedEL(init="elements", ..., n_components=nb_elements, embed_dim=2)`` with zero
  training epochs, its ``crisp_report`` and the reasoner-based ``evaluate_model``.

Inputs
------
``--dataset`` (one or more identifiers of :data:`experiments.capacity.common.CAPACITY_DATASETS`
or ``all``), ``--seed`` (random LP objectives), ``--out``.

Outputs
-------
``<out>/<dataset>.json`` (default ``experiments/results/raw-capacity/exact/``).

Usage
-----
``python -m experiments.capacity.exact --dataset pizza tk3``
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.capacity.common import (
    CAPACITY_DATASETS,
    DEFAULT_DATASETS,
    CAPACITY_RAW_DIR,
    KBTables,
    Reference,
    build_tables,
    load_capacity_dataset,
    membership_equals_labels,
    pair_metrics,
    peak_rss_gb,
    reference_entailments,
    restrict,
    runs_summary,
    strip_diagonal,
    write_json,
)

DEFAULT_OUT = CAPACITY_RAW_DIR / "exact"


def same_model(a, b) -> bool:
    """``True`` iff two least models have equal pruned labels, edges and dead blocks.

    Parameters
    ----------
    a, b : LeastModel

    Returns
    -------
    bool
    """
    if a.labels.shape != b.labels.shape or not np.array_equal(a.dead, b.dead):
        return False
    if (a.labels.astype(bool) != b.labels.astype(bool)).nnz:
        return False
    ea = np.unique(a.edges, axis=0) if len(a.edges) else np.empty((0, 3), dtype=np.int64)
    eb = np.unique(b.edges, axis=0) if len(b.edges) else np.empty((0, 3), dtype=np.int64)
    return bool(ea.shape == eb.shape and np.array_equal(ea, eb))


def lp_size(kb: KBTables) -> Dict[str, int]:
    """Variable and (estimated) constraint counts of the literal Horn LP.

    Parameters
    ----------
    kb : KBTables

    Returns
    -------
    dict
        ``n_variables`` and ``n_constraints_est`` (the number of rows the dense LP
        builder in ``mowl.geometric.least_model`` emits).
    """
    nb = kb.nb_classes + kb.nb_individuals
    nc, nr = kb.nb_classes, kb.nb_rels
    t = kb.tables
    n_inv = 0 if kb.inverse_of is None else int(sum(1 for p in kb.inverse_of if p >= 0))
    cons = (
        (len(t["gci0"]) + len(t["gci1"]) + len(t["gci2"]) + len(t["gci0_bot"]) + len(t["gci1_bot"])) * nb
        + (len(t["gci3"]) + len(t["gci3_bot"]) + len(t["role_inclusion"]) + n_inv + nr) * nb * nb
        + len(t["role_chain"]) * nb ** 3
    )
    return {"n_variables": int(nb * nc + nb * nb * nr + nb), "n_constraints_est": int(cons)}


def reference_block(kb: KBTables, lm, ref: Reference) -> Dict[str, Any]:
    """Compare the least model with the reference over named classes.

    Parameters
    ----------
    kb : KBTables
    lm : LeastModel
    ref : Reference
        Reference over ``kb.named_ids`` (same order).

    Returns
    -------
    dict
        Entailed pair count, precision/recall, unsatisfiable-set agreement, timings.
    """
    t0 = time.perf_counter()
    named = kb.named_ids
    lm_unsat = np.isin(named, lm.unsatisfiable_classes())
    ref_unsat = ~ref.satisfiable
    idx = np.flatnonzero(ref.satisfiable)
    pred = strip_diagonal(lm.subsumption_matrix(named[idx], empty_subsumed_by_all=False))
    refm = restrict(ref.matrix, idx)
    pm = pair_metrics(pred, refm)
    dt = time.perf_counter() - t0
    return {
        "source": ref.source,
        "reference_time_s": ref.time_s,
        "compare_time_s": dt,
        "n_named_satisfiable_ref": int(len(idx)),
        "n_entailed_pairs": int(refm.nnz),
        "n_pred_pairs": pm["n_pred"],
        "tp": pm["tp"],
        "fp": pm["fp"],
        "fn": pm["fn"],
        "precision": pm["precision"],
        "recall": pm["recall"],
        "n_unsat_named_ref": int(ref_unsat.sum()),
        "n_unsat_named_lm": int(lm_unsat.sum()),
        "unsat_named_agree": bool(np.array_equal(lm_unsat, ref_unsat)),
        "n_unsat_named_disagree": int((lm_unsat != ref_unsat).sum()),
        "unsat_named_ref_iris": [ref.iris[i] for i in np.flatnonzero(ref_unsat)[:50]],
        "exact_agreement": bool(
            pm["precision"] == 1.0 and pm["recall"] == 1.0 and np.array_equal(lm_unsat, ref_unsat)
        ),
    }


def lp_block(kb: KBTables, lm, n_random: int, seed: int, max_vars: int,
             max_constraints: Optional[int]) -> Dict[str, Any]:
    """Solve the literal Horn LP (default + random positive objectives) and compare.

    Parameters
    ----------
    kb : KBTables
    lm : LeastModel
        Iteration result.
    n_random : int
        Number of random strictly positive objectives.
    seed : int
        Seed of the random objectives (objective ``i`` uses ``seed + i``).
    max_vars : int
        Skip when the variable count exceeds this.
    max_constraints : int or None
        Skip when the estimated constraint count exceeds this (memory guard).

    Returns
    -------
    dict
    """
    size = lp_size(kb)
    out: Dict[str, Any] = dict(size)
    if size["n_variables"] > max_vars:
        out["skipped"] = f"n_variables {size['n_variables']} > {max_vars}"
        return out
    if max_constraints is not None and size["n_constraints_est"] > max_constraints:
        out["skipped"] = f"n_constraints_est {size['n_constraints_est']} > {max_constraints}"
        return out
    runs: List[Dict[str, Any]] = []
    for i in range(n_random + 1):
        kw: Dict[str, Any] = {"method": "lp", "lp_max_variables": int(max_vars)}
        name = "ones"
        if i > 0:
            rng = np.random.default_rng(seed + i)
            kw["objective_weights"] = rng.uniform(0.01, 1.0, size=size["n_variables"])
            name = f"random_seed{seed + i}"
        t0 = time.perf_counter()
        try:
            lp = kb.least_model(**kw)
        except Exception as exc:  # noqa: BLE001 -- record and continue
            runs.append({"objective": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        runs.append(
            {
                "objective": name,
                "wall_time_s": time.perf_counter() - t0,
                "equal_to_iteration": same_model(lm, lp),
                "lp_max_fractionality": float(getattr(lp, "lp_max_fractionality", float("nan"))),
                "lp_objective": float(getattr(lp, "lp_objective", float("nan"))),
            }
        )
    out["runs"] = runs
    out["all_equal"] = bool(runs and all(r.get("equal_to_iteration") for r in runs))
    out["max_fractionality"] = max(
        (r["lp_max_fractionality"] for r in runs if "lp_max_fractionality" in r), default=None
    )
    return out


def intervals_block(kb: KBTables, lm, order: str) -> Dict[str, Any]:
    """Interval realisation statistics for one ``order``.

    Parameters
    ----------
    kb : KBTables
    lm : LeastModel
    order : {"dfs", "identity"}

    Returns
    -------
    dict
    """
    from mowl.geometric.least_model import interval_realisation

    t0 = time.perf_counter()
    real = interval_realisation(lm, order)
    dt = time.perf_counter() - t0
    rp = real.runs_per_class
    return {
        "order": order,
        "time_s": dt,
        "all_classes": runs_summary(rp),
        "named_classes": runs_summary(rp[kb.named_ids]),
        "total_intervals": int(rp.sum()),
        "role_products": int(real.role_products),
        "n_parameters": int(real.n_parameters()),
        "membership_equals_labels": membership_equals_labels(lm, real),
    }


def geometric_block(bundle, lm, seed: int, n_eval: int) -> Dict[str, Any]:
    """Exact geometric realisation ``GroundedEL(init="elements")`` with zero training epochs.

    Parameters
    ----------
    bundle : DatasetBundle
    lm : LeastModel
    seed : int
        GroundedEL seed (sampling inside ``crisp_report``).
    n_eval : int
        Sample budget for ``crisp_report`` / ``evaluate_model``.

    Returns
    -------
    dict
        ``crisp_report`` fields and ``evaluate_model(use_reasoner=True)`` fields.
    """
    from mowl.models import GroundedEL

    t0 = time.perf_counter()
    nb_el = int(lm.nb_classes + lm.nb_individuals)
    g = GroundedEL(
        bundle.dataset,
        init="elements",
        init_labels=lm.dense_labels(),
        init_edges=lm.element_edges(),
        n_components=nb_el,
        embed_dim=2,
        seed=int(seed),
        log_every=0,
    )
    crisp = g.crisp_report(n_eval=int(n_eval))
    ev = g.evaluate_model(use_reasoner=True, n_eval=int(n_eval))
    return {
        "nb_elements": nb_el,
        "n_components": int(g.n_components),
        "embed_dim": 2,
        "epochs": 0,
        "elem_viol": float(crisp["elem_viol"]),
        "region_viol": float(crisp["region_viol"]),
        "n_empty_classes": int(crisp["n_empty_classes"]),
        "n_dead_elements": int(crisp["n_dead_elements"]),
        "n_axioms_violated_by_depth": dict(crisp["n_axioms_violated_by_depth"]),
        "evaluate_model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ev.items()},
        "exact": bool(ev["exact"]),
        "faithful_named_precision": ev["faithful_named_precision"],
        "faithful_named_recall": ev["faithful_named_recall"],
        "time_s": time.perf_counter() - t0,
    }


def run_exact(
    dataset: str,
    out_dir: Path = DEFAULT_OUT,
    seed: int = 0,
    lp_max_variables: int = 2_000_000,
    lp_max_constraints: Optional[int] = 20_000_000,
    lp_objectives: int = 3,
    geom_max_elements: int = 1500,
    n_eval: int = 10000,
    skip_lp: bool = False,
    skip_geom: bool = False,
    skip_reference: bool = False,
) -> Dict[str, Any]:
    """Run B1 on one dataset and write ``<out_dir>/<dataset>.json``.

    Parameters
    ----------
    dataset : str
        Dataset identifier.
    out_dir : Path
        Output directory.
    seed : int
        Seed for the random LP objectives and the geometric realisation.
    lp_max_variables : int
        LP is solved only up to this many variables.
    lp_max_constraints : int or None
        Memory guard on the estimated LP row count (``None`` disables it).
    lp_objectives : int
        Number of random strictly positive objectives.
    geom_max_elements : int
        Geometric realisation only up to this many elements.
    n_eval : int
        Sample budget of the geometric crisp report.
    skip_lp, skip_geom, skip_reference : bool
        Skip the corresponding block.

    Returns
    -------
    dict
        The payload written to disk.
    """
    from mowl.geometric.errors import InconsistentOntology

    t_all = time.perf_counter()
    payload: Dict[str, Any] = {"dataset": dataset, "seed": int(seed)}
    bundle = load_capacity_dataset(dataset)
    kb = build_tables(bundle)
    payload["table_source"] = "GroundedEL(dataset, embed_dim=1, n_components=1, n_role_components=1)"
    payload["table_build_time_s"] = kb.build_time_s
    payload["sizes"] = kb.sizes()
    print(f"[exact] {dataset}: tables {kb.build_time_s:.1f}s {payload['sizes']}", flush=True)

    rss0 = peak_rss_gb()
    t0 = time.perf_counter()
    lm = kb.least_model(method="iteration")
    dt = time.perf_counter() - t0
    unsat = lm.unsatisfiable_classes()
    payload["least_model"] = {
        "method": "iteration",
        "wall_time_s": dt,
        "peak_rss_gb_before": rss0,
        "peak_rss_gb_after": peak_rss_gb(),
        "n_sweeps": int(lm.n_sweeps),
        "n_blocks": int(lm.n_blocks),
        "n_live_blocks": int((~lm.dead).sum()),
        "nnz_labels": int(lm.labels.nnz),
        "n_edges": int(len(lm.edges)),
        "n_unsat_classes": int(len(unsat)),
        "n_unsat_named": int(np.isin(kb.named_ids, unsat).sum()),
        "consistent": bool(lm.consistent),
    }
    print(f"[exact] {dataset}: least model {payload['least_model']}", flush=True)
    t0 = time.perf_counter()
    chk = kb.check(lm)
    payload["check_model"] = chk
    payload["check_model_time_s"] = time.perf_counter() - t0
    payload["check_model_all_zero"] = bool(all(v == 0 for v in chk.values()))
    print(f"[exact] {dataset}: check_model all zero = {payload['check_model_all_zero']}", flush=True)

    if not skip_reference:
        print(f"[exact] {dataset}: computing reference (CanonicalModel) ...", flush=True)
        try:
            ref = reference_entailments(bundle, [kb.class_iris[i] for i in kb.named_ids])
            payload["reference"] = reference_block(kb, lm, ref)
        except InconsistentOntology as exc:
            payload["reference"] = {"error": f"InconsistentOntology: {exc}"}
        print(f"[exact] {dataset}: reference {payload['reference']}", flush=True)

    if not skip_lp:
        payload["lp"] = lp_block(kb, lm, lp_objectives, seed, lp_max_variables, lp_max_constraints)
        print(f"[exact] {dataset}: lp {payload['lp']}", flush=True)

    payload["intervals"] = {o: intervals_block(kb, lm, o) for o in ("dfs", "identity")}
    for o, blk in payload["intervals"].items():
        a = blk["all_classes"]
        print(
            f"[exact] {dataset}: intervals[{o}] max {a['max']} mean {a['mean']:.3f} "
            f"share1 {a['share_one']:.3f} params {blk['n_parameters']} "
            f"membership_ok {blk['membership_equals_labels']}",
            flush=True,
        )

    nb_el = kb.nb_classes + kb.nb_individuals
    if skip_geom:
        payload["geometric"] = {"skipped": "--skip-geom"}
    elif nb_el > geom_max_elements:
        payload["geometric"] = {"skipped": f"nb_elements {nb_el} > {geom_max_elements}"}
    else:
        try:
            payload["geometric"] = geometric_block(bundle, lm, seed, n_eval)
        except Exception as exc:  # noqa: BLE001 -- record the failure in the result
            payload["geometric"] = {"error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}
        print(f"[exact] {dataset}: geometric {payload['geometric']}", flush=True)

    payload["total_time_s"] = time.perf_counter() - t_all
    payload["peak_rss_gb"] = peak_rss_gb()
    write_json(payload, Path(out_dir) / f"{dataset}.json")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser of B1."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", nargs="+", required=True,
                   help=f"Dataset ids ({', '.join(CAPACITY_DATASETS)}) or 'all'.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lp-max-variables", type=int, default=2_000_000)
    p.add_argument("--lp-max-constraints", type=int, default=20_000_000,
                   help="Memory guard on the estimated LP row count (0 disables).")
    p.add_argument("--lp-objectives", type=int, default=3)
    p.add_argument("--geom-max-elements", type=int, default=1500)
    p.add_argument("--n-eval", type=int, default=10000)
    p.add_argument("--skip-lp", action="store_true")
    p.add_argument("--skip-geom", action="store_true")
    p.add_argument("--skip-reference", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    names = list(DEFAULT_DATASETS) if args.dataset == ["all"] else args.dataset
    failed = []
    for name in names:
        try:
            run_exact(
                name,
                out_dir=args.out,
                seed=args.seed,
                lp_max_variables=args.lp_max_variables,
                lp_max_constraints=args.lp_max_constraints or None,
                lp_objectives=args.lp_objectives,
                geom_max_elements=args.geom_max_elements,
                n_eval=args.n_eval,
                skip_lp=args.skip_lp,
                skip_geom=args.skip_geom,
                skip_reference=args.skip_reference,
            )
        except Exception:  # noqa: BLE001 -- report and continue with the others
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"[exact] FAILED: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
