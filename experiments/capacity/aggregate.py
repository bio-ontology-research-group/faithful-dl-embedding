"""B4: aggregate the capacity experiments into ``CAPACITY.md`` and ``capacity_curve.csv``.

Purpose
-------
Collect the JSON results of B1 (``raw-capacity/exact``), B2 (``raw-capacity/budget``)
and B3 (``raw-capacity/baselines``) and render

(i)   the exactness table per dataset (B1);
(ii)  intervals per class in one dimension (B1);
(iii) the budget curve per dataset (B2);
(iv)  the single-box baselines per dataset x method x d, mean ± sd over seeds (B3);

plus a tidy CSV with columns ``dataset, family, capacity_name, capacity,
params_per_class, params_total, modelness, precision, recall, n_unsat_named, seed``
(``family`` in ``{exact, budget, elem, elbe, box2el, transbox}``).

Inputs
------
``--raw`` (default ``experiments/results/raw-capacity``).

Outputs
-------
``--md`` (default ``experiments/results/CAPACITY.md``) and ``--csv`` (default
``experiments/results/capacity_curve.csv``).

Conventions
-----------
For ``exact`` and ``budget`` rows ``modelness`` is 1.0 when ``check_model`` reports no
violated ground instance (empty otherwise), ``params_total`` is ``n_parameters()`` of
the interval realisation (2 per interval + 3 per role product) and
``params_per_class = params_total / n_classes`` (all classes of the normalised
ontology, auxiliaries included). For baselines ``params_total`` counts all trainable
parameters and ``params_per_class`` the class-indexed ones divided by ``n_classes``.

Usage
-----
``python -m experiments.capacity.aggregate``
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.capacity.common import CAPACITY_DATASETS, CAPACITY_RAW_DIR, CORRUPT_DATASETS
from experiments.config import RESULTS_DIR

CSV_COLUMNS = (
    "dataset",
    "family",
    "capacity_name",
    "capacity",
    "params_per_class",
    "params_total",
    "modelness",
    "precision",
    "recall",
    "precision_live",
    "recall_live",
    "protect",
    "n_unsat_named",
    "seed",
)
METHOD_ORDER = ("elem", "elbe", "box2el", "transbox")


def _load_dir(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for f in sorted(path.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def _ds_key(name: str) -> int:
    return CAPACITY_DATASETS.index(name) if name in CAPACITY_DATASETS else len(CAPACITY_DATASETS)


def _f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{nd}f}"
    return str(x)


def _ms(vals: Sequence[float], nd: int = 3) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}"


def _table(header: Sequence[str], rows: List[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def exact_tables(exact: List[Dict[str, Any]]) -> str:
    """Render tables (i) and (ii)."""
    rows_i, rows_ii = [], []
    for p in sorted(exact, key=lambda p: _ds_key(p["dataset"])):
        s, lm, ref = p["sizes"], p["least_model"], p.get("reference", {})
        lp = p.get("lp", {})
        if "skipped" in lp:
            lp_s = "skipped (vars)" if "n_variables" in lp.get("skipped", "") else "skipped"
        elif "runs" in lp:
            lp_s = (f"{'equal' if lp['all_equal'] else '**differs**'} ×{len(lp['runs'])}, "
                    f"frac {lp['max_fractionality']:.0e}, "
                    f"{max(r.get('wall_time_s', 0) for r in lp['runs']):.1f}s")
        else:
            lp_s = "–"
        g = p.get("geometric", {})
        g_s = ("skipped" if "skipped" in g else ("error" if "error" in g else
               f"{'exact' if g.get('exact') else '**not exact**'} (P {g.get('faithful_named_precision'):.2f}, "
               f"R {g.get('faithful_named_recall'):.2f})"))
        rows_i.append([
            p["dataset"], _f(s["n_classes"]), _f(s["n_named_classes"]), _f(s["n_aux_classes"]),
            _f(s["n_roles"]), _f(s["n_individuals"]),
            f"{lm['wall_time_s']:.2f}", f"{lm['peak_rss_gb_after']:.1f}", _f(lm["n_sweeps"]),
            _f(lm["nnz_labels"]), _f(lm["n_edges"]),
            f"{lm['n_unsat_classes']} / {lm['n_unsat_named']}", _f(lm["consistent"]),
            _f(p["check_model_all_zero"]),
            _f(ref.get("n_entailed_pairs")), _f(ref.get("precision")), _f(ref.get("recall")),
            _f(ref.get("unsat_named_agree")), lp_s, g_s,
        ])
        iv = p["intervals"]
        d, i = iv["dfs"], iv["identity"]
        rows_ii.append([
            p["dataset"], _f(d["all_classes"]["max"]), f"{d['all_classes']['mean']:.3f}",
            f"{d['all_classes']['median']:.0f}", f"{d['all_classes']['p95']:.1f}",
            f"{100 * d['all_classes']['share_one']:.1f}", f"{100 * d['named_classes']['share_one']:.1f}",
            _f(d["total_intervals"]), _f(d["role_products"]), _f(d["n_parameters"]),
            _f(i["all_classes"]["max"]), f"{i['all_classes']['mean']:.3f}",
            f"{100 * i['all_classes']['share_one']:.1f}", _f(i["n_parameters"]),
            _f(d["membership_equals_labels"] and i["membership_equals_labels"]),
            _hist(d["all_classes"]["hist"]),
        ])
    t1 = _table(
        ["dataset", "classes", "named", "aux", "roles", "inds", "LM s", "RSS GB", "sweeps",
         "nnz labels", "edges", "unsat all / named", "consistent", "check_model = 0",
         "entailed named pairs", "P", "R", "unsat agree", "LP vs iteration", "geometric (d=2)"],
        rows_i,
    )
    t2 = _table(
        ["dataset", "dfs max", "dfs mean", "median", "p95", "% one (all)", "% one (named)",
         "intervals", "role products", "params", "identity max", "identity mean",
         "identity % one", "identity params", "membership = labels", "dfs histogram"],
        rows_ii,
    )
    return ("## (i) Exactness of the least-model solver (B1)\n\n" + t1 + "\n\n"
            "## (ii) Intervals per class in one dimension (B1)\n\n" + t2 + "\n")


def _hist(h: Dict[str, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in h.items())


def budget_table(budget: List[Dict[str, Any]]) -> str:
    """Render table (iii)."""
    by_ds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in budget:
        by_ds[p["dataset"]].append(p)
    parts = ["## (iii-b) SUPERSEDED: per-class budget, at most K intervals per class (B2)\n",
             "Kept for the record: a uniform cap collapses on large ontologies, and the GO runs "
             "predate the fix of a solver overflow (their `check_model` failures).\n"]
    for ds in sorted(by_ds, key=_ds_key):
        rows = []
        ps = sorted(by_ds[ds], key=lambda p: (p["order_info"]["order"], p["K"]))
        for p in ps:
            rows.append([
                p["order_info"]["order"], _f(p["K"]), _f(p["iterations"]), p["status"],
                _f(p["max_runs"]), _f(p["total_intervals"]), _f(p["n_parameters"]),
                f"{p['label_inflation']:.3f}", _f(p["n_named_made_unsat"]),
                _f(p.get("precision")), _f(p.get("recall")), _f(p.get("precision_live")),
                _f(p["check_model_all_zero"]),
                f"{p['wall_time_s']:.1f}",
            ])
        ex = ps[0]
        parts.append(
            f"### {ds}\n\nexact: max runs {ex['exact_max_runs']:,}, "
            f"{ex['exact_total_intervals']:,} intervals, {ex['exact_n_parameters']:,} parameters\n\n"
            + _table(["order", "K", "iterations", "status", "max runs", "intervals", "params",
                      "label inflation", "named made unsat", "P", "R", "P (still-sat classes)",
                      "check_model = 0", "time s"],
                     rows)
            + "\n"
        )
    return "\n".join(parts)


def baseline_table(base: List[Dict[str, Any]], title: str = "## (iv) Single-box baselines (B3)") -> str:
    """Render table (iv)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for p in base:
        c = p["config"]
        groups[(c["dataset"], c["method"], int(c["dim"]))].append(p)
    rows = []
    for key in sorted(groups, key=lambda k: (_ds_key(k[0]), METHOD_ORDER.index(k[1])
                                             if k[1] in METHOD_ORDER else 9, k[2])):
        ps = groups[key]
        rows.append([
            key[0], key[1], str(key[2]), str(len(ps)), str(sorted({p["config"]["epochs"] for p in ps})),
            _ms([p["modelness_all"] for p in ps]), _ms([p["precision"] for p in ps]),
            _ms([p["recall"] for p in ps]), _ms([float(p["n_unsat_named"]) for p in ps], 1),
            f"{np.mean([p['params_per_class'] for p in ps]):.0f}",
            _f(int(np.mean([p["params"]["n_params_total"] for p in ps]))),
        ])
    if not rows:
        return f"{title}\n\nNo baseline results yet.\n"
    return (f"{title}, mean ± sd over seeds\n\n"
            + _table(["dataset", "method", "d", "seeds", "epochs", "modelness_all", "P (crisp)",
                      "R (crisp)", "unsat named", "params / class", "params total"], rows)
            + "\n")


def budget_total_table(total: List[Dict[str, Any]]) -> str:
    """Render table (iii): total-budget curve, both protect settings."""
    by: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for p in total:
        if p["order_info"]["order"] == "dfs":
            by[(p["dataset"], p["protect"] + ", cost = " + p.get("cost", "size"))].append(p)
    parts = ["## (iii) Budget curve: sound compression under a total interval budget (B2)\n",
             "Target = n + f·(T_exact − n) intervals (n classes with ≥ 1 interval); the globally "
             "smallest gaps are filled first, then the model is re-closed. `protect = disjoint` "
             "skips gaps whose filling makes a block unsatisfiable at once (the target may then "
             "be unreachable). `cost` ranks the gaps: `size` = blocks in the gap, `weight` = labels "
             "added once subclasses inherit. P/R (live) are over named classes still "
             "satisfiable. A status `aborted: out of memory` marks the last model computable "
             "under a 40 GB limit.\n"]
    for ds, prot in sorted(by, key=lambda k: (_ds_key(k[0]), k[1])):
        rows = []
        for p in sorted(by[(ds, prot)], key=lambda p: -p["fraction"]):
            rows.append([
                f"{p['fraction']:g}", _f(p["target_total"]), _f(p["achieved_total"]),
                _f(p["n_parameters"]), _f(p["max_runs"]), _f(p["n_named_made_unsat"]),
                _f(p.get("precision_live")), _f(p.get("recall_live")),
                _f(p.get("false_pairs_per_class_live"), 2), f"{p['label_inflation']:.3f}",
                _f(p["check_model_all_zero"]), p["status"],
            ])
        parts.append(f"### {ds}, protect = {prot}\n\n" + _table(
            ["f", "target", "achieved intervals", "params", "max runs", "named made unsat",
             "P (live)", "R (live)", "false pairs / class", "label inflation",
             "check_model = 0", "status"], rows) + "\n")
    return "\n".join(parts)


def csv_rows(exact, budget, base, total=()) -> List[Dict[str, Any]]:
    """Tidy rows of ``capacity_curve.csv``."""
    rows: List[Dict[str, Any]] = []
    for p in total:
        if p["order_info"]["order"] != "dfs":
            continue
        n_cls = p["sizes"]["n_classes"]
        rows.append({
            "dataset": p["dataset"], "family": "budget_total", "capacity_name": "fraction",
            "capacity": p["achieved_total"], "params_per_class": p["n_parameters"] / n_cls,
            "params_total": p["n_parameters"],
            "modelness": 1.0 if p["check_model_all_zero"] else None,
            "precision": p.get("precision"), "recall": p.get("recall"),
            "precision_live": p.get("precision_live"), "recall_live": p.get("recall_live"),
            "protect": p["protect"] + ("" if p.get("cost", "size") == "size" else "/" + p["cost"]),
            "n_unsat_named": p["n_unsat_named"], "seed": None,
        })
    for p in exact:
        d = p["intervals"]["dfs"]
        ref = p.get("reference", {})
        n_cls = p["sizes"]["n_classes"]
        rows.append({
            "dataset": p["dataset"], "family": "exact", "capacity_name": "K_max_dfs",
            "capacity": d["all_classes"]["max"],
            "params_per_class": d["n_parameters"] / n_cls, "params_total": d["n_parameters"],
            "modelness": 1.0 if p["check_model_all_zero"] else None,
            "precision": ref.get("precision"), "recall": ref.get("recall"),
            "n_unsat_named": p["least_model"]["n_unsat_named"], "seed": None,
        })
    for p in budget:
        if p["order_info"]["order"] != "dfs":
            continue
        n_cls = p["sizes"]["n_classes"]
        rows.append({
            "dataset": p["dataset"], "family": "budget", "capacity_name": "K",
            "capacity": p["K"], "params_per_class": p["n_parameters"] / n_cls,
            "params_total": p["n_parameters"],
            "modelness": 1.0 if p["check_model_all_zero"] else None,
            "precision": p.get("precision"), "recall": p.get("recall"),
            "precision_live": p.get("precision_live"), "recall_live": p.get("recall_live"),
            "n_unsat_named": p["n_unsat_named"], "seed": None,
        })
    for p in base:
        c = p["config"]
        rows.append({
            "dataset": c["dataset"], "family": c["method"], "capacity_name": "dim",
            "capacity": c["dim"], "params_per_class": p["params_per_class"],
            "params_total": p["params"]["n_params_total"], "modelness": p["modelness_all"],
            "precision": p["precision"], "recall": p["recall"],
            "precision_live": p["precision"], "recall_live": p["recall"],
            "n_unsat_named": p["n_unsat_named"], "seed": c["seed"],
        })
    for r in rows:
        r.setdefault("precision_live", r.get("precision"))
        r.setdefault("recall_live", r.get("recall"))
        r.setdefault("protect", None)
    return rows


def aggregate(raw: Path = CAPACITY_RAW_DIR, md: Path = RESULTS_DIR / "CAPACITY.md",
              csv_path: Path = RESULTS_DIR / "capacity_curve.csv",
              notes: Optional[str] = None) -> Dict[str, int]:
    """Render ``CAPACITY.md`` and ``capacity_curve.csv``.

    Parameters
    ----------
    raw : Path
        Root of the raw results (``exact/``, ``budget/``, ``baselines/``).
    md, csv_path : Path
        Outputs.
    notes : str, optional
        Markdown inserted after the title (run notes).

    Returns
    -------
    dict
        Number of result files per family.
    """
    raw = Path(raw)
    corrupt = [p for p in _load_dir(raw / "exact") if p["dataset"] in CORRUPT_DATASETS]
    exact = [p for p in _load_dir(raw / "exact") if p["dataset"] not in CORRUPT_DATASETS]
    budget = [p for p in _load_dir(raw / "budget") if p["dataset"] not in CORRUPT_DATASETS]
    total = [p for p in _load_dir(raw / "budget-total") if p["dataset"] not in CORRUPT_DATASETS]
    base = [p for p in _load_dir(raw / "baselines") if p["config"]["dataset"] not in CORRUPT_DATASETS]
    text = ["# Capacity: exactness, intervals per class, budget curve, single-box baselines\n",
            "Generated by `python -m experiments.capacity.aggregate` from "
            f"`{raw.relative_to(RESULTS_DIR.parent) if raw.is_relative_to(RESULTS_DIR.parent) else raw}`. "
            "P/R are exact (not sampled) precision/recall of the named subsumptions "
            "`A ⊑ B` (`A ≠ B`) holding in the model against the CanonicalModel entailments, "
            "over named classes satisfiable in the reference (B1, B3) or in the exact least "
            "model (B2); in B2 a class made unsatisfiable counts as subsumed by every class, "
            "in B3 a degenerate region (side or diameter ≤ 1e-3) counts as unsatisfiable. "
            "When a model predicts no pair, P is reported as 1.\n"]
    if notes:
        text.append(notes.strip() + "\n")
    text.append(exact_tables(exact) if exact else "## (i)/(ii)\n\nNo B1 results.\n")
    note = ("The Box²EL benchmark ontologies (`galen_j24`, `go_j24`, `anatomy_j24`) are left out of "
            "all tables: in the upstream release the helper classes of the normalisation share "
            "IRIs with real classes (mOWL issue #178), so the files are not sound encodings of "
            "the ontologies.")
    for p in sorted(corrupt, key=lambda p: p["dataset"]):
        lm = p.get("least_model", {})
        note += (f" `{p['dataset']}`: {p['sizes']['n_classes']:,} classes, "
                 f"{lm.get('n_dead_blocks', '?'):,} dead blocks in the least model."
                 if isinstance(lm.get("n_dead_blocks"), int) else "")
    text.append(note + "\n")
    text.append(budget_total_table(total) if total else "## (iii)\n\nNo total-budget results.\n")
    text.append(budget_table(budget) if budget else "## (iii-b)\n\nNo per-class B2 results.\n")
    text.append(baseline_table(base))
    smoke = _load_dir(raw / "baselines-smoke")
    if smoke:
        text.append(baseline_table(
            smoke, "## (iv-smoke) CPU smoke test of B3 (≤ 200 epochs; NOT the experiment, not in the CSV)"))
    Path(md).write_text("\n".join(text), encoding="utf-8")
    rows = csv_rows(exact, budget, base, total)
    with Path(csv_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in CSV_COLUMNS})
    return {"exact": len(exact), "budget": len(budget), "budget_total": len(total),
            "baselines": len(base), "csv_rows": len(rows)}


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser of B4."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", type=Path, default=CAPACITY_RAW_DIR)
    p.add_argument("--md", type=Path, default=RESULTS_DIR / "CAPACITY.md")
    p.add_argument("--csv", type=Path, default=RESULTS_DIR / "capacity_curve.csv")
    p.add_argument("--notes", type=Path, default=None, help="Markdown file with run notes.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    notes = args.notes.read_text(encoding="utf-8") if args.notes else None
    counts = aggregate(args.raw, args.md, args.csv, notes)
    print(f"[aggregate] {counts} -> {args.md}, {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
