"""Aggregate raw run JSONs into per-dataset Markdown tables, MultiBoxEL k-curves, and CSV (Task 8, Spec §5.5).

CLI usage
---------
::

    python -m experiments.aggregate [--raw results/raw] [--out results/tables] [--csv results/all_runs.csv]

Produces
--------
- ``<out>/{dataset}.md`` for every dataset present in ``--raw`` (rows = methods in
  ``config.METHODS`` order, stratified by ``dim`` when a dataset has multiple
  dimensions; columns = Spec §5.3 metrics formatted as ``mean ± sd`` with seed
  count ``n``).
- ``<out>/multibox_curve.md`` comparing ``multibox-k1``, ``multibox-k2``,
  ``multibox-k3``, ``multibox-k4`` (and ``multibox-k2-init``) across datasets and
  dimensions.
- ``<csv>`` (default ``results/all_runs.csv``, or ``<out>/../all_runs.csv`` when
  ``--out`` is overridden and ``--csv`` is not passed): flat table with one row
  per JSON file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.config import (
    DATASETS,
    DATASET_TASKS,
    METHODS,
    RAW_RESULTS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
)

SUBSUMPTION_COLUMNS: List[Tuple[str, str]] = [
    ("rank_f_mrr", "F-MRR"),
    ("rank_f_hits@1", "F-H@1"),
    ("rank_f_hits@3", "F-H@3"),
    ("rank_f_hits@10", "F-H@10"),
    ("rank_f_auc", "F-AUC"),
    ("rank_mrr", "MRR"),
    ("auc_pos_neg", "AUC(+/-)"),
    ("pair_fvr", "Pair-FVR"),
    ("fvr", "FVR"),
    ("modelness_all", "Modelness"),
    ("helly_violation_rate", "HVR"),
    ("degeneracy_rate", "DegRate"),
    ("mean_effective_components", "Eff-k"),
    ("mean_epoch_time_s", "Epoch(s)"),
    ("total_train_time_s", "Train(s)"),
    ("peak_gpu_memory_mb", "GPU(MB)"),
]

SATISFACTION_COLUMNS: List[Tuple[str, str]] = [
    ("modelness_all", "Modelness"),
    ("modelness_gci0", "Mod(gci0)"),
    ("modelness_gci1", "Mod(gci1)"),
    ("modelness_gci1_bot", "Mod(gci1_bot)"),
    ("helly_violation_rate", "HVR"),
    ("degeneracy_rate", "DegRate"),
    ("n_certificates", "Certs"),
    ("mean_effective_components", "Eff-k"),
    ("param_count", "Params"),
    ("mean_epoch_time_s", "Epoch(s)"),
    ("total_train_time_s", "Train(s)"),
    ("peak_gpu_memory_mb", "GPU(MB)"),
]

CURVE_METHODS: List[str] = [
    "multibox-k1",
    "multibox-k2",
    "multibox-k3",
    "multibox-k4",
    "multibox-k2-init",
]


def load_raw_runs(raw_dir: Path) -> List[Dict[str, Any]]:
    """Load all ``*.json`` run files in ``raw_dir`` sorted by filename."""
    if not raw_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    for p in sorted(raw_dir.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["_source_file"] = p.name
        records.append(payload)
    return records


def _format_mean_sd(values: Sequence[Optional[float]], decimals: int = 4) -> str:
    """Format numeric values as ``mean ± sd`` (or ``mean`` when ``n == 1``, ``—`` when empty/null)."""
    clean = [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if not clean:
        return "—"
    mean_v = float(np.mean(clean))
    if len(clean) == 1:
        if all(float(v).is_integer() for v in clean) and decimals == 0:
            return f"{int(round(mean_v))}"
        return f"{mean_v:.{decimals}f}"
    sd_v = float(np.std(clean, ddof=0))
    return f"{mean_v:.{decimals}f} ± {sd_v:.{decimals}f}"


def _decimals_for_key(metric_key: str) -> int:
    """Choose formatting precision for ``metric_key``."""
    if metric_key in ("n_certificates", "param_count"):
        return 0
    if metric_key in ("peak_gpu_memory_mb",):
        return 1
    if metric_key in ("mean_effective_components", "total_train_time_s"):
        return 2
    return 4


def render_dataset_markdown(
    dataset: str,
    runs: Sequence[Dict[str, Any]],
) -> str:
    """Render the Markdown summary table for ``dataset`` across dimensions and methods."""
    task = DATASET_TASKS.get(dataset, "subsumption")
    cols = SUBSUMPTION_COLUMNS if task == "subsumption" else SATISFACTION_COLUMNS

    by_dim_method: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    dims_seen = set()
    for r in runs:
        cfg = r["config"]
        d = int(cfg["dim"])
        m = str(cfg["method"])
        dims_seen.add(d)
        by_dim_method[(d, m)].append(r)

    sorted_dims = sorted(dims_seen)
    method_order = list(METHODS.keys())
    for _, m in sorted(by_dim_method.keys()):
        if m not in method_order:
            method_order.append(m)

    lines: List[str] = [
        f"# Results: `{dataset}` ({task})",
        "",
    ]

    for dim in sorted_dims:
        lines.append(f"## Dimension `d = {dim}`")
        lines.append("")
        headers = ["Method", "k", "Init", "Anchor", "n"] + [
            label for _, label in cols
        ]
        sep = ["---", "---:", "---", "---:", "---:"] + ["---:" for _ in cols]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for m_id in method_order:
            cell_runs = by_dim_method.get((dim, m_id), [])
            if not cell_runs:
                continue
            cfg0 = cell_runs[0]["config"]
            k_val = str(cfg0.get("n_components", METHODS.get(m_id, {}).get("n_components", "—")))
            init_val = str(cfg0.get("init") or "random")
            anchor_val = f"{float(cfg0.get('anchor_weight', 0.0)):g}"
            n_seeds = str(len(cell_runs))

            row_cells = [f"`{m_id}`", k_val, init_val, anchor_val, n_seeds]
            for metric_key, _ in cols:
                vals = [cr["metrics"].get(metric_key) for cr in cell_runs]
                row_cells.append(
                    _format_mean_sd(vals, decimals=_decimals_for_key(metric_key))
                )
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_multibox_curve_markdown(runs: Sequence[Dict[str, Any]]) -> str:
    """Render ``multibox_curve.md`` showing metrics as a function of ``k in {1, 2, 3, 4}``."""
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    pairs_seen = set()
    for r in runs:
        cfg = r["config"]
        m_id = str(cfg["method"])
        if m_id not in CURVE_METHODS:
            continue
        ds = str(cfg["dataset"])
        dim = int(cfg["dim"])
        pairs_seen.add((ds, dim))
        grouped[(ds, dim, m_id)].append(r)

    dataset_order = {d: i for i, d in enumerate(DATASETS)}
    sorted_pairs = sorted(
        pairs_seen, key=lambda pair: (dataset_order.get(pair[0], 999), pair[0], pair[1])
    )

    lines: List[str] = [
        "# MultiBoxEL Relaxation Curve (`k ∈ {1, 2, 3, 4}`)",
        "",
        "| Dataset | d | Method | k | n | Modelness | HVR | DegRate | Eff-k | F-MRR | F-H@10 | AUC(+/-) | Pair-FVR | FVR | Train(s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    curve_keys = [
        ("modelness_all", 4),
        ("helly_violation_rate", 4),
        ("degeneracy_rate", 4),
        ("mean_effective_components", 2),
        ("rank_f_mrr", 4),
        ("rank_f_hits@10", 4),
        ("auc_pos_neg", 4),
        ("pair_fvr", 4),
        ("fvr", 4),
        ("total_train_time_s", 2),
    ]

    for ds, dim in sorted_pairs:
        for m_id in CURVE_METHODS:
            cell_runs = grouped.get((ds, dim, m_id), [])
            if not cell_runs:
                continue
            cfg0 = cell_runs[0]["config"]
            k_val = str(cfg0.get("n_components", "—"))
            n_seeds = str(len(cell_runs))
            row = [f"`{ds}`", str(dim), f"`{m_id}`", k_val, n_seeds]
            for m_key, dec in curve_keys:
                vals = [cr["metrics"].get(m_key) for cr in cell_runs]
                row.append(_format_mean_sd(vals, decimals=dec))
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def write_all_runs_csv(runs: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    """Write a flat CSV table containing one row per run JSON."""
    config_fields = [
        "dataset",
        "task",
        "method",
        "model",
        "n_components",
        "init",
        "anchor_weight",
        "dim",
        "seed",
        "epochs",
        "batch_size",
        "effective_batch_size",
        "oom_retries",
        "device",
    ]
    timing_fields = [
        "wall_clock_s",
        "total_train_time_s",
        "mean_epoch_time_s",
        "eval_time_s",
    ]
    env_fields = [
        "git_hash_experiments",
        "git_hash_mowl",
        "torch_version",
        "cuda_version",
        "gpu_name",
    ]

    metric_keys_set = set()
    for r in runs:
        metric_keys_set.update(r.get("metrics", {}).keys())
    metric_fields = sorted(metric_keys_set)

    fieldnames = (
        ["source_file"]
        + config_fields
        + metric_fields
        + [f"timing_{k}" for k in timing_fields]
        + env_fields
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            cfg = r.get("config", {})
            m = r.get("metrics", {})
            t = r.get("timings", {})
            env = r.get("env", {})
            row: Dict[str, Any] = {"source_file": r.get("_source_file", "")}
            for k in config_fields:
                row[k] = cfg.get(k)
            for k in metric_fields:
                row[k] = m.get(k)
            for k in timing_fields:
                row[f"timing_{k}"] = t.get(k)
            for k in env_fields:
                row[k] = env.get(k)
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``experiments/aggregate.py``."""
    parser = argparse.ArgumentParser(
        description="Aggregate raw JSON experiment results into Markdown tables and CSV."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_RESULTS_DIR,
        help="Directory containing raw JSON run files (default: results/raw/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TABLES_DIR,
        help="Directory for generated Markdown tables (default: results/tables/).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path for flat CSV output (default: results/all_runs.csv, or <out>/../all_runs.csv).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``experiments/aggregate.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw_dir = Path(args.raw).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv is not None:
        csv_path = Path(args.csv).expanduser().resolve()
    elif out_dir == TABLES_DIR.resolve():
        csv_path = (RESULTS_DIR / "all_runs.csv").resolve()
    else:
        csv_path = (out_dir.parent / "all_runs.csv").resolve()

    runs = load_raw_runs(raw_dir)
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        ds = str(r.get("config", {}).get("dataset", "unknown"))
        by_dataset[ds].append(r)

    written_tables: List[Path] = []
    for ds in DATASETS:
        if ds not in by_dataset:
            continue
        md_path = out_dir / f"{ds}.md"
        md_path.write_text(
            render_dataset_markdown(ds, by_dataset[ds]),
            encoding="utf-8",
        )
        written_tables.append(md_path)

    for ds in sorted(by_dataset.keys()):
        if ds in DATASETS:
            continue
        md_path = out_dir / f"{ds}.md"
        md_path.write_text(
            render_dataset_markdown(ds, by_dataset[ds]),
            encoding="utf-8",
        )
        written_tables.append(md_path)

    curve_path = out_dir / "multibox_curve.md"
    curve_path.write_text(
        render_multibox_curve_markdown(runs),
        encoding="utf-8",
    )
    written_tables.append(curve_path)

    write_all_runs_csv(runs, csv_path)
    print(
        f"[aggregate] Aggregated {len(runs)} runs from {raw_dir} -> "
        f"{len(written_tables)} Markdown tables in {out_dir} and CSV {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
