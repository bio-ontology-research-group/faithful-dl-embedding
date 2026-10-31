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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.config import (
    DATASETS,
    DATASET_TASKS,
    METHODS,
    OBJECTIVE_DATASETS,
    OBJECTIVE_METHODS,
    PRISM_METHODS,
    RAW_RESULTS_DIR,
    RESULTS_DIR,
    STRUCTURAL_METHODS,
    SYNTHETIC_DATASETS,
    TABLES_DIR,
)

STRUCTURAL_MD_PATH: Path = RESULTS_DIR / "STRUCTURAL.md"

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
    ("modelness_train_margin", "Mod@γ"),
    ("helly_violation_rate", "HVR"),
    ("degeneracy_rate", "DegRate"),
    ("n_certificates", "Certs"),
    ("mean_effective_components", "Eff-k"),
    ("mean_epoch_time_s", "Epoch(s)"),
    ("total_train_time_s", "Train(s)"),
    ("peak_gpu_memory_mb", "GPU(MB)"),
]

SATISFACTION_COLUMNS: List[Tuple[str, str]] = [
    ("modelness_all", "Modelness"),
    ("modelness_train_margin", "Mod@γ"),
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
        if p.name.endswith(".failed.json"):
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["_source_file"] = p.name
        payload["_source_mtime"] = float(p.stat().st_mtime)
        records.append(payload)
    return records


_STRUCTURAL_KEYS = frozenset(
    {
        "n_certificates",
        "param_count",
        "n_components",
        "mean_effective_components",
        "mean_effective_components_materialised",
        "mean_epoch_time_s",
        "total_train_time_s",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_bytes",
        "n_defined",
        "reseed_moves_total",
    }
)


def _is_valid_number(v: Any) -> bool:
    """Return ``True`` iff ``v`` is a non-``None``, non-``NaN`` numeric value."""
    if v is None or isinstance(v, bool):
        return False
    try:
        return not bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return False


def _format_row_n(
    cell_runs: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
) -> str:
    """Format the row ``n`` column as ``'k/n'`` (non-null over total runs)."""
    n_total = len(cell_runs)
    if n_total == 0:
        return "0/0"
    eval_keys = [mk for mk in metric_keys if mk not in _STRUCTURAL_KEYS]
    check_keys = eval_keys if eval_keys else list(metric_keys)
    counts = [
        sum(1 for cr in cell_runs if _is_valid_number(cr.get("metrics", {}).get(mk)))
        for mk in check_keys
    ]
    non_zero_counts = [c for c in counts if c > 0]
    k_valid = min(non_zero_counts) if non_zero_counts else 0
    return f"{k_valid}/{n_total}"


def _format_mean_sd(values: Sequence[Optional[float]], decimals: int = 4) -> str:
    """Format numeric values as ``mean ± sd`` (``ddof=1`` when ``k >= 2``, ``mean`` when ``k == 1``, ``—`` when all null).

    When ``0 < k < n`` (some runs are ``None``/``NaN``), appends `` (k/n)`` so
    missing/NaN values are never silently dropped.
    """
    n_total = len(values)
    clean = [float(v) for v in values if _is_valid_number(v)]
    k_valid = len(clean)
    if k_valid == 0:
        return "—"
    mean_v = float(np.mean(clean))
    if k_valid == 1:
        if all(float(v).is_integer() for v in clean) and decimals == 0:
            base = f"{int(round(mean_v))}"
        else:
            base = f"{mean_v:.{decimals}f}"
    else:
        sd_v = float(np.std(clean, ddof=1))
        base = f"{mean_v:.{decimals}f} ± {sd_v:.{decimals}f}"
    if k_valid < n_total:
        return f"{base} ({k_valid}/{n_total})"
    return base


def _decimals_for_key(metric_key: str) -> int:
    """Choose formatting precision for ``metric_key``."""
    if metric_key in ("n_certificates", "param_count"):
        return 0
    if metric_key in ("peak_gpu_memory_mb", "n_defined", "reseed_moves_total"):
        return 1
    if metric_key in (
        "mean_effective_components",
        "mean_effective_components_materialised",
        "total_train_time_s",
    ):
        return 2
    return 4


def _infer_effective_dim(cfg: Dict[str, Any]) -> int:
    """Return ``cfg['effective_dim']`` if present, else infer ``max(d, 3)`` for ``PrismEL`` and ``d`` otherwise."""
    if cfg.get("effective_dim") is not None:
        return int(cfg["effective_dim"])
    d = int(cfg["dim"])
    m_id = str(cfg.get("method", ""))
    model_name = str(cfg.get("model") or METHODS.get(m_id, {}).get("model", ""))
    if model_name == "PrismEL":
        return max(d, 3)
    return d


def _method_suite_label(m_id: str) -> str:
    """Return ``'structural'``, ``'objectives'``, or ``'baseline'`` for ``m_id``."""
    if m_id in STRUCTURAL_METHODS:
        return "structural"
    if m_id in OBJECTIVE_METHODS or m_id in PRISM_METHODS:
        return "objectives"
    return "baseline"


def render_dataset_markdown(
    dataset: str,
    runs: Sequence[Dict[str, Any]],
) -> str:
    """Render the Markdown summary table for ``dataset`` across dimensions and methods."""
    task = DATASET_TASKS.get(dataset, "subsumption")
    cols = SUBSUMPTION_COLUMNS if task == "subsumption" else SATISFACTION_COLUMNS

    by_dim_method: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    dims_seen = set()
    has_nominal_d2_prism = False
    for r in runs:
        cfg = r["config"]
        nom_d = int(cfg["dim"])
        eff_d = _infer_effective_dim(cfg)
        m = str(cfg["method"])
        model_name = str(cfg.get("model") or METHODS.get(m, {}).get("model", ""))
        if model_name == "PrismEL" and nom_d < 3 and eff_d == 3:
            has_nominal_d2_prism = True
        dims_seen.add(eff_d)
        by_dim_method[(eff_d, m)].append(r)

    sorted_dims = sorted(dims_seen)
    method_order = list(METHODS.keys())
    for _, m in sorted(by_dim_method.keys()):
        if m not in method_order:
            method_order.append(m)

    lines: List[str] = [
        f"# Results: `{dataset}` ({task})",
        "",
    ]
    if has_nominal_d2_prism:
        lines.append(
            "- Note: prism needs ≥ 3 dims; nominal d=2 cells trained at d=3."
        )
        lines.append("")

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
            n_seeds = _format_row_n(cell_runs, [mk for mk, _ in cols])

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
        "| Dataset | d | Method | k | n | Modelness | Mod@γ | HVR | DegRate | Eff-k | F-MRR | F-H@10 | AUC(+/-) | Pair-FVR | FVR | Train(s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    curve_keys = [
        ("modelness_all", 4),
        ("modelness_train_margin", 4),
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
            n_seeds = _format_row_n(cell_runs, [mk for mk, _ in curve_keys])
            row = [f"`{ds}`", str(dim), f"`{m_id}`", k_val, n_seeds]
            for m_key, dec in curve_keys:
                vals = [cr["metrics"].get(m_key) for cr in cell_runs]
                row.append(_format_mean_sd(vals, decimals=dec))
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def _format_point_modelness(cell_runs: Sequence[Dict[str, Any]], decimals: int = 4) -> str:
    """Format ``point_modelness`` ``(pos_frac / hardneg_frac)`` across ``cell_runs``."""
    pos_vals: List[Optional[float]] = []
    neg_vals: List[Optional[float]] = []
    for cr in cell_runs:
        pm = cr.get("metrics", {}).get("point_modelness")
        if isinstance(pm, dict):
            pos_vals.append(pm.get("pos_frac"))
            neg_vals.append(pm.get("hardneg_frac"))
        else:
            pos_vals.append(None)
            neg_vals.append(None)
    if not any(_is_valid_number(v) for v in pos_vals) and not any(
        _is_valid_number(v) for v in neg_vals
    ):
        return "—"
    return f"{_format_mean_sd(pos_vals, decimals=decimals)} / {_format_mean_sd(neg_vals, decimals=decimals)}"


def render_objectives_markdown(
    dataset: str,
    runs: Sequence[Dict[str, Any]],
) -> str:
    """Render ``objectives_<dataset>.md`` comparing objectives, backbones, k, lr, and prisms."""
    grouped: Dict[Tuple[int, float, str], List[Dict[str, Any]]] = defaultdict(list)
    dims_seen = set()
    lrs_seen = set()
    has_nominal_d2_prism = False
    dirty_exp = any(bool(r.get("env", {}).get("git_dirty_experiments")) for r in runs)
    dirty_mowl = any(bool(r.get("env", {}).get("git_dirty_mowl")) for r in runs)
    exp_hashes = sorted(
        {str(r.get("env", {}).get("git_hash_experiments", "unknown"))[:7] for r in runs if r.get("env")}
    )
    mowl_hashes = sorted(
        {str(r.get("env", {}).get("git_hash_mowl", "unknown"))[:7] for r in runs if r.get("env")}
    )

    for r in runs:
        cfg = r["config"]
        nom_d = int(cfg["dim"])
        eff_d = _infer_effective_dim(cfg)
        lr = float(cfg.get("lr", cfg.get("learning_rate", 0.001)))
        m_id = str(cfg["method"])
        model_name = str(cfg.get("model") or METHODS.get(m_id, {}).get("model", ""))
        if model_name == "PrismEL" and nom_d < 3 and eff_d == 3:
            has_nominal_d2_prism = True
        dims_seen.add(eff_d)
        lrs_seen.add(lr)
        grouped[(eff_d, lr, m_id)].append(r)

    method_order = list(METHODS.keys())
    for _, _, m_id in sorted(grouped.keys()):
        if m_id not in method_order:
            method_order.append(m_id)

    exp_flag = "dirty" if dirty_exp else "clean"
    mowl_flag = "dirty" if dirty_mowl else "clean"
    lines: List[str] = [
        f"# Objectives & Prisms: `{dataset}`",
        "",
        f"- Provenance: `experiments` `{','.join(exp_hashes) or 'n/a'}` ({exp_flag}), "
        f"`mowl` `{','.join(mowl_hashes) or 'n/a'}` ({mowl_flag}).",
        "- Margin semantics: `Mod strict` evaluates GCI losses at `margin = 0`; `Mod@γ` evaluates at the training margin `γ` (`relaxing` for Box²EL/MultiBoxEL, `tightening` for ELBE/MultiELBE/PrismEL).",
        "- Suite membership: the `suite` column distinguishes `objectives` methods (30 objective-grid methods + `prism`/`prism-init`) from `baseline` 10-29 methods.",
    ]
    if has_nominal_d2_prism:
        lines.append(
            "- Prism dimension note (*): prism needs ≥ 3 dims; nominal d=2 cells trained at d=3."
        )
    lines.extend(
        [
            "",
            "| method | suite | backbone | k | objective | lr | d | Mod strict | Mod@γ | point-Mod (pos/hardneg) | HVR | Deg | eff-k | seeds n/k |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for dim in sorted(dims_seen):
        for lr in sorted(lrs_seen):
            for m_id in method_order:
                cell_runs = grouped.get((dim, lr, m_id), [])
                if not cell_runs:
                    continue
                cfg0 = cell_runs[0]["config"]
                spec = METHODS.get(m_id, {})
                suite_label = _method_suite_label(m_id)
                backbone = str(
                    cfg0.get("backbone")
                    or spec.get("backbone")
                    or cfg0.get("model", "—")
                )
                k_val = str(
                    cfg0.get("n_components", spec.get("n_components", "—"))
                )
                obj_val = str(
                    cfg0.get("objective_variant")
                    or spec.get("objective_variant")
                    or cfg0.get("objective", "axiom")
                )
                lr_str = f"{lr:g}"
                d_str = str(dim)

                mod_strict = _format_mean_sd(
                    [cr["metrics"].get("modelness_all") for cr in cell_runs],
                    decimals=4,
                )
                mod_gamma = _format_mean_sd(
                    [cr["metrics"].get("modelness_train_margin") for cr in cell_runs],
                    decimals=4,
                )
                pmod_str = _format_point_modelness(cell_runs, decimals=4)
                hvr_str = _format_mean_sd(
                    [cr["metrics"].get("helly_violation_rate") for cr in cell_runs],
                    decimals=4,
                )
                deg_str = _format_mean_sd(
                    [cr["metrics"].get("degeneracy_rate") for cr in cell_runs],
                    decimals=4,
                )
                eff_k_str = _format_mean_sd(
                    [cr["metrics"].get("mean_effective_components") for cr in cell_runs],
                    decimals=2,
                )
                seeds_nk = _format_row_n(
                    cell_runs,
                    ["modelness_all", "helly_violation_rate", "degeneracy_rate"],
                )

                row = [
                    f"`{m_id}`",
                    suite_label,
                    backbone,
                    k_val,
                    obj_val,
                    lr_str,
                    d_str,
                    mod_strict,
                    mod_gamma,
                    pmod_str,
                    hvr_str,
                    deg_str,
                    eff_k_str,
                    seeds_nk,
                ]
                lines.append("| " + " | ".join(row) + " |")

    if has_nominal_d2_prism:
        lines.append("")
        lines.append("*Footnote: prism needs ≥ 3 dims; nominal d=2 cells trained at d=3.*")
    lines.append("")
    return "\n".join(lines)


def _is_structural_seed_pass(run_rec: Dict[str, Any]) -> bool:
    """Return ``True`` iff a single run satisfies ``HVR == 0``, ``Deg == 0``, ``existence_violation_rate == 0``, and ``modelness_all >= 0.99``."""
    m = run_rec.get("metrics", {})
    hvr = m.get("helly_violation_rate")
    deg = m.get("degeneracy_rate")
    mod = m.get("modelness_all")
    evr = m.get("existence_violation_rate")
    if (
        not _is_valid_number(hvr)
        or not _is_valid_number(deg)
        or not _is_valid_number(mod)
    ):
        return False
    if "existence_violation_rate" in m and evr is not None:
        if not _is_valid_number(evr) or float(evr) > 1e-12:
            return False
    return (
        float(hvr) <= 1e-12
        and float(deg) <= 1e-12
        and float(mod) >= 0.99 - 1e-12
    )


def _dedupe_by_seed_lr(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate runs within a ``(dataset, dim, method)`` cell by ``(seed, lr)``, keeping the newest."""
    by_key: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for idx, r in enumerate(runs):
        cfg = r.get("config", {})
        seed = int(cfg.get("seed", idx))
        lr = float(cfg.get("lr", cfg.get("learning_rate", 0.001)))
        key = (seed, lr)
        if key not in by_key:
            by_key[key] = r
        else:
            prev = by_key[key]
            cur_mtime = float(r.get("_source_mtime", 0.0))
            prev_mtime = float(prev.get("_source_mtime", 0.0))
            if cur_mtime >= prev_mtime:
                by_key[key] = r
    return [by_key[k] for k in sorted(by_key.keys())]


def structural_table(
    runs_or_raw: Union[Sequence[Dict[str, Any]], Path, str] = RAW_RESULTS_DIR,
    out_path: Optional[Union[Path, str]] = None,
    *,
    raw_dir: Optional[Union[Path, str]] = None,
) -> str:
    """Render and write ``STRUCTURAL.md`` from raw JSON runs of structural methods.

    Per ``(dataset, d, method)`` for synthetic datasets:
    - ``Pass``: ``PASS`` iff 3/3 seeds achieve ``HVR = 0``, ``Deg = 0``,
      ``existence_violation_rate = 0``, and strict ``Mod >= 0.99`` at the best ``lr``,
      else ``FAIL``.
    - ``Success (all lr)``: per-seed success fraction over all deduplicated ``(seed, lr)`` cells.
    - ``Best lr``: learning rate maximizing passing seeds (ties broken by lowest
      mean HVR, highest mean strict Mod, then smaller lr).
    - At ``best_lr``: mean strict ``Mod``, mean ``HVR``, mean ``Deg``, mean
      ``ExistViol``, mean materialised effective ``k`` (``Eff-k (mat.)``), mean
      epoch time, mean ``n_defined``, and mean ``reseed_moves_total``.
    Plus a Pizza sanity section when Pizza JSONs exist for structural methods.
    """
    if raw_dir is not None:
        runs = load_raw_runs(Path(raw_dir).expanduser().resolve())
    elif isinstance(runs_or_raw, (Path, str)):
        runs = load_raw_runs(Path(runs_or_raw).expanduser().resolve())
    else:
        runs = list(runs_or_raw)

    struct_set = set(STRUCTURAL_METHODS)
    syn_set = set(SYNTHETIC_DATASETS)
    struct_runs = [
        r
        for r in runs
        if str(r.get("config", {}).get("method", "")) in struct_set
        or str(r.get("config", {}).get("objective", "")) == "structural"
    ]

    syn_grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    pizza_grouped: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
    has_nominal_d2_prism = False

    for r in struct_runs:
        cfg = r.get("config", {})
        ds = str(cfg.get("dataset", ""))
        m_id = str(cfg.get("method", ""))
        nom_d = int(cfg.get("dim", 0))
        eff_d = _infer_effective_dim(cfg)
        lr = float(cfg.get("lr", cfg.get("learning_rate", 0.001)))
        model_name = str(cfg.get("model") or METHODS.get(m_id, {}).get("model", ""))
        if ds == "pizza":
            pizza_grouped[(m_id, nom_d, lr)].append(r)
        elif ds in syn_set:
            if model_name == "PrismEL" and nom_d < 3 and eff_d == 3:
                has_nominal_d2_prism = True
            syn_grouped[(ds, nom_d, m_id)].append(r)

    method_order = list(STRUCTURAL_METHODS)
    for _, _, m_id in sorted(syn_grouped.keys()):
        if m_id not in method_order:
            method_order.append(m_id)
    for m_id, _, _ in sorted(pizza_grouped.keys()):
        if m_id not in method_order:
            method_order.append(m_id)

    dataset_order = {d: i for i, d in enumerate(DATASETS)}
    ds_dim_pairs = sorted(
        {(ds, d) for ds, d, _ in syn_grouped.keys()},
        key=lambda pair: (dataset_order.get(pair[0], 999), pair[0], pair[1]),
    )

    lines: List[str] = [
        "# Structural Losses Results (`objective = 'structural'`)",
        "",
        "- Pass criterion: 3/3 seeds with `HVR = 0`, `Deg = 0`, `ExistViol = 0`, and strict `Mod ≥ 0.99` at the best `lr`.",
        "- `Success (all lr)` reports the fraction of `(seed, lr)` cells satisfying `HVR = 0 ∧ Deg = 0 ∧ ExistViol = 0 ∧ Mod ≥ 0.99`.",
    ]
    if has_nominal_d2_prism:
        lines.append(
            "- Note: `prism-struct` needs ≥ 3 dims; nominal `d = 2` cells are trained at `effective_dim = 3`."
        )
    lines.extend(
        [
            "",
            "## Synthetic Benchmarks (`tk3`–`tk6`)",
            "",
            "| Dataset | d | Method | Pass | Success (all lr) | Best lr | Mod strict | HVR | Deg | ExistViol | Eff-k (mat.) | Epoch(s) | n_defined | reseed_moves_total |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for ds, dim in ds_dim_pairs:
        for m_id in method_order:
            raw_cell_runs = syn_grouped.get((ds, dim, m_id), [])
            if not raw_cell_runs:
                continue
            cell_runs = _dedupe_by_seed_lr(raw_cell_runs)
            n_tot = len(cell_runs)
            n_succ = sum(1 for r in cell_runs if _is_structural_seed_pass(r))
            succ_str = f"{n_succ}/{n_tot} ({n_succ / n_tot:.2f})" if n_tot > 0 else "0/0"

            by_lr: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
            for r in cell_runs:
                cfg_r = r.get("config", {})
                lr_r = float(cfg_r.get("lr", cfg_r.get("learning_rate", 0.001)))
                by_lr[lr_r].append(r)

            def _lr_rank_key(lr_val: float) -> Tuple[int, float, float, float]:
                lr_runs = by_lr[lr_val]
                pass_seeds = {
                    int(rr.get("config", {}).get("seed", idx))
                    for idx, rr in enumerate(lr_runs)
                    if _is_structural_seed_pass(rr)
                }
                hvrs = [
                    float(rr["metrics"]["helly_violation_rate"])
                    for rr in lr_runs
                    if _is_valid_number(rr.get("metrics", {}).get("helly_violation_rate"))
                ]
                mods = [
                    float(rr["metrics"]["modelness_all"])
                    for rr in lr_runs
                    if _is_valid_number(rr.get("metrics", {}).get("modelness_all"))
                ]
                mean_hvr = float(np.mean(hvrs)) if hvrs else float("inf")
                mean_mod = float(np.mean(mods)) if mods else float("-inf")
                return (len(pass_seeds), -mean_hvr, mean_mod, -lr_val)

            best_lr = max(by_lr.keys(), key=_lr_rank_key)
            best_runs = by_lr[best_lr]
            pass_seeds_best = {
                int(rr.get("config", {}).get("seed", idx))
                for idx, rr in enumerate(best_runs)
                if _is_structural_seed_pass(rr)
            }
            passed = len(pass_seeds_best) >= 3 and len(pass_seeds_best) == len(best_runs)
            pass_flag = "PASS" if passed else "FAIL"

            mod_str = _format_mean_sd(
                [rr.get("metrics", {}).get("modelness_all") for rr in best_runs],
                decimals=4,
            )
            hvr_str = _format_mean_sd(
                [rr.get("metrics", {}).get("helly_violation_rate") for rr in best_runs],
                decimals=4,
            )
            deg_str = _format_mean_sd(
                [rr.get("metrics", {}).get("degeneracy_rate") for rr in best_runs],
                decimals=4,
            )
            exist_str = _format_mean_sd(
                [rr.get("metrics", {}).get("existence_violation_rate") for rr in best_runs],
                decimals=4,
            )
            eff_k_vals = [
                rr.get("metrics", {}).get(
                    "mean_effective_components_materialised",
                    rr.get("metrics", {}).get("mean_effective_components"),
                )
                for rr in best_runs
            ]
            eff_k_str = _format_mean_sd(eff_k_vals, decimals=2)
            ep_vals = [
                rr.get("metrics", {}).get(
                    "mean_epoch_time_s",
                    rr.get("timings", {}).get("mean_epoch_time_s"),
                )
                for rr in best_runs
            ]
            ep_str = _format_mean_sd(ep_vals, decimals=4)
            ndef_str = _format_mean_sd(
                [rr.get("metrics", {}).get("n_defined") for rr in best_runs],
                decimals=1,
            )
            reseed_str = _format_mean_sd(
                [rr.get("metrics", {}).get("reseed_moves_total") for rr in best_runs],
                decimals=1,
            )

            row = [
                f"`{ds}`",
                str(dim),
                f"`{m_id}`",
                pass_flag,
                succ_str,
                f"{best_lr:g}",
                mod_str,
                hvr_str,
                deg_str,
                exist_str,
                eff_k_str,
                ep_str,
                ndef_str,
                reseed_str,
            ]
            lines.append("| " + " | ".join(row) + " |")

    if pizza_grouped:
        lines.extend(
            [
                "",
                "## Pizza Sanity (`pizza`)",
                "",
                "| Method | d | lr | n | F-MRR | AUC(+/-) | FVR | Mod strict | Epoch(s) | n_defined | reseed_moves_total |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        pizza_keys = sorted(
            pizza_grouped.keys(),
            key=lambda item: (
                method_order.index(item[0]) if item[0] in method_order else 999,
                item[0],
                item[1],
                item[2],
            ),
        )
        for m_id, dim, lr in pizza_keys:
            p_runs = pizza_grouped[(m_id, dim, lr)]
            n_seeds = _format_row_n(
                p_runs, ["rank_f_mrr", "auc_pos_neg", "fvr", "modelness_all"]
            )
            fmrr_str = _format_mean_sd(
                [rr.get("metrics", {}).get("rank_f_mrr") for rr in p_runs], decimals=4
            )
            auc_str = _format_mean_sd(
                [rr.get("metrics", {}).get("auc_pos_neg") for rr in p_runs], decimals=4
            )
            fvr_str = _format_mean_sd(
                [rr.get("metrics", {}).get("fvr") for rr in p_runs], decimals=4
            )
            mod_str = _format_mean_sd(
                [rr.get("metrics", {}).get("modelness_all") for rr in p_runs], decimals=4
            )
            ep_vals = [
                rr.get("metrics", {}).get(
                    "mean_epoch_time_s",
                    rr.get("timings", {}).get("mean_epoch_time_s"),
                )
                for rr in p_runs
            ]
            ep_str = _format_mean_sd(ep_vals, decimals=4)
            ndef_str = _format_mean_sd(
                [rr.get("metrics", {}).get("n_defined") for rr in p_runs], decimals=1
            )
            reseed_str = _format_mean_sd(
                [rr.get("metrics", {}).get("reseed_moves_total") for rr in p_runs],
                decimals=1,
            )
            p_row = [
                f"`{m_id}`",
                str(dim),
                f"{lr:g}",
                n_seeds,
                fmrr_str,
                auc_str,
                fvr_str,
                mod_str,
                ep_str,
                ndef_str,
                reseed_str,
            ]
            lines.append("| " + " | ".join(p_row) + " |")

    lines.append("")
    md_text = "\n".join(lines)

    resolved_out: Optional[Path] = None
    if out_path is not None:
        resolved_out = Path(out_path).expanduser().resolve()
    elif isinstance(runs_or_raw, (Path, str)) and Path(runs_or_raw).expanduser().resolve() == RAW_RESULTS_DIR.resolve():
        resolved_out = STRUCTURAL_MD_PATH.resolve()

    if resolved_out is not None:
        resolved_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_out.write_text(md_text, encoding="utf-8")

    return md_text


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
        "objective",
        "objective_variant",
        "component_aggregation",
        "tau_start",
        "tau_end",
        "tau_anneal",
        "diversity_reg",
        "lr",
        "learning_rate",
        "dim",
        "effective_dim",
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
        "git_dirty_experiments",
        "git_dirty_mowl",
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
                if k == "effective_dim" and cfg.get("effective_dim") is None and "dim" in cfg:
                    row[k] = _infer_effective_dim(cfg)
                else:
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
    obj_method_set = set(OBJECTIVE_METHODS) | set(PRISM_METHODS)
    for ds in DATASETS:
        if ds not in by_dataset:
            continue
        md_path = out_dir / f"{ds}.md"
        md_path.write_text(
            render_dataset_markdown(ds, by_dataset[ds]),
            encoding="utf-8",
        )
        written_tables.append(md_path)
        if ds in OBJECTIVE_DATASETS or any(
            str(r.get("config", {}).get("method")) in obj_method_set
            for r in by_dataset[ds]
        ):
            obj_path = out_dir / f"objectives_{ds}.md"
            obj_path.write_text(
                render_objectives_markdown(ds, by_dataset[ds]),
                encoding="utf-8",
            )
            written_tables.append(obj_path)

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

    struct_set = set(STRUCTURAL_METHODS)
    if any(
        str(r.get("config", {}).get("method", "")) in struct_set
        or str(r.get("config", {}).get("objective", "")) == "structural"
        for r in runs
    ):
        struct_path = (
            STRUCTURAL_MD_PATH.resolve()
            if out_dir == TABLES_DIR.resolve()
            else (out_dir.parent / "STRUCTURAL.md").resolve()
        )
        structural_table(runs, struct_path)
        written_tables.append(struct_path)

    write_all_runs_csv(runs, csv_path)
    print(
        f"[aggregate] Aggregated {len(runs)} runs from {raw_dir} -> "
        f"{len(written_tables)} Markdown tables in {out_dir} and CSV {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
