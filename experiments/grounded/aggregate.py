#!/usr/bin/env python3
"""Aggregate GroundedEL E1 synthetic cell JSONs into ``experiments/results/GROUNDED.md`` (design §9 E1).

CLI usage
---------
::

    python -m experiments.grounded.aggregate \\
        [--in-dir /mnt/data1/gfe/runs/grounded-e1] \\
        [--out experiments/results/GROUNDED.md]

Guarantees
----------
- Loads all ``*.json`` cell outputs in ``--in-dir`` and groups them by
  ``(theory, K, dim)``.
- Produces a Markdown report with per-``(theory, K, d)`` rows reporting:
  seed count ``n``, ``exact`` rate, mean (± sd when ``n >= 2``) ``elem_viol``,
  ``region_viol``, ``exist``, ``dead_ok`` rate, ``faithful_named_precision``,
  ``faithful_named_recall``, and ``wall_time_s``.
- Writes ``experiments/results/GROUNDED.md`` atomically via ``os.replace``.

Non-guarantees
--------------
- If ``--in-dir`` contains no ``*.json`` files, writes a valid Markdown header
  and empty table without raising.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.config import RESULTS_DIR
from experiments.grounded.run_synthetic import DEFAULT_E1_OUT_DIR
from experiments.grounded.theories import THEORIES

DEFAULT_GROUNDED_MD_PATH: Path = RESULTS_DIR / "GROUNDED.md"
_K_ORDER: Tuple[str, ...] = ("1", "2", "4", "8", "full")


def load_grounded_runs(in_dir: Path | str = DEFAULT_E1_OUT_DIR) -> List[Dict[str, Any]]:
    """Load all ``*.json`` run records from ``in_dir``, deduplicated by ``(theory, K, Kr, dim, reseed_every, seed)``.

    Parameters
    ----------
    in_dir : Path or str, default ``/mnt/data1/gfe/runs/grounded-e1``
        Directory containing E1 JSON files.

    Returns
    -------
    List[Dict[str, Any]]
        Loaded cell dictionaries.

    Guarantees
    ----------
    - Skips hidden temporary files (``.*.tmp``) and ``*.failed.json`` files.
    - Returns ``[]`` if ``in_dir`` does not exist.

    Non-guarantees
    --------------
    - Raises :class:`json.JSONDecodeError` if a non-temporary ``*.json`` file is
      malformed.
    """
    p_dir = Path(in_dir).expanduser().resolve()
    if not p_dir.exists():
        return []

    by_cell_key: Dict[Tuple[str, str, str, int, int, int], Dict[str, Any]] = {}
    for p in sorted(p_dir.glob("*.json")):
        if p.name.startswith(".") or p.name.endswith(".failed.json"):
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        if "theory" not in cfg or "K" not in cfg or "dim" not in cfg:
            continue
        key = (
            str(cfg["theory"]),
            str(cfg["K"]),
            str(cfg.get("Kr", cfg["K"])),
            int(cfg["dim"]),
            int(cfg.get("reseed_every", 0)),
            int(cfg.get("seed", 0)),
        )
        payload["_source_file"] = p.name
        by_cell_key[key] = payload
    return list(by_cell_key.values())


def _k_sort_key(k_str: str) -> Tuple[int, int, str]:
    """Sort numeric ``K`` values in ascending order followed by ``'full'``."""
    s = str(k_str).strip().lower()
    if s == "full":
        return (1, 10**9, s)
    try:
        return (0, int(s), s)
    except ValueError:
        return (2, 0, s)


def _theory_sort_key(theory: str) -> Tuple[int, str]:
    """Order theories according to ``THEORIES``, then alphabetically."""
    if theory in THEORIES:
        return (THEORIES.index(theory), theory)
    return (len(THEORIES), theory)


def _format_mean_sd(
    vals: Sequence[Optional[float]],
    decimals: int = 4,
) -> str:
    """Format numeric values as ``mean`` (when ``n == 1``) or ``mean ± sd`` (when ``n >= 2``)."""
    clean = [
        float(v)
        for v in vals
        if v is not None and not isinstance(v, bool) and not np.isnan(float(v))
    ]
    if not clean:
        return "—"
    mean_v = float(np.mean(clean))
    if len(clean) == 1:
        return f"{mean_v:.{decimals}f}"
    sd_v = float(np.std(clean, ddof=1))
    return f"{mean_v:.{decimals}f} ± {sd_v:.{decimals}f}"


def _format_bool_rate(flags: Sequence[Optional[bool]]) -> str:
    """Format boolean flags as ``rate (k/n)``."""
    clean = [bool(f) for f in flags if f is not None]
    if not clean:
        return "—"
    n_true = sum(1 for f in clean if f)
    n_tot = len(clean)
    rate = n_true / n_tot
    return f"{rate:.2f} ({n_true}/{n_tot})"


def summarize_cell_group(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregated statistics for a ``(theory, K, dim)`` group across seeds.

    Parameters
    ----------
    runs : Sequence[Dict[str, Any]]
        Run records sharing ``(theory, K, dim)``.

    Returns
    -------
    Dict[str, Any]
        Aggregated statistics dictionary.

    Guarantees
    ----------
    - Computes exact rate, mean ``elem_viol``, ``region_viol``, ``exist``,
      ``dead_ok`` rate, ``faithful_named_precision``, ``faithful_named_recall``,
      and ``wall_time_s``.

    Non-guarantees
    --------------
    - Requires ``runs`` to be non-empty.
    """
    if not runs:
        raise ValueError("runs must be non-empty")

    cfg0 = runs[0]["config"]
    theory = str(cfg0["theory"])
    k_str = str(cfg0["K"])
    eff_k = int(cfg0.get("n_components", 0))
    dim = int(cfg0["dim"])
    reseed_every = int(cfg0.get("reseed_every", 0))
    n_seeds = len(runs)

    exact_flags = [bool(r["metrics"].get("exact", False)) for r in runs]
    exact_tol1e4_flags = [
        bool(r["metrics"].get("exact_tol1e-4", r["metrics"].get("exact", False)))
        for r in runs
    ]
    dead_flags = [bool(r["metrics"].get("dead_ok", False)) for r in runs]
    elem_viols = [float(r["metrics"]["elem_viol"]) for r in runs]
    region_viols = [float(r["metrics"]["region_viol"]) for r in runs]
    region_viols_tol1e4 = [
        float(r["metrics"].get("region_viol_tol1e-4", r["metrics"]["region_viol"]))
        for r in runs
    ]
    exists = [float(r["metrics"]["exist"]) for r in runs]
    n_false_deads = [
        r["metrics"].get("n_false_dead", r.get("n_false_dead")) for r in runs
    ]
    n_empty_classes_list = [
        r["metrics"].get(
            "n_empty_classes",
            r.get("crisp_report", {}).get(
                "n_empty_classes", r.get("n_empty_classes")
            ),
        )
        for r in runs
    ]
    faith_precs = [
        r["metrics"].get("faithful_named_precision") for r in runs
    ]
    faith_recs = [
        r["metrics"].get("faithful_named_recall") for r in runs
    ]
    wall_times = [
        float(r.get("wall_time_s", r.get("timings", {}).get("wall_time_s", 0.0)))
        for r in runs
    ]

    clean_precs = [float(v) for v in faith_precs if v is not None]
    clean_recs = [float(v) for v in faith_recs if v is not None]
    clean_false_deads = [float(v) for v in n_false_deads if v is not None]
    clean_empty_classes = [float(v) for v in n_empty_classes_list if v is not None]

    return {
        "theory": theory,
        "K": k_str,
        "n_components": eff_k,
        "dim": dim,
        "reseed_every": reseed_every,
        "n": n_seeds,
        "exact_rate": float(np.mean(exact_flags)),
        "exact_count": int(sum(exact_flags)),
        "exact_tol1e-4_rate": float(np.mean(exact_tol1e4_flags)),
        "exact_tol1e-4_count": int(sum(exact_tol1e4_flags)),
        "elem_viol_mean": float(np.mean(elem_viols)),
        "region_viol_mean": float(np.mean(region_viols)),
        "region_viol_tol1e-4_mean": float(np.mean(region_viols_tol1e4)),
        "exist_mean": float(np.mean(exists)),
        "dead_ok_rate": float(np.mean(dead_flags)),
        "dead_ok_count": int(sum(dead_flags)),
        "n_false_dead_mean": (
            float(np.mean(clean_false_deads)) if clean_false_deads else None
        ),
        "n_empty_classes_mean": (
            float(np.mean(clean_empty_classes)) if clean_empty_classes else None
        ),
        "faithful_precision_mean": (
            float(np.mean(clean_precs)) if clean_precs else None
        ),
        "faithful_recall_mean": (
            float(np.mean(clean_recs)) if clean_recs else None
        ),
        "wall_time_mean_s": float(np.mean(wall_times)),
        "formatted": {
            "exact_rate": _format_bool_rate(exact_flags),
            "exact_tol1e-4/3": _format_bool_rate(exact_tol1e4_flags),
            "exact_tol1e-4_rate": _format_bool_rate(exact_tol1e4_flags),
            "elem_viol": _format_mean_sd(elem_viols, decimals=4),
            "region_viol": _format_mean_sd(region_viols, decimals=4),
            "region_viol_tol1e-4": _format_mean_sd(region_viols_tol1e4, decimals=4),
            "exist": _format_mean_sd(exists, decimals=4),
            "dead_ok": _format_bool_rate(dead_flags),
            "n_false_dead": _format_mean_sd(n_false_deads, decimals=2),
            "n_empty_classes": _format_mean_sd(n_empty_classes_list, decimals=2),
            "faithful_precision": _format_mean_sd(faith_precs, decimals=4),
            "faithful_recall": _format_mean_sd(faith_recs, decimals=4),
            "wall_time_s": _format_mean_sd(wall_times, decimals=2),
        },
    }


def render_grounded_markdown(runs: Sequence[Dict[str, Any]]) -> str:
    """Render the Markdown summary table per ``theory × K × d × reseed`` for ``GROUNDED.md``.

    Parameters
    ----------
    runs : Sequence[Dict[str, Any]]
        Loaded E1 cell JSON records.

    Returns
    -------
    str
        Formatted Markdown document.

    Guarantees
    ----------
    - Includes columns for ``Theory``, ``K``, ``eff_K``, ``d``, ``reseed``, ``n``,
      ``Exact Rate``, ``exact_tol1e-4/3``, ``elem_viol``, ``region_viol``,
      ``region_viol_tol1e-4``, ``exist``, ``dead_ok``, ``Faithful Prec``,
      ``Faithful Rec``, and ``Wall Time (s)``.

    Non-guarantees
    --------------
    - Does not write to disk.
    """
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        cfg = r["config"]
        key = (
            str(cfg["theory"]),
            str(cfg["K"]),
            int(cfg["dim"]),
            int(cfg.get("reseed_every", 0)),
        )
        grouped[key].append(r)

    sorted_keys = sorted(
        grouped.keys(),
        key=lambda t: (_theory_sort_key(t[0]), t[2], _k_sort_key(t[1]), t[3]),
    )

    headers = [
        "Theory",
        "K",
        "eff_K",
        "d",
        "reseed",
        "n",
        "Exact Rate",
        "exact_tol1e-4/3",
        "elem_viol",
        "region_viol",
        "region_viol_tol1e-4",
        "exist",
        "dead_ok",
        "n_false_dead",
        "n_empty_classes",
        "Faithful Prec",
        "Faithful Rec",
        "Wall Time (s)",
    ]
    align = [
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
    ]

    lines: List[str] = [
        "# GroundedEL Synthetic Evaluation (`GROUNDED.md`, Design §9 E1)",
        "",
        f"Total cells loaded: **{len(runs)}**",
        "",
        "## Summary Table (`theory × K × d × reseed`)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for key in sorted_keys:
        cell_runs = sorted(grouped[key], key=lambda r: int(r["config"].get("seed", 0)))
        summary = summarize_cell_group(cell_runs)
        fmt = summary["formatted"]
        row = [
            f"`{summary['theory']}`",
            str(summary["K"]),
            str(summary["n_components"]),
            str(summary["dim"]),
            str(summary["reseed_every"]),
            str(summary["n"]),
            fmt["exact_rate"],
            fmt["exact_tol1e-4/3"],
            fmt["elem_viol"],
            fmt["region_viol"],
            fmt["region_viol_tol1e-4"],
            fmt["exist"],
            fmt["dead_ok"],
            fmt["n_false_dead"],
            fmt["n_empty_classes"],
            fmt["faithful_precision"],
            fmt["faithful_recall"],
            fmt["wall_time_s"],
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def write_grounded_markdown(
    in_dir: Path | str = DEFAULT_E1_OUT_DIR,
    out_path: Path | str = DEFAULT_GROUNDED_MD_PATH,
) -> Path:
    """Load E1 runs from ``in_dir`` and write the aggregated Markdown table to ``out_path``.

    Parameters
    ----------
    in_dir : Path or str, default ``/mnt/data1/gfe/runs/grounded-e1``
        Directory of E1 cell JSONs.
    out_path : Path or str, default ``experiments/results/GROUNDED.md``
        Destination Markdown path.

    Returns
    -------
    Path
        Resolved path to the written Markdown file.

    Guarantees
    ----------
    - Writes atomically via a temporary file and ``os.replace``.
    - Never writes under ``/tmp``.

    Non-guarantees
    --------------
    - Overwrites ``out_path`` if it already exists.
    """
    runs = load_grounded_runs(in_dir=in_dir)
    md = render_grounded_markdown(runs)
    target = Path(out_path).expanduser().resolve()
    if str(target).startswith("/tmp/"):
        raise ValueError(f"Refusing to write under /tmp: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.tmp"
    tmp_path.write_text(md, encoding="utf-8")
    os.replace(tmp_path, target)
    return target


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """CLI entry point for ``experiments/grounded/aggregate.py``."""
    parser = argparse.ArgumentParser(
        description="Aggregate GroundedEL E1 synthetic JSON runs into GROUNDED.md."
    )
    parser.add_argument(
        "--in-dir",
        "--raw",
        dest="in_dir",
        type=Path,
        default=DEFAULT_E1_OUT_DIR,
        help="Directory containing E1 cell JSON files (default: /mnt/data1/gfe/runs/grounded-e1).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_GROUNDED_MD_PATH,
        help="Output Markdown path (default: experiments/results/GROUNDED.md).",
    )
    args = parser.parse_args(argv)
    written = write_grounded_markdown(in_dir=args.in_dir, out_path=args.out)
    print(f"Wrote aggregated GroundedEL table to {written}")
    return written


if __name__ == "__main__":
    main()
