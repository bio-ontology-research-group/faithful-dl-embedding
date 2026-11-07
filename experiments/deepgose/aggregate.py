#!/usr/bin/env python3
"""Aggregate E3 evaluation JSONs (``eval_{model}_{combine}_{n}_ids{..}.json``) into
``experiments/results/DEEPGOSE.md``.

CLI usage
---------
::

    python -m experiments.deepgose.aggregate [--in-dir DIR] [--out experiments/results/DEEPGOSE.md]

``--in-dir`` holds one sub-directory per ontology (``cc/``, ``mf/``, ``bp/``) with the
evaluation JSONs written by ``experiments.deepgose.evaluate`` (and optionally the
``metrics_*.json`` training records, used for the best-epoch column).

Guarantees
----------
- Single-model rows (``n_models_used == 1``) are grouped by ``(ont, model)`` and reported
  as mean ± sd over seeds (sd only when ``n >= 2``).
- Ensemble rows (``n_models_used >= 2``) are reported individually with their combine rule.
- Columns: Fmax raw / propagated, Smin propagated, AUPR propagated, mean per-term AUC,
  hierarchy violation rate and ancestor-closed rate of the raw scores, and for GroundedEL
  checkpoints the GO-box crisp report (elem_viol, region_viol, region_viol_tol1e-4,
  n_empty_classes, n_false_dead).
- Writes the Markdown atomically via ``os.replace``.

Non-guarantees
--------------
- Files that do not match the ``_ids`` naming (legacy ``eval_{model}_{combine}_{n}.json``)
  are ignored, because their model ids are ambiguous.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.config import RESULTS_DIR

DEFAULT_IN_DIR = Path(os.environ.get("GFE_DEEPGOSE_RESULTS", RESULTS_DIR / "raw-deepgose"))
DEFAULT_OUT = RESULTS_DIR / "DEEPGOSE.md"
FNAME_RE = re.compile(r"^eval_(?P<model>.+)_(?P<combine>avg|min|max)_(?P<n>\d+)_ids(?P<ids>[\d-]+)\.json$")
MODEL_ORDER = [
    "mlp_esm", "deepgozero_esm", "grounded_esm_l0", "grounded_esm",
    "grounded_esm_tau8", "grounded_esm_pre16",
]
MODEL_LABEL = {
    "mlp_esm": "MLP (ESM2), BCE only",
    "deepgozero_esm": "DeepGOZero (ESM2) = MLP + ELEmbeddings loss",
    "grounded_esm_l0": "GroundedEL head, λ_tbox = 0 (box head, no TBox)",
    "grounded_esm": "GroundedEL (ESM2), λ_tbox = 1",
    "grounded_esm_tau8": "GroundedEL (ESM2), λ_tbox = 1, τ annealed over first 8 epochs",
    "grounded_esm_pre16": "GroundedEL (ESM2), λ_tbox = 1, 16 TBox-only pre-training epochs, then τ = 0.01",
}


def _load_rows(in_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ont_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        for f in sorted(ont_dir.glob("eval_*_ids*.json")):
            m = FNAME_RE.match(f.name)
            if not m:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            raw, prop = d["raw"], d["propagated"]
            minprop = d.get("min_propagated") or {}
            hier = d.get("hierarchy_consistency") or {}
            crisp = d.get("crisp_report") or {}
            ids = [int(x) for x in m.group("ids").split("-")]
            best_ep: Optional[int] = None
            if len(ids) == 1:
                mp = ont_dir / f"metrics_{m.group('model')}_{ids[0]}.json"
                if mp.is_file():
                    try:
                        best_ep = int(json.loads(mp.read_text(encoding="utf-8")).get("best_val_bce_epoch"))
                    except Exception:
                        best_ep = None
            rows.append(
                {
                    "ont": ont_dir.name,
                    "model": m.group("model"),
                    "combine": m.group("combine"),
                    "n": int(d.get("n_models_used", m.group("n"))),
                    "ids": ids,
                    "best_epoch": best_ep,
                    "fmax_raw": float(raw["fmax"]),
                    "fmax_prop": float(prop["fmax"]),
                    "fmax_minprop": float(minprop["fmax"]) if minprop.get("fmax") is not None else None,
                    "smin_prop": float(prop["smin"]),
                    "aupr_prop": float(prop["aupr"]),
                    "auc_prop": float(prop["avg_auc"]) if prop.get("avg_auc") is not None else None,
                    "hier_viol": hier.get("hier_viol_rate"),
                    "anc_closed": hier.get("ancestor_closed_rate"),
                    "elem_viol": crisp.get("elem_viol"),
                    "region_viol": crisp.get("region_viol"),
                    "region_viol_tol": crisp.get("region_viol_tol1e-4"),
                    "n_empty": crisp.get("n_empty_classes"),
                    "n_false_dead": crisp.get("n_false_dead"),
                }
            )
    return rows


def _ms(vals: Sequence[Optional[float]], fmt: str = "{:.3f}") -> str:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return fmt.format(xs[0])
    return f"{fmt.format(np.mean(xs))} ± {fmt.format(np.std(xs))}"


def _fmt1(v: Any, fmt: str = "{:.3f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return fmt.format(float(v))


COLS = [
    ("fmax_raw", "Fmax raw", "{:.3f}"),
    ("fmax_prop", "Fmax prop", "{:.3f}"),
    ("fmax_minprop", "Fmax minprop", "{:.3f}"),
    ("smin_prop", "Smin", "{:.2f}"),
    ("aupr_prop", "AUPR", "{:.3f}"),
    ("auc_prop", "AUC", "{:.3f}"),
    ("hier_viol", "HierViol", "{:.3f}"),
    ("anc_closed", "AncClosed", "{:.3f}"),
    ("elem_viol", "elem_viol", "{:.3f}"),
    ("region_viol", "region_viol", "{:.3f}"),
    ("region_viol_tol", "rv@1e-4", "{:.3f}"),
    ("n_empty", "empty", "{:.0f}"),
    ("n_false_dead", "false_dead", "{:.0f}"),
]


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    onts = sorted({r["ont"] for r in rows}, key=lambda o: {"mf": 0, "bp": 1, "cc": 2}.get(o, 9))
    out: List[str] = [
        "# E3: DeepGO-SE-style function prediction with GroundedEL (`DEEPGOSE.md`)",
        "",
        "Generated by `python -m experiments.deepgose.aggregate`. Protocol: DeepGO-SE data, splits and",
        "`compute_metrics` (Fmax, Smin, AUPR, mean per-term AUC) on the SwissProt test split; checkpoint by",
        "validation BCE; *raw* = model scores, *prop* = max-propagated to ancestors, *minprop* = min-propagated",
        "to descendants (s'(p,G) = min over G ⊑ G' of s(p,G')). HierViol = fraction of",
        "(protein, G ⊑ G') pairs with s(p,G) > s(p,G') + 1e-6 in the raw scores; AncClosed = fraction of test",
        "proteins whose raw prediction set at t_max is ancestor-closed. The crisp-report columns are the GO-box",
        "faithfulness of GroundedEL checkpoints (`region_viol` over 10⁴ sampled points); baselines have none.",
        "Single-model rows: mean ± sd over seeds. Ensemble rows: all listed seeds combined by `avg` or `min`.",
        "",
    ]
    header = "| model | S | best ep | " + " | ".join(c[1] for c in COLS) + " |"
    sep = "|---|---|---|" + "|".join("---:" for _ in COLS) + "|"
    for ont in onts:
        out += [f"## {ont.upper()}", "", header, sep]
        single = defaultdict(list)
        ens: List[Dict[str, Any]] = []
        for r in rows:
            if r["ont"] != ont:
                continue
            (single[r["model"]].append(r) if r["n"] == 1 else ens.append(r))
        models = sorted(single.keys() | {r["model"] for r in ens}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
        for model in models:
            label = MODEL_LABEL.get(model, model)
            rs = single.get(model, [])
            if rs:
                eps = sorted(r["best_epoch"] for r in rs if r["best_epoch"] is not None)
                ep_s = f"{eps[0]}–{eps[-1]}" if len(eps) > 1 and eps[0] != eps[-1] else (str(eps[0]) if eps else "—")
                cells = [_ms([r[k] for r in rs], fmt) for k, _, fmt in COLS]
                out.append(f"| {label} | 1 (n={len(rs)}) | {ep_s} | " + " | ".join(cells) + " |")
            for r in sorted((r for r in ens if r["model"] == model), key=lambda r: (r["n"], r["combine"])):
                cells = [_fmt1(r[k], fmt) for k, _, fmt in COLS]
                out.append(f"| {label}, {r['combine']} of seeds {','.join(map(str, r['ids']))} | {r['n']} | — | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description="Aggregate E3 evaluation JSONs into DEEPGOSE.md.")
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    rows = _load_rows(args.in_dir)
    text = render_markdown(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({len(rows)} evaluation rows)")
    return args.out


if __name__ == "__main__":
    main()
