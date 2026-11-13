"""Faithfulness evaluation for E-OMQA query answering and TBox satisfaction.

Given a trained GroundedEL run directory (``results.json`` + ``best.pt`` / ``last.pt``), rebuilds the model exactly
as ``experiments/omqa/train.py --eval-only`` does and evaluates set-level query faithfulness and TBox axiom
satisfaction:

1. **Query-level crisp answer sets:** Over the valid candidate entities of :class:`~experiments.omqa.train.Scorer`,
   computes two crisp answer sets at threshold ``--margin m`` (default ``0.0``):
   - ``region``: entities with compiled multi-box region depth ``> m``.
   - ``beam``: entities with Gödel t-norm (max-min) beam score ``> m`` (``k`` from the cell's args, default ``16``).
   Each answer set :math:`A(q)` is evaluated against:
   - the certain answers :math:`C(q) = \\text{easy}(q) \\cup \\text{hard}(q)` (variants ``region`` and ``beam``), and
   - the hard-only answers :math:`H(q)` in the filtered setting where easy answers
     :math:`E(q) = \\text{easy}(q) \\setminus \\text{hard}(q)` are removed from :math:`A(q)`
     (variants ``region_hard`` and ``beam_hard``, also exposed as ``hard_*`` fields on ``region``/``beam``).
   Reported per query shape (macro over queries) and macro over shapes: ``precision``, ``recall``, ``f1``,
   ``jaccard``, ``size`` (``|A|``), ``exact_match`` (:math:`A = C` or :math:`A \\setminus E = H`),
   ``empty_rate`` (:math:`A = \\emptyset`), ``complete`` (fraction of queries with ``recall == 1``), and
   ``sound`` (fraction of queries with ``precision == 1``).
2. **TBox faithfulness:** Reports :meth:`GroundedEL.evaluate_model` and :meth:`GroundedEL.crisp_report` overall and
   per axiom type at tolerances ``0`` and ``1e-4``.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MOWL_JVM_MEMORY", "64m")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallopt(-1, -1)  # M_TRIM_THRESHOLD = -1 (keep sbrk heap cached)
    _libc.mallopt(-4, 0)   # M_MMAP_MAX = 0 (avoid repeated mmap/munmap per chunk)
except Exception:  # pragma: no cover
    pass

import numpy as np
import torch as th

from experiments.omqa import protocol as omqa_protocol
from experiments.omqa.train import Scorer, build_model

CANONICAL_SHAPES: Tuple[str, ...] = ("1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up")
AXIOM_TYPES: Tuple[str, ...] = (
    "gci0",
    "gci0_bot",
    "gci1",
    "gci1_bot",
    "gci2",
    "gci3",
    "gci3_bot",
    "role_inclusion",
    "role_chain",
)
ABOX_TYPES: Tuple[str, ...] = ("class_assertion", "object_property_assertion")
METRIC_KEYS: Tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "jaccard",
    "size",
    "|A|",
    "exact_match",
    "empty_rate",
    "complete",
    "sound",
)


def compute_set_metrics_from_counts(
    tp: th.Tensor,
    pred_size: th.Tensor,
    gt_size: th.Tensor,
) -> Dict[str, th.Tensor]:
    """Compute per-query set faithfulness metrics from ``(tp, |A|, |T|)`` 1-D count tensors.

    Guarantees:
    - All returned tensors are 1-D ``torch.float64`` of shape ``(B,)``.
    - For queries with ``|T| > 0`` and ``|A| == 0``, ``precision = 0``, ``recall = 0``, ``f1 = 0``,
      ``jaccard = 0``, ``exact_match = 0``, ``empty_rate = 1``, ``complete = 0``, ``sound = 0``.
    - ``exact_match == 1`` iff ``A == T`` (``tp == |A|`` and ``tp == |T|``).
    - ``complete == 1`` iff ``recall == 1`` (``tp == |T|``).
    - ``sound == 1`` iff ``precision == 1`` (``tp == |A|`` with ``|A| > 0``, or ``|A| == |T| == 0``).

    Non-guarantees:
    - Assumes ``0 <= tp <= min(pred_size, gt_size)``.
    """
    tp_f = tp.to(th.float64)
    a_f = pred_size.to(th.float64)
    t_f = gt_size.to(th.float64)

    both_empty = (pred_size == 0) & (gt_size == 0)
    precision = th.where(
        pred_size > 0,
        tp_f / a_f.clamp_min(1.0),
        th.where(both_empty, th.ones_like(tp_f), th.zeros_like(tp_f)),
    )
    recall = th.where(
        gt_size > 0,
        tp_f / t_f.clamp_min(1.0),
        th.ones_like(tp_f),
    )
    denom = a_f + t_f
    f1 = th.where(
        denom > 0,
        2.0 * tp_f / denom.clamp_min(1.0),
        th.ones_like(tp_f),
    )
    union = pred_size + gt_size - tp
    jaccard = th.where(
        union > 0,
        tp_f / union.to(th.float64).clamp_min(1.0),
        th.ones_like(tp_f),
    )
    exact_match = ((tp == pred_size) & (tp == gt_size)).to(th.float64)
    empty_rate = (pred_size == 0).to(th.float64)
    complete = (tp == gt_size).to(th.float64)
    sound = (((pred_size > 0) & (tp == pred_size)) | both_empty).to(th.float64)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "size": a_f,
        "|A|": a_f,
        "exact_match": exact_match,
        "empty_rate": empty_rate,
        "complete": complete,
        "sound": sound,
    }


def _scatter_padded_mask(
    padded: th.Tensor,
    n_entities: int,
) -> th.Tensor:
    """Convert a ``(B, W)`` padded index tensor (pad ``< 0``) into a ``(B, n_entities)`` boolean mask."""
    bsz = int(padded.shape[0])
    mask = th.zeros((bsz, n_entities), dtype=th.bool, device=padded.device)
    if padded.numel() == 0:
        return mask
    valid = (padded >= 0) & (padded < n_entities)
    if bool(valid.any().item()):
        rows = th.arange(bsz, device=padded.device).unsqueeze(1).expand_as(padded)
        mask[rows[valid], padded[valid]] = True
    return mask


def evaluate_batch_sets(
    scores: th.Tensor,
    cand_valid: th.Tensor,
    cert_mask: th.Tensor,
    hard_mask: th.Tensor,
    margin: float = 0.0,
) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor]]:
    """Compute per-query metric tensors against ``C(q)`` and filtered ``H(q)`` for one batch of scores.

    Parameters
    ----------
    scores : torch.Tensor
        Entity scores of shape ``(B, N)``.
    cand_valid : torch.Tensor
        Boolean mask of shape ``(N,)`` marking valid candidate entities.
    cert_mask : torch.Tensor
        Boolean mask of shape ``(B, N)`` for :math:`C(q) = \\text{easy}(q) \\cup \\text{hard}(q)`.
    hard_mask : torch.Tensor
        Boolean mask of shape ``(B, N)`` for :math:`H(q) = \\text{hard}(q)`.
    margin : float, default 0.0
        Strict inclusion threshold: an entity is inside iff ``score > margin``.

    Returns
    -------
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ``(certain_metrics, hard_filtered_metrics)`` where each dict maps metric names to ``(B,)`` tensors.
    """
    valid_row = cand_valid.to(device=scores.device, dtype=th.bool).unsqueeze(0)
    pred_c = (scores > float(margin)) & valid_row
    cert = cert_mask | hard_mask
    easy_only = cert & (~hard_mask)

    tp_c = (pred_c & cert).sum(dim=1)
    size_c = pred_c.sum(dim=1)
    gt_c = cert.sum(dim=1)
    m_cert = compute_set_metrics_from_counts(tp_c, size_c, gt_c)

    pred_h = pred_c & (~easy_only)
    tp_h = (pred_h & hard_mask).sum(dim=1)
    size_h = pred_h.sum(dim=1)
    gt_h = hard_mask.sum(dim=1)
    m_hard = compute_set_metrics_from_counts(tp_h, size_h, gt_h)
    return m_cert, m_hard


def _load_split_prepared(data_root: Path, test_case: str, part: str = "test") -> Dict[Any, omqa_protocol.PreparedSplit]:
    """Load prepared split, using the ``.prepared.pt`` cache if present without writing to ``data_root`` otherwise."""
    cpath = Path(data_root) / test_case / f"{part}.prepared.pt"
    return omqa_protocol.load_prepared(Path(data_root), test_case, part, cache=cpath.exists())


def evaluate_case_faithfulness(
    reg_scorer: Scorer,
    beam_scorer: Scorer,
    prepared: Dict[Any, omqa_protocol.PreparedSplit],
    margin: float = 0.0,
    shapes: Optional[Sequence[str]] = None,
    max_q: Optional[int] = None,
    batch_size: int = 256,
    extra_margins: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Evaluate ``region`` and ``beam`` set faithfulness across all requested shapes of a test case."""
    shape_filter = set(shapes) if shapes else None
    margins = [float(margin)] + [float(m) for m in (extra_margins or []) if float(m) != float(margin)]
    sfx = lambda m: "" if m == margins[0] else f"@{m:g}"  # noqa: E731
    VARIANTS = tuple(f"{v}{sfx(m)}" for m in margins for v in ("region", "beam", "region_hard", "beam_hard"))
    n_entities = int(reg_scorer.cand_valid.shape[0])
    cand_valid = reg_scorer.cand_valid

    by_name: Dict[str, Tuple[Any, omqa_protocol.PreparedSplit]] = {}
    for structure, ps in prepared.items():
        name = omqa_protocol.QUERY_NAMES.get(structure, str(structure))
        if shape_filter is not None and name not in shape_filter:
            continue
        by_name[name] = (structure, ps)

    ordered_names = [s for s in CANONICAL_SHAPES if s in by_name] + [
        s for s in by_name if s not in CANONICAL_SHAPES
    ]

    case_out: Dict[str, Any] = {}
    for name in ordered_names:
        structure, ps = by_name[name]
        total_q = len(ps.queries)
        n_q = min(total_q, int(max_q)) if (max_q is not None and max_q > 0) else total_q
        if n_q == 0:
            continue
        qs = ps.queries[:n_q]

        sums: Dict[str, Dict[str, float]] = {
            v: {k: 0.0 for k in METRIC_KEYS}
            for v in VARIANTS
        }

        for lo in range(0, n_q, batch_size):
            hi = min(lo + batch_size, n_q)
            batch = qs[lo:hi]
            with th.no_grad():
                sc_reg = reg_scorer.score_entities(structure, batch).detach()
                sc_beam = beam_scorer.score_entities(structure, batch).detach()

            dev = sc_reg.device
            a_pad, _ = omqa_protocol._pad_slice(ps.hard_flat, ps.hard_off, lo, hi, -1, dev)
            f_pad, _ = omqa_protocol._pad_slice(ps.filt_flat, ps.filt_off, lo, hi, -1, dev)
            hard_mask = _scatter_padded_mask(a_pad, n_entities)
            cert_mask = _scatter_padded_mask(f_pad, n_entities) | hard_mask

            per_variant = []
            for m in margins:
                reg_c, reg_h = evaluate_batch_sets(sc_reg, cand_valid, cert_mask, hard_mask, margin=m)
                beam_c, beam_h = evaluate_batch_sets(sc_beam, cand_valid, cert_mask, hard_mask, margin=m)
                per_variant += [(f"region{sfx(m)}", reg_c), (f"region_hard{sfx(m)}", reg_h),
                                (f"beam{sfx(m)}", beam_c), (f"beam_hard{sfx(m)}", beam_h)]

            for v_name, m_dict in per_variant:
                stacked = th.stack([m_dict[k] for k in METRIC_KEYS], dim=0).sum(dim=1).cpu().tolist()
                for k, val in zip(METRIC_KEYS, stacked):
                    sums[v_name][k] += float(val)

        shape_res: Dict[str, Dict[str, float]] = {}
        for v_name in VARIANTS:
            shape_res[v_name] = {k: sums[v_name][k] / n_q for k in METRIC_KEYS}
            shape_res[v_name]["n"] = float(n_q)
        for base_v, hard_v in [(f"region{sfx(m)}", f"region_hard{sfx(m)}") for m in margins] + [(f"beam{sfx(m)}", f"beam_hard{sfx(m)}") for m in margins]:
            for k in METRIC_KEYS:
                shape_res[base_v][f"hard_{k}"] = shape_res[hard_v][k]
        case_out[name] = shape_res

    if case_out:
        macro: Dict[str, Dict[str, float]] = {}
        shape_dicts = list(case_out.values())
        for v_name in VARIANTS:
            all_keys = list(shape_dicts[0][v_name].keys())
            macro[v_name] = {
                k: float(np.mean([sd[v_name][k] for sd in shape_dicts]))
                for k in all_keys
            }
        case_out["macro"] = macro
    return case_out


def evaluate_tbox_faithfulness(model: Any, n_eval: int = 10000) -> Dict[str, Any]:
    """Compute overall and per-axiom-type crisp satisfaction at tolerances ``0`` and ``1e-4``."""
    ev_0 = model.evaluate_model(use_reasoner=False, n_eval=n_eval, tol=0.0)
    cr_0 = model.crisp_report(n_eval=n_eval, tol=0.0)
    cr_1e4 = model.crisp_report(n_eval=n_eval, tol=1e-4)

    td = dict(model.training_datasets)
    by_axiom: Dict[str, Dict[str, Any]] = {}
    try:
        for ax_type in AXIOM_TYPES + ABOX_TYPES:
            if ax_type not in td or len(td[ax_type]) == 0:
                continue
            n_ax = int(len(td[ax_type]))
            model._training_datasets = {ax_type: td[ax_type]}
            sub_0 = model.crisp_report(n_eval=n_eval, tol=0.0)
            sub_1e4 = model.crisp_report(n_eval=n_eval, tol=1e-4)
            entry: Dict[str, Any] = {
                "n": n_ax,
                "elem_viol": float(sub_0["elem_viol"]),
                "region_viol": float(sub_0["region_viol"]),
                "region_viol_tol0": float(sub_0["region_viol"]),
                "region_viol_tol1e-4": float(sub_1e4["region_viol"]),
                "n_violated_tol0": int(sub_0["n_axioms_violated_by_depth"]["0"]),
                "n_violated_tol1e-4": int(sub_0["n_axioms_violated_by_depth"]["1e-4"]),
            }
            by_axiom[ax_type] = entry
    finally:
        model._training_datasets = td

    out: Dict[str, Any] = {
        "elem_viol": float(ev_0["elem_viol"]),
        "region_viol": float(ev_0["region_viol"]),
        "region_viol_tol0": float(cr_0["region_viol"]),
        "region_viol_tol1e-4": float(cr_1e4["region_viol"]),
        "exist": float(ev_0["exist"]),
        "dead_ok": bool(ev_0["dead_ok"]),
        "n_empty_classes": int(ev_0["n_empty_classes"]),
        "n_dead_elements": int(ev_0["n_dead_elements"]),
        "exact": bool(ev_0["exact"]),
        "exact_tol1e-4": bool(ev_0["exact_tol1e-4"]),
        "n_axioms_violated_by_depth": {str(k): int(v) for k, v in cr_0["n_axioms_violated_by_depth"].items()},
        "by_axiom_type": by_axiom,
    }
    for ax_type, entry in by_axiom.items():
        out[ax_type] = entry
    return out


def format_markdown_report(results: Dict[str, Any], test_cases: Sequence[str]) -> str:
    """Format a concise Markdown summary table of query-level and TBox faithfulness."""
    lines: List[str] = []
    meta = results.get("args", {})
    lines.append(
        f"### Faithfulness: `{meta.get('name', 'run')}` "
        f"(ckpt={meta.get('ckpt')}, epoch={meta.get('epoch')}, margin={meta.get('margin', 0.0)}, "
        f"beam_k={meta.get('beam_k', 16)})"
    )
    tbox = results.get("tbox", {})
    if tbox:
        ax_parts = []
        for ax, info in tbox.get("by_axiom_type", {}).items():
            if ax in AXIOM_TYPES:
                ax_parts.append(
                    f"{ax} (n={info['n']}): reg_viol={info['region_viol_tol0']:.4f} "
                    f"(tol1e-4={info['region_viol_tol1e-4']:.4f}), elem_viol={info['elem_viol']:.4f}"
                )
            else:
                ax_parts.append(f"{ax} (n={info['n']}): elem_viol={info['elem_viol']:.4f}")
        lines.append(
            f"- **TBox/ABox**: elem_viol={tbox.get('elem_viol', 0.0):.4f}, "
            f"region_viol(tol=0)={tbox.get('region_viol_tol0', 0.0):.4f}, "
            f"region_viol(tol=1e-4)={tbox.get('region_viol_tol1e-4', 0.0):.4f}, "
            f"exist={tbox.get('exist', 0.0):.4f}, exact={tbox.get('exact', False)}"
        )
        if ax_parts:
            lines.append("- **By axiom type**: " + "; ".join(ax_parts))

    for tc in test_cases:
        case_data = results.get(tc, {})
        if not case_data:
            continue
        lines.append("")
        lines.append(f"#### Case `{tc}`")
        lines.append(
            "| shape | n | variant | P(C) | R(C) | F1(C) | Jacc(C) | \\|A\\| | EM(C) | empty | complete | sound | P(H) | R(H) | F1(H) | Jacc(H) | EM(H) |"
        )
        lines.append(
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        row_keys = [k for k in case_data.keys() if k != "macro"] + (["macro"] if "macro" in case_data else [])
        for sh in row_keys:
            sd = case_data[sh]
            for v in ("region", "beam"):
                if v not in sd:
                    continue
                m = sd[v]
                mh = sd.get(f"{v}_hard", {})
                label = f"**{sh}**" if sh == "macro" else sh
                lines.append(
                    f"| {label} | {int(round(m.get('n', 0.0)))} | {v} | "
                    f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['jaccard']:.3f} | "
                    f"{m['size']:.1f} | {m['exact_match']:.3f} | {m['empty_rate']:.3f} | "
                    f"{m['complete']:.3f} | {m['sound']:.3f} | "
                    f"{mh.get('precision', m.get('hard_precision', 0.0)):.3f} | "
                    f"{mh.get('recall', m.get('hard_recall', 0.0)):.3f} | "
                    f"{mh.get('f1', m.get('hard_f1', 0.0)):.3f} | "
                    f"{mh.get('jaccard', m.get('hard_jaccard', 0.0)):.3f} | "
                    f"{mh.get('exact_match', m.get('hard_exact_match', 0.0)):.3f} |"
                )
    return "\n".join(lines)


def _resolve_checkpoint(run_dir: Path, ckpt: str) -> Path:
    if ckpt in ("best", "last"):
        cand = run_dir / f"{ckpt}.pt"
        if cand.is_file():
            return cand
    p = Path(ckpt)
    if p.is_file():
        return p
    cand2 = run_dir / ckpt
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(f"Checkpoint {ckpt!r} not found under {run_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path, help="directory with results.json and best.pt/last.pt")
    ap.add_argument("--ckpt", default="best", help="checkpoint to load: 'best', 'last', or path (default: best)")
    ap.add_argument("--tables", default=None, help="override tables.pt path from results.json")
    ap.add_argument("--data-root", default=None, help="override data_root path from results.json")
    ap.add_argument("--test-cases", nargs="*", default=None, help="test cases (default: from results.json)")
    ap.add_argument("--shapes", nargs="*", default=None, help="restrict evaluation to these query shapes")
    ap.add_argument("--max-q", type=int, default=None, help="max queries per shape (default: all)")
    ap.add_argument("--margin", type=float, default=0.0, help="crisp inclusion threshold m: score > m (default: 0.0)")
    ap.add_argument("--extra-margins", type=float, nargs="*", default=[], help="further thresholds evaluated in the same pass; variants get the suffix @m")
    ap.add_argument("--beam-k", type=int, default=None, help="override beam width k (default: cell args or 16)")
    ap.add_argument("--device", default=None, help="torch device (default: cpu or cuda if available)")
    ap.add_argument("--batch-size", type=int, default=256, help="query evaluation batch size")
    ap.add_argument("--n-eval", type=int, default=None, help="sample budget for TBox crisp_report")
    ap.add_argument("--out", default=None, type=Path, help="output JSON path")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    ckpt_path = _resolve_checkpoint(run_dir, args.ckpt)
    ck = th.load(ckpt_path, map_location="cpu", weights_only=False)

    res_json_path = run_dir / "results.json"
    if res_json_path.is_file():
        res_json = json.loads(res_json_path.read_text())
        raw_cell_args: Dict[str, Any] = dict(res_json.get("args", {}))
    elif isinstance(ck, dict) and "args" in ck:
        res_json = {}
        raw_cell_args = dict(ck["args"])
    else:
        raise FileNotFoundError(f"Neither {res_json_path} nor checkpoint 'args' found in {run_dir}")

    device = args.device or ("cuda" if th.cuda.is_available() else "cpu")
    if str(device).startswith("cpu"):
        th.set_num_threads(min(8, os.cpu_count() or 8))
    tables_path = args.tables or raw_cell_args.get("tables")
    data_root_path = args.data_root or raw_cell_args.get("data_root")
    if not tables_path or not data_root_path:
        raise ValueError("Both --tables and --data-root must be provided or present in results.json args")

    test_cases: List[str] = (
        list(args.test_cases)
        if args.test_cases
        else list(raw_cell_args.get("test_cases") or ["test_ind_ded", "test_ded", "test_ind"])
    )
    beam_k = int(args.beam_k if args.beam_k is not None else (raw_cell_args.get("beam_k") or 16))
    max_boxes = int(raw_cell_args.get("max_boxes", 64))
    n_eval = int(args.n_eval if args.n_eval is not None else raw_cell_args.get("n_eval", 10000))
    benchmark = str(raw_cell_args.get("benchmark", "eomqa"))

    cell_ns = argparse.Namespace(**raw_cell_args)
    cell_ns.device = device
    cell_ns.tables = str(tables_path)
    cell_ns.data_root = str(data_root_path)

    th.manual_seed(int(getattr(cell_ns, "seed", 0)))
    np.random.seed(int(getattr(cell_ns, "seed", 0)))

    tables = th.load(tables_path, weights_only=False)
    model = build_model(tables, cell_ns)
    model.init_module()
    state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.module.load_state_dict({k: v.to(model.device) for k, v in state_dict.items()})
    model.module.eval()

    epoch = int(ck.get("epoch", res_json.get("tested_epoch", -1))) if isinstance(ck, dict) else -1
    reg_scorer = Scorer(model, tables, benchmark, exact=False, max_boxes=max_boxes, scorer="region", beam_k=beam_k)
    beam_scorer = Scorer(model, tables, benchmark, exact=False, max_boxes=max_boxes, scorer="beam", beam_k=beam_k)

    out_data: Dict[str, Any] = {}
    data_root = Path(data_root_path)
    for tc in test_cases:
        prepared = _load_split_prepared(data_root, tc, "test")
        out_data[tc] = evaluate_case_faithfulness(
            reg_scorer=reg_scorer,
            beam_scorer=beam_scorer,
            prepared=prepared,
            margin=args.margin,
            extra_margins=args.extra_margins,
            shapes=args.shapes,
            max_q=args.max_q,
            batch_size=args.batch_size,
        )

    out_data["tbox"] = evaluate_tbox_faithfulness(model, n_eval=n_eval)
    out_data["args"] = {
        "name": raw_cell_args.get("name", run_dir.name),
        "run_dir": str(run_dir),
        "ckpt": str(args.ckpt),
        "ckpt_path": str(ckpt_path),
        "epoch": epoch,
        "tables": str(tables_path),
        "data_root": str(data_root_path),
        "test_cases": test_cases,
        "shapes": list(args.shapes) if args.shapes else None,
        "max_q": args.max_q,
        "margin": float(args.margin),
        "beam_k": beam_k,
        "max_boxes": max_boxes,
        "device": device,
        "n_eval": n_eval,
        "cell_args": raw_cell_args,
    }

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_data, indent=2, default=str))

    print(format_markdown_report(out_data, test_cases), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
