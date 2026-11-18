"""Combine several trained GroundedEL models for E-OMQA query answering, and diagnose missed answers.

A single geometric model makes true everything that holds in it, which includes statements the knowledge base
does not entail. Certain answers hold in every model, so several independently trained models can be combined.
This script loads N run directories (``results.json`` + checkpoint), scores every test query with each model
(beam scorer, as in training) and ranks with the E-OMQA protocol under the following combinations:

``single``  each model alone (mean and standard deviation over the models are reported)
``mean``    mean of the scores
``min``     minimum of the scores (an answer must score well in all models: certain-answer reading)
``max``     maximum of the scores (an answer may come from any model)
``median``  median of the scores
``zmean``   mean of the per-query standardised scores
``rank``    mean rank (Borda)
``rrf``     reciprocal rank fusion, sum of 1 / (60 + rank)
``vote``    number of models in which the answer is crisply true (score > 0), ties broken by the mean score

Each combination is evaluated for the first 1, 2, ..., N models (``--sizes``).

Diagnosis (``--diagnose``): for every hard answer and every single model, the answer is either in the top 3
(``hit``), crisply true in the model but outranked (``derived_outranked``), or not true in the model
(``not_derived``, score <= 0: a fact or axiom on its derivation is not satisfied by the regions). Also reported:
the mean number of non-answers that are crisply true per query (``false_true``), and the share of hard answers
that are crisply true in all / in at least one of the models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch as th

from experiments.omqa import protocol as P
from experiments.omqa.train import Scorer, build_model

METHODS = ("mean", "min", "max", "median", "zmean", "rank", "rrf", "vote")
NEG = -1e6


def load_scorer(run_dir: Path, ckpt: str, device: str, tables_cache: Dict[str, Any]) -> Scorer:
    res = json.loads((run_dir / "results.json").read_text())
    a = dict(res["args"])
    ns = argparse.Namespace(**a)
    ns.device = device
    if a["tables"] not in tables_cache:
        tables_cache[a["tables"]] = th.load(a["tables"], weights_only=False)
    tables = tables_cache[a["tables"]]
    th.manual_seed(int(a.get("seed", 0)))
    model = build_model(tables, ns)
    model.init_module()
    ck = th.load(run_dir / f"{ckpt}.pt", map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.module.load_state_dict({k: v.to(model.device) for k, v in sd.items()})
    model.module.eval()
    return Scorer(model, tables, a.get("benchmark", "eomqa"), exact=False, max_boxes=int(a.get("max_boxes", 64)),
                  scorer="beam", beam_k=int(a.get("beam_k") or 16))


def ranks_desc(s: th.Tensor) -> th.Tensor:
    """0-based rank of every entry per row (0 = highest score)."""
    order = th.argsort(s, dim=1, descending=True, stable=True)
    r = th.empty_like(order)
    r.scatter_(1, order, th.arange(s.shape[1], device=s.device).unsqueeze(0).expand_as(order))
    return r


def combine(S: th.Tensor, method: str, valid: th.Tensor) -> th.Tensor:
    """``S``: (M, B, N) scores of M models; returns (B, N) combined scores (higher = better)."""
    if method == "mean":
        out = S.mean(0)
    elif method == "min":
        out = S.min(0).values
    elif method == "max":
        out = S.max(0).values
    elif method == "median":
        out = S.contiguous().median(0).values
    elif method == "zmean":
        v = valid.unsqueeze(0).unsqueeze(0)
        n = v.sum()
        mu = (S * v).sum(-1, keepdim=True) / n
        sd = (((S - mu) ** 2) * v).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-9)
        out = ((S - mu) / sd).mean(0)
    elif method in ("rank", "rrf"):
        R = th.stack([ranks_desc(S[m]) for m in range(S.shape[0])]).to(th.float)
        out = -R.mean(0) if method == "rank" else (1.0 / (60.0 + R)).sum(0)
    elif method == "vote":
        m = S.mean(0)
        span = (m.max(1, keepdim=True).values - m.min(1, keepdim=True).values).clamp_min(1e-9)
        out = (S > 0).sum(0).to(th.float) + 0.5 * (m - m.min(1, keepdim=True).values) / span
    else:
        raise ValueError(method)
    return out.masked_fill(~valid.unsqueeze(0), NEG)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path, help="run directories (same tables)")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--test-cases", nargs="*", default=["test_ded", "test_ind_ded"])
    ap.add_argument("--sizes", nargs="*", type=int, default=None, help="ensemble sizes (default: 1..N)")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-q", type=int, default=None)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    cache: Dict[str, Any] = {}
    scorers = [load_scorer(r, args.ckpt, args.device, cache) for r in args.runs]
    M = len(scorers)
    sizes = args.sizes or list(range(1, M + 1))
    valid = scorers[0].cand_valid
    data_root = Path(args.data_root or json.loads((args.runs[0] / "results.json").read_text())["args"]["data_root"])
    out: Dict[str, Any] = {"runs": [str(r) for r in args.runs], "ckpt": args.ckpt, "sizes": sizes}

    for tc in args.test_cases:
        prepared = P.load_prepared(data_root, tc, "test")
        res: Dict[str, Any] = {}
        for structure, ps in prepared.items():
            name = P.QUERY_NAMES.get(structure, str(structure))
            qs = ps.queries if args.max_q is None else ps.queries[: args.max_q]
            keys = [f"single{m}" for m in range(M)] + [f"{meth}@{k}" for k in sizes if k > 1 for meth in args.methods]
            h3 = {k: 0.0 for k in keys}
            mrr = {k: 0.0 for k in keys}
            diag = {k: 0.0 for k in ("hit", "derived_outranked", "not_derived", "false_true", "n_true",
                                     "true_all", "true_any", "n_hard_w")}
            n_q = 0
            for i in range(0, len(qs), args.batch_size):
                batch = qs[i:i + args.batch_size]
                B = len(batch)
                S = th.stack([sc.score_entities(structure, batch) for sc in scorers])  # (M, B, N)
                a_pad, n_hard = P._pad_slice(ps.hard_flat, ps.hard_off, i, i + B, 0, S.device)
                f_pad, _ = P._pad_slice(ps.filt_flat, ps.filt_off, i, i + B, -1, S.device)

                def acc(key: str, scores: th.Tensor):
                    r, qi = P._ranks_padded(scores, a_pad, n_hard, f_pad)
                    cnt = th.bincount(qi, minlength=B).to(r.dtype)
                    h3[key] += float((th.bincount(qi, weights=(r <= 3).to(r.dtype), minlength=B) / cnt).sum())
                    mrr[key] += float((th.bincount(qi, weights=1.0 / r, minlength=B) / cnt).sum())
                    return r, qi, cnt

                hv = th.arange(a_pad.shape[1], device=S.device).unsqueeze(0) < n_hard.unsqueeze(1)
                true_stack = []
                for m in range(M):
                    r, qi, cnt = acc(f"single{m}", S[m])
                    if args.diagnose:
                        pos = (S[m] > 0) & valid.unsqueeze(0)
                        a_true = pos.gather(1, a_pad)[hv]  # per (query, hard answer), same order as r
                        true_stack.append(a_true)
                        w = 1.0 / cnt[qi]  # macro over queries
                        hit = r <= 3
                        diag["hit"] += float((w * hit).sum()) / M
                        diag["derived_outranked"] += float((w * (~hit & a_true)).sum()) / M
                        diag["not_derived"] += float((w * (~hit & ~a_true)).sum()) / M
                        f_true = th.where(f_pad >= 0, pos.gather(1, f_pad.clamp_min(0)), th.zeros_like(f_pad, dtype=th.bool))
                        n_true = pos.sum(1).to(th.float)
                        diag["n_true"] += float(n_true.sum()) / M
                        diag["false_true"] += float((n_true - f_true.sum(1)).sum()) / M
                if args.diagnose:
                    T = th.stack(true_stack)
                    w = 1.0 / cnt[qi]  # qi and cnt depend only on the answer sets, so the last model's are valid
                    diag["true_all"] += float((w * T.all(0)).sum())
                    diag["true_any"] += float((w * T.any(0)).sum())
                for k in sizes:
                    if k <= 1:
                        continue
                    for meth in args.methods:
                        acc(f"{meth}@{k}", combine(S[:k], meth, valid))
                n_q += B
            entry: Dict[str, Any] = {"n": n_q}
            singles = [h3[f"single{m}"] / n_q for m in range(M)]
            entry["hits3"] = {"single_mean": float(np.mean(singles)), "single_sd": float(np.std(singles)), "single": singles}
            entry["mrr"] = {"single_mean": float(np.mean([mrr[f"single{m}"] / n_q for m in range(M)]))}
            for k in keys[M:]:
                entry["hits3"][k] = h3[k] / n_q
                entry["mrr"][k] = mrr[k] / n_q
            if args.diagnose:
                entry["diagnosis"] = {k: v / n_q for k, v in diag.items() if k != "n_hard_w"}
            res[name] = entry
            print(f"[{tc}] {name} n={n_q} single {entry['hits3']['single_mean']:.4f} " +
                  " ".join(f"{m}@{sizes[-1]} {entry['hits3'].get(f'{m}@{sizes[-1]}', float('nan')):.4f}" for m in args.methods), flush=True)
            if args.diagnose:
                print("   diag " + json.dumps({k: round(v, 4) for k, v in entry["diagnosis"].items()}), flush=True)
        shapes = list(res)
        avg: Dict[str, Any] = {"hits3": {}, "mrr": {}}
        for k in res[shapes[0]]["hits3"]:
            if k == "single":
                per_model = np.mean([res[s]["hits3"]["single"] for s in shapes], axis=0)
                avg["hits3"]["single"] = per_model.tolist()
                avg["hits3"]["single_mean"] = float(per_model.mean())
                avg["hits3"]["single_sd"] = float(per_model.std())
            elif k not in ("single_mean", "single_sd"):
                avg["hits3"][k] = float(np.mean([res[s]["hits3"][k] for s in shapes]))
        for k in res[shapes[0]]["mrr"]:
            avg["mrr"][k] = float(np.mean([res[s]["mrr"][k] for s in shapes]))
        if args.diagnose:
            avg["diagnosis"] = {k: float(np.mean([res[s]["diagnosis"][k] for s in shapes])) for k in res[shapes[0]]["diagnosis"]}
        res["avg"] = avg
        out[tc] = res
        print(f"[{tc}] AVG " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in avg["hits3"].items() if k != "single"}), flush=True)
        if args.diagnose:
            print(f"[{tc}] AVG diag " + json.dumps({k: round(v, 4) for k, v in avg["diagnosis"].items()}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
