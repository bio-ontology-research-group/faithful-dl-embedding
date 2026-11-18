"""Predicted facts of a QTO link predictor as extra ABox input (for GroundedEL, via build_eomqa --extra-triples).

    python -m experiments.omqa.hybrid.predicted_facts --adj IN.pt --data-dir <QTO data dir> --tbox tbox_<kb>.json \
        --thresholds 0.9 0.7 0.5 0.3 --max-per-head-rel 3 --out-dir DIR [--ref-dirs DIR ...]

From a QTO neural adjacency (NOT closed) list the entries ``(h, r, t)`` with score >= threshold that are not training
triples of the data dir. Facts whose inverse ``(t, inv(r), h)`` is a training triple are excluded as well (they are
not new for GroundedEL, whose tables contain both directions of every training triple; counted separately; disable
with ``--keep-inverse-of-train``). At most ``--max-per-head-rel`` highest-scoring new tails are kept per ``(h, r)``.
Writes ``DIR/pred_t<thr>.tsv`` (``h<TAB>r<TAB>t<TAB>score``) per threshold and ``DIR/counts.json``.

Diagnostic precision (never used to choose anything): reference set = the (anchor, relation, answer) triples of the
1p queries of the test sets ``test_ind``, ``test_ind_ded``, ``test_ded`` (QTO data dirs ``<kb>-ind``,
``<kb>-ind_ded``, ``<kb>-ded``, same suffix as ``--data-dir``), easy + hard answers, closed under the inverse pairs,
minus the training triples. ``precision_all`` divides by all new facts (a lower bound: most (h, r) pairs are not
test queries); ``precision_judgeable`` divides by the new facts whose (h, r) or (t, inv(r)) is a 1p test query
(for those the reference answer set is complete, so a miss is a false fact in the test graph).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.omqa.hybrid.qto_io import load_qto_adj, read_stats, read_train
from experiments.omqa.hybrid.tbox import Rules, load_rules


def topk_new_facts(mats, train: np.ndarray, rules: Optional[Rules], min_thr: float, k: int,
                   exclude_inverse_of_train: bool = True) -> Dict[str, Any]:
    """All new facts with score >= min_thr, capped at k per (h, r). Returns arrays h, r, t, score (float32) and
    exclusion counts."""
    n = mats[0].shape[0]
    hs, rs, ts, ss = [], [], [], []
    n_train_excl = n_inv_excl = n_cap_excl = 0
    for r, m in enumerate(mats):
        sel = m.data >= min_thr
        if not sel.any():
            continue
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(m.indptr))[sel]
        cols = m.indices[sel].astype(np.int64)
        vals = m.data[sel].astype(np.float32)
        key = rows * n + cols
        tr = train[train[:, 1] == r]
        bad = np.isin(key, tr[:, 0] * n + tr[:, 2])
        n_train_excl += int(bad.sum())
        if exclude_inverse_of_train and rules is not None:
            inv_src = [s for s, t in rules.inverse.items() if t == r]  # relations s with inv(s) = r
            if inv_src:
                ti = train[np.isin(train[:, 1], inv_src)]
                bad_inv = np.isin(key, ti[:, 2] * n + ti[:, 0]) & ~bad
                n_inv_excl += int(bad_inv.sum())
                bad |= bad_inv
        rows, cols, vals = rows[~bad], cols[~bad], vals[~bad]
        if len(vals) == 0:
            continue
        order = np.lexsort((cols, -vals, rows))  # by head, then score descending (ties: smaller tail first)
        rows, cols, vals = rows[order], cols[order], vals[order]
        first = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1]
        rank = np.arange(len(rows)) - np.repeat(first, np.diff(np.r_[first, len(rows)]))
        keep = rank < k
        n_cap_excl += int((~keep).sum())
        hs.append(rows[keep]); ts.append(cols[keep]); ss.append(vals[keep]); rs.append(np.full(int(keep.sum()), r, dtype=np.int64))
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt))
    return {"h": cat(hs, np.int64), "r": cat(rs, np.int64), "t": cat(ts, np.int64), "score": cat(ss, np.float32),
            "n_excluded_train": n_train_excl, "n_excluded_inverse_of_train": n_inv_excl, "n_excluded_cap": n_cap_excl}


def write_tsv(path: Path, h, r, t, score) -> None:
    with open(path, "w") as fh:
        for a, b, c, s in zip(h.tolist(), r.tolist(), t.tolist(), score.tolist()):
            fh.write(f"{a}\t{b}\t{c}\t{s:.6f}\n")


def read_tsv(path: Path) -> np.ndarray:
    """(n, 3) int64 triples from a tsv ``h r t [score]`` (score column optional and ignored)."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def reference_set(ref_dirs: Sequence[Path], rules: Optional[Rules]):
    """Reference triples (easy + hard 1p answers, inverse-closed) and the set of 1p query keys (anchor, rel)."""
    from experiments.omqa.build_eomqa import norm, safe_load
    ref = set()
    qkeys = set()
    used = []
    for d in ref_dirs:
        qp = d / "test-queries.pkl"
        if not qp.is_file():
            continue
        used.append(str(d))
        qs = safe_load(qp)
        ans = [safe_load(d / f"test-{s}-answers.pkl") for s in ("easy", "hard")]
        for q in qs.get(("e", ("r",)), ()):
            e, (r,) = norm(q)
            qkeys.add((e, r))
            for a in ans:
                for x in a.get(q, ()):
                    ref.add((e, r, int(x)))
    if rules is not None:
        ref |= {(t, rules.inverse[r], h) for h, r, t in ref if r in rules.inverse}
    return ref, qkeys, used


def default_ref_dirs(data_dir: Path) -> List[Path]:
    name = data_dir.name
    parts = name.split("-")
    kb, suffix = parts[0], ("-" + "-".join(parts[2:]) if len(parts) > 2 else "")
    return [data_dir.parent / f"{kb}-{c}{suffix}" for c in ("ind", "ind_ded", "ded")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adj", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tbox", required=True, help="resolved tbox json (inverse pairs)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.9, 0.7, 0.5, 0.3])
    ap.add_argument("--max-per-head-rel", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-inverse-of-train", action="store_true")
    ap.add_argument("--ref-dirs", nargs="*", help="QTO data dirs with test 1p queries (default: <kb>-{ind,ind_ded,ded}<suffix>)")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nent, nrel = read_stats(data_dir)
    rules = load_rules(args.tbox)
    train = read_train(data_dir)
    mats = load_qto_adj(args.adj, nent)
    if len(mats) != nrel:
        raise SystemExit(f"adjacency has {len(mats)} relations, stats.txt {nrel}")
    thr = sorted(set(args.thresholds), reverse=True)
    res = topk_new_facts(mats, train, rules, min(thr), args.max_per_head_rel, not args.keep_inverse_of_train)
    del mats
    n_train_rel = np.bincount(train[:, 1], minlength=nrel)
    ref_dirs = [Path(x) for x in args.ref_dirs] if args.ref_dirs is not None else default_ref_dirs(data_dir)
    ref, qkeys, used = reference_set(ref_dirs, rules)
    train_set = set(map(tuple, train.tolist()))
    ref -= train_set
    counts: Dict[str, Any] = {
        "adj": args.adj, "data_dir": str(data_dir), "max_per_head_rel": args.max_per_head_rel, "n_train": int(len(train)),
        "exclusions_at_min_threshold": {k: res[k] for k in ("n_excluded_train", "n_excluded_inverse_of_train", "n_excluded_cap")},
        "reference": {"dirs": used, "n_ref_facts_not_in_train": len(ref), "n_1p_query_keys": len(qkeys),
                      "definition": "easy+hard answers of the 1p test queries of the listed dirs, inverse-closed, minus training triples"},
        "thresholds": {},
    }
    inv = rules.inverse
    print(f"{'thr':>5s} {'n_new':>9s} {'/train':>7s} {'prec_all':>9s} {'n_judgeable':>11s} {'prec_judg':>9s}")
    for t in thr:
        sel = res["score"] >= t
        h, r, tt, s = res["h"][sel], res["r"][sel], res["t"][sel], res["score"][sel]
        write_tsv(out / f"pred_t{t:g}.tsv", h, r, tt, s)
        per_rel = np.bincount(r, minlength=nrel)
        entry: Dict[str, Any] = {
            "n_new": int(sel.sum()), "ratio_to_train": round(float(sel.sum()) / max(len(train), 1), 4),
            "per_relation": {str(i): {"n_new": int(per_rel[i]), "n_train": int(n_train_rel[i]),
                                      "ratio": (round(per_rel[i] / n_train_rel[i], 4) if n_train_rel[i] else None)}
                             for i in range(nrel) if per_rel[i] or n_train_rel[i]},
        }
        if used:
            trip = list(zip(h.tolist(), r.tolist(), tt.tolist()))
            hit = np.array([x in ref for x in trip], dtype=bool)
            judg = np.array([(a, b) in qkeys or (b in inv and (c, inv[b]) in qkeys) for a, b, c in trip], dtype=bool)
            entry["precision_all"] = round(float(hit.mean()), 4) if len(hit) else None
            entry["n_judgeable"] = int(judg.sum())
            entry["precision_judgeable"] = round(float(hit[judg].mean()), 4) if judg.any() else None
            entry["n_correct"] = int(hit.sum())
        counts["thresholds"][f"{t:g}"] = entry
        print(f"{t:5g} {entry['n_new']:9d} {entry['ratio_to_train']:7.4f} {str(entry.get('precision_all')):>9s} "
              f"{str(entry.get('n_judgeable')):>11s} {str(entry.get('precision_judgeable')):>9s}")
    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    print(f"exclusions at threshold {min(thr):g}: {counts['exclusions_at_min_threshold']}; reference {counts['reference']['n_ref_facts_not_in_train']} facts from {used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
