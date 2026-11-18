"""Rule closure of a QTO neural adjacency under the mined E-OMQA ontology (deduction from predicted facts).

Rules (one premise each; see ``tbox.py`` for the id conventions):

1. role inclusion ``r ⊑ s``:      ``(a, r, b)`` gives ``(a, s, b)``
2. inverse pair ``s = inv(r)``:    ``(a, r, b)`` gives ``(b, s, a)``
3. range axiom ``∃r⁻ ⊑ A``:        ``(a, r, b)`` gives ``(b, type, A)`` (and, with rule 2, ``(A, type_inv, b)``)
4. class inclusion ``A ⊑ B``:      ``(x, type, A)`` gives ``(x, type, B)``

Max-combination: a derived fact gets ``decay * score(premise)``; an entry keeps the maximum of its old value and all
derived values. Only premises with score >= ``--min-score`` propagate. Rules are applied in passes (Gauss-Seidel:
each update sees the previous ones) until a pass changes no entry or ``--max-iter`` passes ran. With ``decay = 1``
the result is the least fixpoint under max. Everything stays sparse (CSR, float32); rule 3 uses length-n column
maxima, never an n x n dense matrix.

    python -m experiments.omqa.hybrid.closure_adj --adj IN.pt --tbox tbox_<kb>.json --data-dir <QTO data dir> \
        --out OUT.pt [--decay 1.0] [--min-score 0.0] [--max-iter 10] [--report report.json] [--eval-dir DIR]

``--eval-dir`` (optional, diagnostic): 1p test queries of a QTO data dir; reports the adjacency scores of
(anchor, relation, hard answer) and of random non-answers before and after the closure.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from experiments.omqa.hybrid.qto_io import load_qto_adj, read_stats, save_qto_adj
from experiments.omqa.hybrid.tbox import Rules, load_rules


def _canon(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m, dtype=np.float32)
    m.sum_duplicates()  # also sorts indices
    m.eliminate_zeros()
    return m


def coo_max(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape) -> sp.csr_matrix:
    """CSR matrix from COO entries, duplicates combined by max (scipy would sum them)."""
    if len(vals) == 0:
        return sp.csr_matrix(shape, dtype=np.float32)
    key = rows.astype(np.int64) * shape[1] + cols.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, vals = key[order], vals[order].astype(np.float32)
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    v = np.maximum.reduceat(vals, start)
    k = key[start]
    return _canon(sp.csr_matrix((v, (k // shape[1], k % shape[1])), shape=shape))


def col_max(m: sp.csr_matrix, n: int) -> np.ndarray:
    """Length-n vector of column maxima of a non-negative sparse matrix (0 for empty columns)."""
    v = np.zeros(n, dtype=np.float32)
    if m.nnz:
        np.maximum.at(v, m.indices, m.data)
    return v


class Closure:
    def __init__(self, mats: List[sp.csr_matrix], rules: Rules, decay: float = 1.0, min_score: float = 0.0,
                 log: Callable[[str], None] = print):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.m = [_canon(x) if not (isinstance(x, sp.csr_matrix) and x.has_canonical_format) else x for x in mats]
        self.n = self.m[0].shape[0]
        self.nrel = len(self.m)
        self.rules = rules
        self.decay = float(decay)
        self.min_score = float(min_score)
        self.log = log
        self.last_changed = [0] * self.nrel
        for r, s in rules.role_inclusion + list(rules.inverse.items()):
            if max(r, s) >= self.nrel:
                raise ValueError(f"rule uses relation {max(r, s)} but the adjacency has {self.nrel} relations")
        if rules.type_rel is not None and rules.type_rel >= self.nrel:
            raise ValueError("type relation outside the adjacency")

    # -- helpers
    def premise(self, r: int) -> sp.csr_matrix:
        """Entries of relation r that may propagate, already multiplied by decay."""
        m = self.m[r]
        if self.min_score > 0.0 and m.nnz and m.data.min() < self.min_score:
            m = m.copy()  # never touch the stored matrix (eliminate_zeros works in place)
            m.data[m.data < self.min_score] = 0
            m.eliminate_zeros()
        if self.decay != 1.0:
            m = m * np.float32(self.decay)
        return m

    def merge(self, r: int, d: sp.csr_matrix, it: int) -> bool:
        """m[r] = max(m[r], d); True if any entry changed."""
        if d.nnz == 0:
            return False
        old = self.m[r]
        new = old.maximum(d)
        if new.dtype != np.float32:
            new = new.astype(np.float32)
        new.sum_duplicates()
        # canonical CSR with union structure: equal nnz <=> same structure (union contains old)
        changed = new.nnz != old.nnz or not np.array_equal(new.data, old.data)
        if changed:
            self.m[r] = new
            self.last_changed[r] = it
        return changed

    # -- one pass
    def step(self, it: int) -> int:
        ru = self.rules
        n_changed = 0
        for r, s in ru.role_inclusion:  # rule 1
            n_changed += self.merge(s, self.premise(r), it)
        for r in sorted(ru.inverse):  # rule 2 (both directions of each pair are in the map)
            s = ru.inverse[r]
            n_changed += self.merge(s, _canon(self.premise(r).T), it)
        if ru.type_rel is not None and ru.range_axiom:  # rule 3
            rows, cols, vals = [], [], []
            by_rel: Dict[int, List[int]] = {}
            for r, a in ru.range_axiom:
                by_rel.setdefault(r, []).append(a)
            for r, classes in by_rel.items():
                v = col_max(self.premise(r), self.n)
                b = np.flatnonzero(v)
                for a in classes:
                    rows.append(b)
                    cols.append(np.full(len(b), a, dtype=np.int64))
                    vals.append(v[b])
            d = coo_max(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (self.n, self.n))
            n_changed += self.merge(ru.type_rel, d, it)
        if ru.type_rel is not None and ru.class_inclusion:  # rule 4, iterated to its own fixpoint (cheap)
            t = ru.type_rel
            for _ in range(len(ru.class_inclusion) + 1):
                p = self.premise(t).tocsc()
                rows, cols, vals = [], [], []
                for a, b in ru.class_inclusion:
                    lo, hi = p.indptr[a], p.indptr[a + 1]
                    rows.append(p.indices[lo:hi])
                    cols.append(np.full(hi - lo, b, dtype=np.int64))
                    vals.append(p.data[lo:hi])
                d = coo_max(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (self.n, self.n))
                ch = self.merge(t, d, it)
                n_changed += ch
                if not ch:
                    break
        return n_changed

    def run(self, max_iter: int = 10) -> Dict[str, Any]:
        it = 0
        converged = False
        for it in range(1, max_iter + 1):
            t0 = time.time()
            n_changed = self.step(it)
            self.log(f"[closure] pass {it}: {n_changed} matrix updates changed entries, "
                     f"nnz {sum(x.nnz for x in self.m):,} ({time.time() - t0:.1f}s)")
            if n_changed == 0:
                converged = True
                break
        return {"iterations": it, "converged": converged}


def closure(mats: List[sp.csr_matrix], rules: Rules, decay: float = 1.0, min_score: float = 0.0, max_iter: int = 10,
            log: Callable[[str], None] = print) -> tuple[List[sp.csr_matrix], Dict[str, Any]]:
    c = Closure(mats, rules, decay, min_score, log)
    info = c.run(max_iter)
    info["last_changed_pass"] = c.last_changed
    return c.m, info


def mat_stats(mats: Sequence[sp.csr_matrix]) -> List[Dict[str, int]]:
    return [{"nnz": int(m.nnz), "n_one": int((m.data == 1.0).sum())} for m in mats]


# ----------------------------------------------------------------------------------------------- diagnostics
def load_1p(data_dir: Path, split: str = "test") -> Dict[tuple, Dict[str, set]]:
    from experiments.omqa.build_eomqa import norm, safe_load
    qs = safe_load(data_dir / f"{split}-queries.pkl")
    hard = safe_load(data_dir / f"{split}-hard-answers.pkl")
    easy = safe_load(data_dir / f"{split}-easy-answers.pkl")
    out = {}
    for q in qs.get(("e", ("r",)), ()):
        out[norm(q)] = {"hard": {int(x) for x in hard.get(q, ())}, "easy": {int(x) for x in easy.get(q, ())}}
    return out


def eval_triples(data_dir: Path, n_random: int = 2000, seed: int = 0) -> Dict[str, np.ndarray]:
    """(anchor, rel, hard answer) triples of the 1p test queries and random non-answers (anchor, rel) from the 1p
    queries, entity uniform over all entities not among the easy or hard answers of that query."""
    q1 = load_1p(data_dir)
    n, _ = read_stats(data_dir)
    hard = np.array([(e, r, a) for (e, (r,)), d in q1.items() for a in sorted(d["hard"])], dtype=np.int64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    keys = sorted(q1)
    neg = []
    while len(neg) < n_random:
        e, (r,) = keys[rng.integers(len(keys))]
        x = int(rng.integers(n))
        if x not in q1[(e, (r,))]["hard"] and x not in q1[(e, (r,))]["easy"]:
            neg.append((e, r, x))
    return {"hard": hard, "random_non_answer": np.array(neg, dtype=np.int64)}


def score_stats(mats: Sequence[sp.csr_matrix], trip: np.ndarray) -> Dict[str, float]:
    s = np.zeros(len(trip), dtype=np.float32)
    for r in np.unique(trip[:, 1]):
        sel = trip[:, 1] == r
        s[sel] = np.asarray(mats[r][trip[sel, 0], trip[sel, 2]]).ravel()
    return {"n": int(len(s)), "mean": round(float(s.mean()), 4) if len(s) else None,
            "share_ge_0.5": round(float((s >= 0.5).mean()), 4) if len(s) else None,
            "share_eq_1": round(float((s == 1.0).mean()), 4) if len(s) else None,
            "share_gt_0": round(float((s > 0).mean()), 4) if len(s) else None}


def row_false_positives(mats: Sequence[sp.csr_matrix], q1: Dict[tuple, Dict[str, set]]) -> Dict[str, float]:
    """Per 1p test query (anchor, r): entries of row ``anchor`` of relation r that are not easy/hard answers, with
    score >= 0.5 and == 1 (mean per query and total); and recall of the answers at those cut-offs."""
    fp05 = fp1 = tp05 = tp1 = n_ans = 0
    for (e, (r,)), d in q1.items():
        m = mats[r]
        lo, hi = m.indptr[e], m.indptr[e + 1]
        cols, vals = m.indices[lo:hi], m.data[lo:hi]
        ans = d["hard"] | d["easy"]
        is_ans = np.fromiter((int(c) in ans for c in cols), dtype=bool, count=len(cols))
        fp05 += int(((vals >= 0.5) & ~is_ans).sum()); fp1 += int(((vals == 1.0) & ~is_ans).sum())
        tp05 += int(((vals >= 0.5) & is_ans).sum()); tp1 += int(((vals == 1.0) & is_ans).sum())
        n_ans += len(ans)
    nq = max(len(q1), 1)
    return {"n_queries": len(q1), "n_answers": n_ans, "non_answers_ge_0.5_per_query": round(fp05 / nq, 3),
            "non_answers_eq_1_per_query": round(fp1 / nq, 3), "answers_ge_0.5_per_query": round(tp05 / nq, 3),
            "answers_eq_1_per_query": round(tp1 / nq, 3)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adj", required=True, help="QTO neural adjacency cache (neural_adj/<key>_10_<thrshd>.pt)")
    ap.add_argument("--tbox", required=True, help="resolved tbox json (python -m experiments.omqa.hybrid.tbox)")
    ap.add_argument("--data-dir", required=True, help="QTO data dir (stats.txt gives nentity / nrelation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--fraction", type=int, default=10)
    ap.add_argument("--report")
    ap.add_argument("--eval-dir", help="QTO data dir whose 1p test queries are scored before/after (diagnostic)")
    args = ap.parse_args(argv)
    t_start = time.time()
    nent, nrel = read_stats(args.data_dir)
    rules = load_rules(args.tbox)
    t0 = time.time()
    mats = load_qto_adj(args.adj, nent)
    if len(mats) != nrel or mats[0].shape != (nent, nent):
        raise SystemExit(f"adjacency has {len(mats)} relations of shape {mats[0].shape}; stats.txt says {nrel} x {nent}")
    t_load = time.time() - t0
    print(f"[closure] loaded {args.adj}: {nrel} relations, nnz {sum(m.nnz for m in mats):,} ({t_load:.1f}s)")
    before = mat_stats(mats)
    ev = ev_before = None
    if args.eval_dir:
        ev = eval_triples(Path(args.eval_dir))
        q1 = load_1p(Path(args.eval_dir))
        ev_before = {k: score_stats(mats, v) for k, v in ev.items()}
        ev_before["rows_of_1p_queries"] = row_false_positives(mats, q1)
    t0 = time.time()
    mats, info = closure(mats, rules, args.decay, args.min_score, args.max_iter)
    t_close = time.time() - t0
    after = mat_stats(mats)
    t0 = time.time()
    save_qto_adj(mats, args.out, args.fraction)
    t_save = time.time() - t0
    report: Dict[str, Any] = {
        "adj": str(args.adj), "out": str(args.out), "tbox": str(args.tbox), "decay": args.decay, "min_score": args.min_score,
        "max_iter": args.max_iter, **info,
        "per_relation": [{"rel": r, "nnz_before": b["nnz"], "nnz_after": a["nnz"], "n_one_before": b["n_one"],
                          "n_one_after": a["n_one"], "last_changed_pass": info["last_changed_pass"][r]}
                         for r, (b, a) in enumerate(zip(before, after))],
        "nnz_before": sum(b["nnz"] for b in before), "nnz_after": sum(a["nnz"] for a in after),
        "n_one_before": sum(b["n_one"] for b in before), "n_one_after": sum(a["n_one"] for a in after),
        "seconds": {"load": round(t_load, 1), "closure": round(t_close, 1), "save": round(t_save, 1), "total": round(time.time() - t_start, 1)},
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
    }
    del report["last_changed_pass"]
    if ev is not None:
        ev_after = {k: score_stats(mats, v) for k, v in ev.items()}
        ev_after["rows_of_1p_queries"] = row_false_positives(mats, q1)
        report["eval_1p"] = {"data_dir": str(args.eval_dir), "before": ev_before, "after": ev_after}
        for k in ev_after:
            print(f"[eval] {k:18s} before {ev_before[k]}\n[eval] {'':18s} after  {ev_after[k]}")
    print(f"[closure] nnz {report['nnz_before']:,} -> {report['nnz_after']:,}; ones {report['n_one_before']:,} -> "
          f"{report['n_one_after']:,}; passes {info['iterations']} (converged {info['converged']}); "
          f"time {report['seconds']}; peak RSS {report['peak_rss_gb']} GB")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
