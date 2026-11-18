"""Exact-reasoning ceiling on E-OMQA: answer every test query over the forward-chaining closure of the input facts.

The input facts are the 1p training answers of ``train_plain`` (setting ``plain``) or of ``train_plain`` and
``train_onto`` (setting ``onto``), optionally with extra triples (``h<TAB>r<TAB>t[<TAB>score]``, the format of
``build_eomqa.py --extra-triples``). They are closed under the inverse pairing and the TBox of a resolved tbox json
(``experiments.omqa.hybrid.tbox``): inverses, role inclusions ``r ⊑ s``, range axioms ``∃r⁻ ⊑ A`` and class
inclusions ``A ⊑ B`` (all one-premise rules, so semi-naive forward chaining is complete). Each test query is then
answered exactly (set semantics) over the closure.

Metric (as ``~/tmp/gfe/omqa/diag/leakage_audit.py``, the source of the paper's "exact reasoning" numbers): filtered
protocol (easy answers removed from the computed set), the computed set ranked first in uniformly random order, the
expected HITS@3 of each hard answer: ``min(1, 3 / (n - inter + 1))`` for a hard answer inside a set of ``n`` filtered
answers of which ``inter`` are hard, 0 outside; averaged over the hard answers of a query, macro over the queries of a
shape (queries without hard answers skipped), then macro over the shapes. Also per shape: recall of the hard answers,
precision, mean filtered set size and mean number of false positives (computed answers that are neither easy nor hard
answers) per query.

    python -m experiments.omqa.hybrid.exact_ceiling --data-root .../data_lubm --setting onto \\
        --tbox ~/tmp/gfe/hybrid/tbox_lubm.json [--extra-triples pred_t0.5.tsv] --out res.json
    python -m experiments.omqa.hybrid.exact_ceiling --summarize res1.json res2.json ...
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from experiments.omqa.build_eomqa import collect_1p, norm, read_extra_triples, safe_load
from experiments.omqa.hybrid.tbox import Rules, load_rules

SHAPE_NAMES = {
    "('e', ('r',))": "1p", "('e', ('r', 'r'))": "2p", "('e', ('r', 'r', 'r'))": "3p",
    "(('e', ('r',)), ('e', ('r',)))": "2i", "(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))": "3i",
    "((('e', ('r',)), ('e', ('r',))), ('r',))": "ip", "(('e', ('r', 'r')), ('e', ('r',)))": "pi",
    "(('e', ('r',)), ('e', ('r',)), ('u',))": "2u", "((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))": "up",
}
SHAPE_ORDER = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
Triple = Tuple[int, int, int]


def shape_name(s: Any) -> str:
    return SHAPE_NAMES.get(str(s), str(s))


# --------------------------------------------------------------------------------------------- closure
def closure(facts: Iterable[Triple], rules: Rules) -> Set[Triple]:
    """Semi-naive forward chaining of one-premise rules (same rule set as ``leakage_audit.closure``)."""
    t_inv, t_fwd, inv = rules.type_inv_rel, rules.type_rel, rules.inverse
    ri: Dict[int, Set[int]] = collections.defaultdict(set)
    for r, s in rules.role_inclusion:
        ri[r].add(s)
    rng: Dict[int, Set[int]] = collections.defaultdict(set)
    for r, a in rules.range_axiom:
        rng[r].add(a)
    ci: Dict[int, Set[int]] = collections.defaultdict(set)
    for a, b in rules.class_inclusion:
        ci[a].add(b)
    T = set(facts)
    frontier = set(T)
    while frontier:
        new: Set[Triple] = set()
        for e, r, a in frontier:
            ir = inv.get(r)
            if ir is not None:
                new.add((a, ir, e))
            for s in ri.get(r, ()):
                new.add((e, s, a))
            for A in rng.get(r, ()):  # ∃r⁻ ⊑ A: (e, r, a) gives a : A
                if t_inv is not None:
                    new.add((A, t_inv, a))
                if t_fwd is not None:
                    new.add((a, t_fwd, A))
            if t_inv is not None and r == t_inv:
                for B in ci.get(e, ()):
                    new.add((B, t_inv, a))
            if t_fwd is not None and r == t_fwd:
                for B in ci.get(a, ()):
                    new.add((e, t_fwd, B))
        frontier = new - T
        T |= frontier
    return T


def adjacency(triples: Iterable[Triple]) -> Dict[int, Dict[int, Set[int]]]:
    adj: Dict[int, Dict[int, Set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for e, r, a in triples:
        adj[r][e].add(a)
    return adj


# --------------------------------------------------------------------------------------------- query answering
def _is_rel_tuple(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) > 0 and all(isinstance(y, str) and y == "r" for y in x)


def _project(cur: Set[int], rels: Sequence[int], adj: Dict[int, Dict[int, Set[int]]]) -> Set[int]:
    for r in rels:
        nxt: Set[int] = set()
        m = adj.get(r, {})
        for e in cur:
            nxt |= m.get(e, set())
        cur = nxt
    return cur


def answer(s: Any, q: Any, adj: Dict[int, Dict[int, Set[int]]]) -> Set[int]:
    """Exact answers of query ``q`` with structure ``s`` (Query2Box format, normalised) over ``adj``."""
    if s[0] == "e":
        return _project({q[0]}, q[1], adj)
    if _is_rel_tuple(s[-1]):  # ip / up: branches then projection
        return _project(answer(s[0], q[0], adj), q[-1], adj)
    if s[-1] == ("u",):
        out: Set[int] = set()
        for i in range(len(s) - 1):
            out |= answer(s[i], q[i], adj)
        return out
    res: Optional[Set[int]] = None
    for i in range(len(s)):
        r = answer(s[i], q[i], adj)
        res = r if res is None else res & r
    return res if res is not None else set()


def expected_hits3(n: int, inter: int, n_hard: int) -> float:
    """Expected HITS@3 averaged over the hard answers, set of ``n`` filtered answers with ``inter`` hard ones first."""
    if not inter:
        return 0.0
    return inter * min(1.0, 3.0 / (n - inter + 1)) / n_hard


def evaluate_case(root: Path, case: str, adj: Dict[int, Dict[int, Set[int]]]) -> Dict[str, Any]:
    qs = safe_load(root / case / "test-queries.pkl")
    hard = safe_load(root / case / "test-hard-answers.pkl")
    easy = safe_load(root / case / "test-easy-answers.pkl")
    per_shape: Dict[str, Dict[str, float]] = {}
    for s, qset in qs.items():
        sn = shape_name(norm(s))
        acc = {"recall": 0.0, "precision": 0.0, "size": 0.0, "fp": 0.0, "hits3": 0.0, "n": 0}
        for q in qset:
            h = set(map(int, hard.get(q, ())))
            if not h:
                continue
            ea = set(map(int, easy.get(q, ())))
            res = answer(norm(s), norm(q), adj) - ea  # filtered: easy answers removed
            inter = len(res & h)
            n = len(res)
            acc["recall"] += inter / len(h)
            acc["precision"] += inter / n if n else 0.0
            acc["size"] += n
            acc["fp"] += n - inter
            acc["hits3"] += expected_hits3(n, inter, len(h))
            acc["n"] += 1
        nq = max(acc["n"], 1)
        per_shape[sn] = {k: (v / nq if k != "n" else v) for k, v in acc.items()}
    shapes = [s for s in SHAPE_ORDER if s in per_shape] + sorted(s for s in per_shape if s not in SHAPE_ORDER)
    macro = {k: sum(per_shape[s][k] for s in shapes) / len(shapes) for k in ("recall", "precision", "size", "fp", "hits3")}
    return {"per_shape": {s: per_shape[s] for s in shapes}, "macro": macro}


# --------------------------------------------------------------------------------------------- input facts
def input_facts(root: Path, setting: str) -> Set[Triple]:
    sets = [("train_plain", "train", ("answers",))]
    if setting == "onto":
        sets.append(("train_onto", "train", ("answers",)))
    one_p = collect_1p(root, sets)
    return {(e, r, a) for (e, r), ans in one_p.items() for a in ans}


def run(root: Path, setting: str, tbox: str, extra: Optional[List[str]], cases: Sequence[str]) -> Dict[str, Any]:
    t0 = time.time()
    rules = load_rules(tbox)
    facts = input_facts(root, setting)
    info: Dict[str, Any] = {"data_root": str(root), "setting": setting, "tbox": str(tbox), "extra_triples": extra,
                            "n_role_inclusion": len(rules.role_inclusion), "n_range_axiom": len(rules.range_axiom),
                            "n_class_inclusion": len(rules.class_inclusion), "n_input_facts": len(facts)}
    if extra:
        ex = read_extra_triples(extra)
        info["n_extra_read"] = len(ex)
        info["n_extra_new"] = len(ex - facts)
        facts |= ex
    clo = closure(facts, rules)
    info["n_closure"] = len(clo)
    adj = adjacency(clo)
    info["cases"] = {c: evaluate_case(root, c, adj) for c in cases}
    info["seconds"] = round(time.time() - t0, 1)
    return info


def format_result(res: Dict[str, Any]) -> str:
    lines = [f"setting {res['setting']}  tbox {res['tbox']}  extra {res['extra_triples']}",
             f"axioms: role_inclusion {res['n_role_inclusion']} range {res['n_range_axiom']} class_inclusion {res['n_class_inclusion']}"
             f"  facts: input {res['n_input_facts']}" + (f" + extra new {res['n_extra_new']}/{res['n_extra_read']}" if res.get("extra_triples") else "")
             + f"  closure {res['n_closure']}"]
    for case, r in res["cases"].items():
        lines.append(f"== {case}")
        lines.append(f"  {'shape':5s} {'n':>6s} {'H@3':>6s} {'recall':>6s} {'prec':>6s} {'|A|':>8s} {'FP/q':>8s}")
        for s, v in r["per_shape"].items():
            lines.append(f"  {s:5s} {v['n']:6d} {v['hits3']:6.3f} {v['recall']:6.3f} {v['precision']:6.3f} {v['size']:8.1f} {v['fp']:8.1f}")
        m = r["macro"]
        lines.append(f"  {'macro':5s} {'':6s} {m['hits3']:6.3f} {m['recall']:6.3f} {m['precision']:6.3f} {m['size']:8.1f} {m['fp']:8.1f}")
    return "\n".join(lines)


def summarize(paths: Sequence[str]) -> str:
    """One table over several result jsons: per case, rows = shape, columns = runs (H@3 / recall / FP per query)."""
    runs = [(Path(p).stem, json.loads(Path(p).read_text())) for p in paths]
    out = []
    cases = list(dict.fromkeys(c for _, r in runs for c in r["cases"]))
    for case in cases:
        out.append(f"== {case}   (cell: expected HITS@3 / hard recall / false positives per query)")
        out.append(f"{'shape':6s}" + "".join(f" {name:>24s}" for name, _ in runs))
        shapes = list(dict.fromkeys(s for _, r in runs if case in r["cases"] for s in r["cases"][case]["per_shape"]))
        for s in shapes + ["macro"]:
            cells = []
            for _, r in runs:
                c = r["cases"].get(case)
                v = None if c is None else (c["macro"] if s == "macro" else c["per_shape"].get(s))
                cells.append(f" {'-':>24s}" if v is None else f" {v['hits3']:.3f} / {v['recall']:.3f} / {v['fp']:7.1f}".rjust(25))
            out.append(f"{s:6s}" + "".join(cells))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="e.g. .../data_lubm (contains train_plain/, test_ded/, ...)")
    ap.add_argument("--setting", choices=["plain", "onto"], default="plain")
    ap.add_argument("--tbox", help="resolved tbox json (experiments.omqa.hybrid.tbox)")
    ap.add_argument("--extra-triples", nargs="+", default=None, metavar="FILE")
    ap.add_argument("--cases", nargs="+", default=["test_ded", "test_ind_ded"])
    ap.add_argument("--out", help="result json")
    ap.add_argument("--summarize", nargs="+", metavar="JSON", help="print one table over result jsons and exit")
    args = ap.parse_args(argv)
    if args.summarize:
        print(summarize(args.summarize))
        return 0
    if not (args.data_root and args.tbox):
        ap.error("--data-root and --tbox are required")
    res = run(Path(args.data_root), args.setting, args.tbox, args.extra_triples, args.cases)
    print(format_result(res), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
