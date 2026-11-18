"""Ontology rules of the mined E-OMQA TBox: loading, id conventions and empirical verification.

Conventions (checked empirically on ``<kb>-ded-onto/train.txt``, see ``verify``):

* ``T_inverse_type`` in the mined json is ``type_inv``: triples ``(A, type_inv, x)`` have a class as head. ``type``
  is its inverse relation (LUBM: ``type`` = 23, ``type_inv`` = 27; NELL has no type relation).
* Inverse pairs: the mined json ``inverse`` map is partial on LUBM (only 23 <-> 27; it was computed from the plain
  1p training answers, which contain only one direction of most relations). The GroundedEL tables use the
  complete pairing of ``build_eomqa.py`` (greedy matching on LUBM, ``(2j, 2j+1)`` on NELL), recorded in the tables'
  ``build_report.json``. ``resolve`` merges that pairing into the tbox json.
* Range axiom ``(r, A)`` in ``range_axiom_exists_rinv_sub_A`` reads ``∃r⁻ ⊑ A``: ``(a, r, b)`` gives ``(b, type, A)``.
* Class inclusion ``(A, B)`` reads ``A ⊑ B``: ``(x, type, A)`` gives ``(x, type, B)``.
* Role inclusion ``(r, s)`` reads ``r ⊑ s``: ``(a, r, b)`` gives ``(a, s, b)``.

CLI (writes a resolved, verified tbox json used by ``closure_adj`` / ``predicted_facts``)::

    python -m experiments.omqa.hybrid.tbox --tbox tbox_lubm.json --build-report tables/lubm_plain/build_report.json \
        --check-train baselines/data/lubm-ded-onto/train.txt --out tbox_lubm.json [--out-all tbox_lubm_all.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class Rules:
    """One-premise rules over relation / entity ids."""

    type_rel: Optional[int] = None
    type_inv_rel: Optional[int] = None
    inverse: Dict[int, int] = field(default_factory=dict)  # symmetric: inverse[inverse[r]] == r
    role_inclusion: List[Tuple[int, int]] = field(default_factory=list)  # r ⊑ s
    range_axiom: List[Tuple[int, int]] = field(default_factory=list)  # (r, A): ∃r⁻ ⊑ A
    class_inclusion: List[Tuple[int, int]] = field(default_factory=list)  # A ⊑ B
    classes: List[int] = field(default_factory=list)

    def check(self) -> None:
        for r, s in self.inverse.items():
            if self.inverse.get(s) != r:
                raise ValueError(f"inverse map not symmetric at {r} -> {s} -> {self.inverse.get(s)}")
        if (self.range_axiom or self.class_inclusion) and self.type_rel is None:
            raise ValueError("range axioms / class inclusions need a type relation")
        if self.type_inv_rel is not None and self.type_rel is not None and self.inverse.get(self.type_rel) != self.type_inv_rel:
            raise ValueError("type and type_inv are not an inverse pair")


def rules_from_json(d: Dict[str, Any]) -> Rules:
    inv = {int(k): int(v) for k, v in d.get("inverse", {}).items() if v is not None}
    t_inv = d.get("T_inverse_type")
    t_fwd = d.get("type_rel", inv.get(t_inv) if t_inv is not None else None)
    rules = Rules(
        type_rel=None if t_fwd is None else int(t_fwd), type_inv_rel=None if t_inv is None else int(t_inv), inverse=inv,
        role_inclusion=[(int(r), int(s)) for r, s in d.get("role_inclusion", [])],
        range_axiom=[(int(r), int(a)) for r, a in d.get("range_axiom_exists_rinv_sub_A", [])],
        class_inclusion=[(int(a), int(b)) for a, b in d.get("class_inclusion", [])],
        classes=[int(c) for c in d.get("classes", [])])
    rules.check()
    return rules


def load_rules(path: str | Path, quiet: bool = False) -> Rules:
    d = json.loads(Path(path).read_text())
    rules = rules_from_json(d)
    n_none = sum(v is None for v in d.get("inverse", {}).values())
    if not d.get("resolved") and not quiet:
        print(f"[tbox] WARNING {path} is not resolved (python -m experiments.omqa.hybrid.tbox); "
              f"{n_none} relations have no inverse in it", file=sys.stderr)
    return rules


def resolve(mined: Dict[str, Any], build_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Mined tbox json + the tables' complete inverse pairing (build_report.json) -> resolved json dict."""
    out = dict(mined)
    inv = {str(k): v for k, v in mined.get("inverse", {}).items()}
    src = "mined json"
    if build_report is not None:
        rep_inv = {str(k): v for k, v in build_report["inverse"].items()}
        for k, v in inv.items():
            if v is not None and rep_inv.get(k) != v:
                raise ValueError(f"inverse of {k}: mined json says {v}, build report says {rep_inv.get(k)}")
        inv = rep_inv
        src = f"build_report ({build_report.get('inverse_method')})"
        if mined.get("T_inverse_type") != build_report.get("type_inv"):
            raise ValueError("type_inv differs between mined json and build report")
    out["inverse"] = inv
    t_inv = mined.get("T_inverse_type")
    out["type_rel"] = inv.get(str(t_inv)) if t_inv is not None else None
    out["inverse_source"] = src
    out["resolved"] = True
    return out


# ----------------------------------------------------------------------------------------------- verification
def rule_instances(rules: Rules, triples: np.ndarray) -> Iterable[Tuple[str, Tuple, np.ndarray]]:
    """Yield (rule type, axiom, conclusions array (n, 3)) for every axiom, one row per premise triple."""
    by_rel: Dict[int, np.ndarray] = {int(r): triples[triples[:, 1] == r] for r in np.unique(triples[:, 1])}
    empty = np.zeros((0, 3), dtype=np.int64)

    def concl(h, r, t):
        return np.stack([h, np.full_like(h, r), t], axis=1) if len(h) else empty

    for r, s in rules.role_inclusion:
        p = by_rel.get(r, empty)
        yield "role_inclusion", (r, s), concl(p[:, 0], s, p[:, 2])
    for r in sorted(rules.inverse):
        p = by_rel.get(r, empty)
        yield "inverse", (r, rules.inverse[r]), concl(p[:, 2], rules.inverse[r], p[:, 0])
    if rules.type_rel is not None:
        for r, a in rules.range_axiom:
            p = by_rel.get(r, empty)
            yield "range", (r, a), concl(p[:, 2], rules.type_rel, np.full(len(p), a, dtype=np.int64))
        if rules.type_inv_rel is not None:
            for r, a in rules.range_axiom:  # rule 3 followed by rule 2 (reported, not filtered separately)
                p = by_rel.get(r, empty)
                yield "range+inverse", (r, a), concl(np.full(len(p), a, dtype=np.int64), rules.type_inv_rel, p[:, 2])
        tp = by_rel.get(rules.type_rel, empty)
        for a, b in rules.class_inclusion:
            p = tp[tp[:, 2] == a]
            yield "class_inclusion", (a, b), concl(p[:, 0], rules.type_rel, np.full(len(p), b, dtype=np.int64))


def _keys(arr: np.ndarray, n_ent: int, n_rel: int) -> np.ndarray:
    return (arr[:, 0] * n_rel + arr[:, 1]) * n_ent + arr[:, 2]


def _member(keys: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    if len(sorted_ref) == 0 or len(keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.searchsorted(sorted_ref, keys).clip(max=len(sorted_ref) - 1)
    return sorted_ref[pos] == keys


def verify(rules: Rules, triples: np.ndarray) -> List[Dict[str, Any]]:
    """Per axiom: number of premises, share whose conclusion is present, and the same restricted to premises whose
    conclusion is *checkable*: its (head, relation) pair occurs in ``triples``. The E-OMQA enriched training facts
    are the certain answers of the sampled 1p training queries only, so a conclusion (h, s, t) can only be present
    when (h, s) was sampled as a query; the checkable share is the meaningful test of a rule reading."""
    ent_ids = [int(triples[:, [0, 2]].max())] + [a for _, a in rules.range_axiom] + [x for ab in rules.class_inclusion for x in ab]
    n_ent = max(ent_ids) + 1
    rel_ids = [int(triples[:, 1].max())] + [x for rs in rules.inverse.items() for x in rs] + [x for rs in rules.role_inclusion for x in rs]
    rel_ids += [r for r, _ in rules.range_axiom] + [x for x in (rules.type_rel, rules.type_inv_rel) if x is not None]
    n_rel = max(rel_ids) + 1
    all_keys = np.unique(_keys(triples, n_ent, n_rel))
    hr_keys = np.unique(triples[:, 0] * n_rel + triples[:, 1])
    rows = []
    for kind, ax, c in rule_instances(rules, triples):
        n = len(c)
        present = _member(_keys(c, n_ent, n_rel), all_keys) if len(c) else np.zeros(0, bool)
        chk = _member(c[:, 0] * n_rel + c[:, 1], hr_keys) if len(c) else np.zeros(0, bool)
        rows.append({"kind": kind, "axiom": list(ax), "n": n, "present": int(present.sum()),
                     "n_checkable": int(chk.sum()), "present_checkable": int((present & chk).sum())})
    return rows


def summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    for r in rows:
        a = agg.setdefault(r["kind"], {"kind": r["kind"], "n_axioms": 0, "n": 0, "present": 0, "n_checkable": 0, "present_checkable": 0})
        a["n_axioms"] += 1
        for k in ("n", "present", "n_checkable", "present_checkable"):
            a[k] += r[k]
    for a in agg.values():
        a["share"] = round(a["present"] / a["n"], 4) if a["n"] else None
        a["share_checkable"] = round(a["present_checkable"] / a["n_checkable"], 4) if a["n_checkable"] else None
    return list(agg.values())


def format_table(summary: List[Dict[str, Any]], title: str) -> str:
    lines = [title, f"{'rule type':16s} {'axioms':>6s} {'instances':>10s} {'share':>7s} {'checkable':>10s} {'share_chk':>9s}"]
    for a in summary:
        sh = "-" if a["share"] is None else f"{a['share']:.4f}"
        sc = "-" if a["share_checkable"] is None else f"{a['share_checkable']:.4f}"
        lines.append(f"{a['kind']:16s} {a['n_axioms']:6d} {a['n']:10d} {sh:>7s} {a['n_checkable']:10d} {sc:>9s}")
    return "\n".join(lines)


def filter_axioms(d: Dict[str, Any], rows: List[Dict[str, Any]], min_share: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Drop the individual axioms whose checkable share is below ``min_share`` (axioms without checkable premises
    are kept). Returns the filtered json dict and the dropped rows."""
    key = {"role_inclusion": "role_inclusion", "range": "range_axiom_exists_rinv_sub_A", "class_inclusion": "class_inclusion"}
    bad: Dict[str, Set[Tuple[int, int]]] = collections.defaultdict(set)
    dropped = []
    for r in rows:
        if r["kind"] in key and r["n_checkable"] and r["present_checkable"] / r["n_checkable"] < min_share:
            bad[r["kind"]].add(tuple(r["axiom"]))
            dropped.append(dict(r, share_checkable=round(r["present_checkable"] / r["n_checkable"], 4)))
        if r["kind"] == "inverse" and r["n_checkable"] and r["present_checkable"] / r["n_checkable"] < min_share:
            raise ValueError(f"inverse pair {r['axiom']} fails the check: misread inverse convention")
    out = dict(d)
    for kind, k in key.items():
        out[k] = [list(ax) for ax in d.get(k, []) if tuple(int(x) for x in ax) not in bad[kind]]
    out["dropped_axioms"] = dropped
    out["verification_min_share"] = min_share
    return out, dropped


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tbox", required=True, help="mined tbox json (tools/mine_axioms.py)")
    ap.add_argument("--build-report", help="build_report.json of the GroundedEL tables (complete inverse pairing)")
    ap.add_argument("--check-train", required=True, help="enriched training triples, e.g. <kb>-ded-onto/train.txt")
    ap.add_argument("--min-share", type=float, default=0.99)
    ap.add_argument("--out", required=True, help="resolved tbox json with the axioms that pass the check")
    ap.add_argument("--out-all", help="resolved tbox json with all mined axioms (no filtering)")
    args = ap.parse_args(argv)
    mined = json.loads(Path(args.tbox).read_text())
    rep = json.loads(Path(args.build_report).read_text()) if args.build_report else None
    d = resolve(mined, rep)
    triples = np.loadtxt(args.check_train, dtype=np.int64, delimiter="\t", ndmin=2).reshape(-1, 3)
    rows = verify(rules_from_json(d), triples)
    print(format_table(summarise(rows), f"== all mined axioms, checked on {args.check_train} ({len(triples)} triples)"))
    out, dropped = filter_axioms(d, rows, args.min_share)
    rows2 = verify(rules_from_json(out), triples)
    summ2 = summarise(rows2)
    print(format_table(summ2, f"== after dropping {len(dropped)} axioms with checkable share < {args.min_share}"))
    for r in dropped:
        print(f"   dropped {r['kind']:15s} {tuple(r['axiom'])}: {r['present_checkable']}/{r['n_checkable']} = {r['share_checkable']}")
    for a in summ2:
        if a["share_checkable"] is not None and a["share_checkable"] < args.min_share:
            raise SystemExit(f"rule type {a['kind']} still fails the check: {a}")
    d["verification"] = summarise(rows)
    out["verification"] = summ2
    out["verification_all_mined"] = summarise(rows)
    out["check_train"] = str(args.check_train)
    if args.out_all:
        Path(args.out_all).write_text(json.dumps(d, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}" + (f" and {args.out_all}" if args.out_all else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
