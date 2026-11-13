"""Build GroundedEL tables for the E-OMQA benchmarks (Andresel et al., CIKM 2023) from the released pickles.

The release (https://github.com/medinaandresel/eomqa) contains Query2Box-format pickles only: queries and
answers over integer ids, plus ontology-based query generalisations. No triples, ontology or names are shipped.
This script reconstructs, reasoner-free and without executing pickled code (allow-list unpickler):

* ABox: every 1p query ``(e, (r,))`` with its answers from ``train_plain/train-{queries,answers}.pkl`` (setting
  ``plain``) or additionally the certain answers of ``train_onto`` (setting ``onto``);
* inverse relation pairs: ``(2j, 2j+1)`` when that convention holds (NELL); otherwise inferred from all 1p
  (query, answer) pairs available (train answers, valid/test easy and hard answers): ``s = inv(r)`` when the
  fraction of ``(e, r, a)`` with ``(a, s, e)`` present exceeds ``--inverse-min-match``;
* TBox from the generalisation files of all sets (Table 2 of the paper): role inclusions ``r ⊑ s`` (same anchor,
  relation changed), range axioms ``∃r⁻ ⊑ A`` (anchor replaced by a class entity ``A`` with the type⁻ relation),
  class inclusions ``A ⊑ A'`` (type⁻ anchors generalised);
* tables: class entities (+ ``owl:Thing``) → classes, other entities → individuals, type triples → class
  assertions, ``∃r⁻ ⊑ A`` → ``gci3 (r⁻, ⊤, A)``, class inclusions → ``gci0``, role inclusions, triples (both
  directions when the inverse is known) → ``object_property_assertion``.

Output: ``<out>/tables.pt`` (torch tensors + index dicts) and ``<out>/build_report.json``.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
try:
    import numpy._core.multiarray as _npma  # numpy >= 2
except ImportError:  # pragma: no cover
    import numpy.core.multiarray as _npma
import torch as th

_ALLOWED = {
    ("collections", "defaultdict"): collections.defaultdict, ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
    ("builtins", "list"): list, ("builtins", "dict"): dict, ("builtins", "tuple"): tuple, ("builtins", "int"): int,
    ("builtins", "str"): str, ("numpy.core.multiarray", "scalar"): _npma.scalar, ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _npma._reconstruct, ("numpy", "ndarray"): np.ndarray,
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"forbidden global {module}.{name}")


def safe_load(path: Path) -> Any:
    """Load a Query2Box pickle with an allow-list unpickler (no arbitrary code execution)."""
    with open(path, "rb") as fh:
        return _SafeUnpickler(fh).load()


def norm(q: Any) -> Any:
    """Convert numpy ints inside a query structure to Python ints (recursively)."""
    if isinstance(q, tuple):
        return tuple(norm(x) for x in q)
    if isinstance(q, (np.integer, int)):
        return int(q)
    return q


def is_1p(q: Any) -> bool:
    return isinstance(q, tuple) and len(q) == 2 and not isinstance(q[0], tuple) and isinstance(q[1], tuple) and len(q[1]) == 1


def collect_1p(root: Path, sets: Iterable[Tuple[str, str, Tuple[str, ...]]]) -> Dict[Tuple[int, int], Set[int]]:
    """Collect 1p (anchor, relation) -> answers over the given (dir, prefix, answer-file-suffixes)."""
    out: Dict[Tuple[int, int], Set[int]] = collections.defaultdict(set)
    for d, prefix, suffixes in sets:
        qpath = root / d / f"{prefix}-queries.pkl"
        if not qpath.is_file():
            continue
        queries = safe_load(qpath)
        answers = [safe_load(root / d / f"{prefix}-{s}.pkl") for s in suffixes if (root / d / f"{prefix}-{s}.pkl").is_file()]
        for q in queries.get(("e", ("r",)), ()):
            qn = norm(q)
            e, (r,) = qn
            for ans in answers:
                if q in ans:
                    out[(e, r)].update(int(a) for a in ans[q])
                elif qn in ans:
                    out[(e, r)].update(int(a) for a in ans[qn])
    return out


def infer_inverses(triples: Set[Tuple[int, int, int]], rels: List[int], min_match: float) -> Dict[int, Optional[int]]:
    """Infer inverse pairs: s = inv(r) if most (e, r, a) have (a, s, e)."""
    by_rel: Dict[int, Set[Tuple[int, int]]] = collections.defaultdict(set)
    for e, r, a in triples:
        by_rel[r].add((e, a))
    inv: Dict[int, Optional[int]] = {}
    match: Dict[int, float] = {}
    for r in rels:
        best, bestf = None, 0.0
        for s in rels:
            if s == r or not by_rel[r]:
                continue
            f = sum((a, e) in by_rel[s] for e, a in by_rel[r]) / len(by_rel[r])
            if f > bestf:
                best, bestf = s, f
        inv[r] = best if bestf >= min_match else None
        match[r] = round(bestf, 3)
    # enforce symmetry
    for r, s in list(inv.items()):
        if s is not None and inv.get(s) not in (r, None):
            inv[r] = None
        if s is not None and inv.get(s) is None:
            inv[s] = r
    return inv, match


def infer_inverses_matching(triples: Set[Tuple[int, int, int]], rels: List[int]) -> Tuple[Dict[int, Optional[int]], Dict[int, float]]:
    """Greedy perfect matching on the symmetrised match score (every relation gets exactly one inverse).

    Appropriate when the KG is known to be completed with inverse edges but the released 1p data covers only
    part of it (E-OMQA LUBM): the one-sided match fractions are then well below 1 but the pairing is still the
    unique maximum of the symmetric score. Consistency with the mined role inclusions is checked by the caller.
    """
    by_rel: Dict[int, Set[Tuple[int, int]]] = collections.defaultdict(set)
    for e, r, a in triples:
        by_rel[r].add((e, a))
    sym: Dict[Tuple[int, int], float] = {}
    for i, r in enumerate(rels):
        for s in rels[i + 1:]:
            if not by_rel[r] or not by_rel[s]:
                continue
            f_rs = sum((a, e) in by_rel[s] for e, a in by_rel[r]) / len(by_rel[r])
            f_sr = sum((a, e) in by_rel[r] for e, a in by_rel[s]) / len(by_rel[s])
            sym[(r, s)] = (f_rs + f_sr) / 2
    inv: Dict[int, Optional[int]] = {r: None for r in rels}
    match: Dict[int, float] = {r: 0.0 for r in rels}
    used: Set[int] = set()
    for (r, s), v in sorted(sym.items(), key=lambda kv: -kv[1]):
        if r in used or s in used:
            continue
        inv[r], inv[s] = s, r
        match[r] = match[s] = round(v, 3)
        used |= {r, s}
    return inv, match


def _query_entities(q: Any, acc: Set[int]) -> None:
    """Collect the anchor entity ids of a Query2Box-structured query instance."""
    if isinstance(q, tuple) and len(q) == 2 and not isinstance(q[0], tuple) and isinstance(q[1], tuple):
        acc.add(int(q[0]))
        return
    if isinstance(q, tuple):
        for x in q:
            if isinstance(x, tuple):
                _query_entities(x, acc)


def collect_all_entities(root: Path) -> Set[int]:
    """Return every entity id occurring as a query anchor or as an answer in any ``*-queries.pkl`` / ``*answers.pkl``."""
    acc: Set[int] = set()
    for f in sorted(root.rglob("*.pkl")):
        if "queries" in f.name:
            for _, qs in safe_load(f).items():
                for q in qs:
                    _query_entities(q, acc)
        elif "answers" in f.name:
            for ans in safe_load(f).values():
                acc.update(int(a) for a in ans)
    return acc


def mine_tbox(root: Path) -> Dict[str, Any]:
    """Mine role inclusions, range axioms and class inclusions from all generalisation files."""
    files = [(d, p) for d in ("train_plain", "train_gen", "train_onto", "train_spe") for p in ("train",)]
    files += [(d, p) for d in ("test_ind", "test_ded", "test_ind_ded") for p in ("test", "valid")]
    pairs: collections.Counter = collections.Counter()
    rng: collections.Counter = collections.Counter()
    hier: collections.Counter = collections.Counter()
    for d, p in files:
        f = root / d / f"{p}-generalizations.pkl"
        if not f.is_file():
            continue
        gens = safe_load(f)
        for q, gset in gens.items():
            qn = norm(q)
            if isinstance(qn[0], tuple):
                continue
            e, r = qn
            for g, _shape in gset:
                gn = norm(g)
                if isinstance(gn[0], tuple):
                    continue
                e2, r2 = gn
                if len(r2) == len(r) and r2[1:] == r[1:]:
                    if e2 == e and r2[0] != r[0]:
                        pairs[(r[0], r2[0])] += 1
                    elif e2 != e and r2[0] != r[0]:
                        rng[(r[0], e2, r2[0])] += 1
                    elif e2 != e and r2[0] == r[0]:
                        hier[(e, e2, r[0])] += 1
    t_cands = collections.Counter(t for (_, _, t) in rng)
    t_inv = t_cands.most_common(1)[0][0] if t_cands else None
    classes = sorted({A for (_, A, t) in rng if t == t_inv} | {a for (a, b, t) in hier if t == t_inv} | {b for (a, b, t) in hier if t == t_inv})
    return {
        "type_inv": t_inv,
        "classes": classes,
        "role_inclusion": sorted((r, s) for (r, s) in pairs if r != t_inv and s != t_inv),
        "range_axiom": sorted((r, A) for (r, A, t) in rng if t == t_inv and r != t_inv),  # ∃r⁻ ⊑ A
        "class_inclusion": sorted((a, b) for (a, b, t) in hier if t == t_inv),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="e.g. .../data_lubm (contains train_plain/, test_ded/, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", choices=["plain", "onto"], default="plain")
    ap.add_argument("--inverse-min-match", type=float, default=0.8)
    ap.add_argument("--even-odd-inverses", action="store_true", help="force the (2j, 2j+1) inverse convention")
    ap.add_argument("--inverse-method", choices=["threshold", "matching"], default="matching",
                    help="threshold: accept s=inv(r) when the one-sided match >= --inverse-min-match; matching: greedy perfect matching on the symmetric score")
    args = ap.parse_args(argv)
    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tbox = mine_tbox(root)
    t_inv = tbox["type_inv"]
    train_sets = [("train_plain", "train", ("answers",))]
    if args.setting == "onto":
        train_sets.append(("train_onto", "train", ("answers",)))
    abox_1p = collect_1p(root, train_sets)
    triples = {(e, r, a) for (e, r), ans in abox_1p.items() for a in ans}
    # all 1p pairs incl. valid/test easy+hard answers, only for inverse inference (never for training)
    all_sets = train_sets + [(d, p, ("easy-answers", "hard-answers")) for d in ("test_ind", "test_ded", "test_ind_ded") for p in ("valid", "test")]
    all_1p = collect_1p(root, all_sets)
    all_triples = {(e, r, a) for (e, r), ans in all_1p.items() for a in ans}
    rels = sorted({r for _, r, _ in all_triples} | {r for r, _ in tbox["role_inclusion"]} | {s for _, s in tbox["role_inclusion"]} | {r for r, _ in tbox["range_axiom"]})
    if args.even_odd_inverses:
        inv = {r: (r + 1 if r % 2 == 0 else r - 1) for r in rels}
        match = {}
    elif args.inverse_method == "matching":
        inv, match = infer_inverses_matching(all_triples, rels)
    else:
        inv, match = infer_inverses(all_triples, rels, args.inverse_min_match)
    # consistency of the inverse pairing with the mined role inclusions: r ⊑ s should come with inv(r) ⊑ inv(s)
    incl = set(map(tuple, tbox["role_inclusion"]))
    incl_consistent = sum((inv.get(r), inv.get(s)) in incl for r, s in incl)
    type_fwd = inv.get(t_inv) if t_inv is not None else None

    # classes and individuals
    class_ents = set(tbox["classes"])
    if t_inv is not None:
        class_ents |= {e for e, r, _ in triples if r == t_inv}
    ents = {e for e, _, _ in triples} | {a for _, _, a in triples}
    # every entity that occurs anywhere in the release (query anchors or answers of any split/setting) is a KB
    # entity: those without training facts become individuals with no assertions (untrained elements), exactly
    # as an embedding baseline keeps an untrained vector for them
    all_ents = collect_all_entities(root)
    n_factless = len(all_ents - ents - class_ents)
    ents |= all_ents
    individuals = sorted(ents - class_ents)
    class_list = sorted(class_ents)
    class_index = {f"C{c}": i for i, c in enumerate(class_list)}
    class_index["owl:Thing"] = len(class_index)
    top_id = class_index["owl:Thing"]
    cid = {c: class_index[f"C{c}"] for c in class_list}
    ind_index = {f"I{e}": i for i, e in enumerate(individuals)}
    iid = {e: ind_index[f"I{e}"] for e in individuals}
    role_ids = [r for r in rels if r not in (t_inv, type_fwd)]
    role_index = {f"R{r}": i for i, r in enumerate(role_ids)}
    rid = {r: role_index[f"R{r}"] for r in role_ids}
    inverse_of = [rid[inv[r]] if inv.get(r) is not None and inv[r] in rid else -1 for r in role_ids]

    class_assertion: Set[Tuple[int, int]] = set()
    role_assertion: Set[Tuple[int, int, int]] = set()
    n_type_dropped = 0
    for e, r, a in triples:
        if r == t_inv:  # (A, type⁻, x): x : A
            if e in cid and a in iid:
                class_assertion.add((iid[a], cid[e]))
            else:
                n_type_dropped += 1
        elif r == type_fwd:  # (x, type, A)
            if a in cid and e in iid:
                class_assertion.add((iid[e], cid[a]))
            else:
                n_type_dropped += 1
        elif e in iid and a in iid and r in rid:
            role_assertion.add((iid[e], rid[r], iid[a]))
            if inv.get(r) is not None and inv[r] in rid:
                role_assertion.add((iid[a], rid[inv[r]], iid[e]))
    gci0 = sorted({(cid[a], cid[b]) for a, b in tbox["class_inclusion"] if a in cid and b in cid})
    gci3 = sorted({(rid[inv[r]], top_id, cid[A]) for r, A in tbox["range_axiom"] if r in rid and inv.get(r) in rid and A in cid})
    # range axiom ∃r⁻ ⊑ A could not be expressed when r has no inverse id: record them
    gci3_unexpressed = [(r, A) for r, A in tbox["range_axiom"] if not (r in rid and inv.get(r) in rid and A in cid)]
    role_inclusion = sorted({(rid[r], rid[s]) for r, s in tbox["role_inclusion"] if r in rid and s in rid})

    tables = {
        "class_index_dict": class_index, "individual_index_dict": ind_index, "object_property_index_dict": role_index,
        "top_id": top_id, "inverse_of": inverse_of, "type_inv_rel": t_inv, "type_fwd_rel": type_fwd,
        "entity_to_individual": {int(k): int(v) for k, v in iid.items()}, "entity_to_class": {int(k): int(v) for k, v in cid.items()},
        "relation_to_role": {int(k): int(v) for k, v in rid.items()},
        "gci0": th.tensor(gci0, dtype=th.long).reshape(-1, 2), "gci3": th.tensor(gci3, dtype=th.long).reshape(-1, 3),
        "role_inclusion": th.tensor(role_inclusion, dtype=th.long).reshape(-1, 2),
        "class_assertion": th.tensor(sorted(class_assertion), dtype=th.long).reshape(-1, 2),
        "object_property_assertion": th.tensor(sorted(role_assertion), dtype=th.long).reshape(-1, 3),
    }
    th.save(tables, out / "tables.pt")
    report = {
        "data_root": str(root), "setting": args.setting, "n_entities": len(ents), "n_individuals": len(individuals),
        "n_classes": len(class_list), "n_roles": len(role_ids), "relations": rels, "inverse": {str(k): v for k, v in inv.items()},
        "inverse_match": {str(k): v for k, v in match.items()}, "type_inv": t_inv, "type_fwd": type_fwd,
        "n_triples_1p": len(triples), "n_role_assertions": len(role_assertion), "n_class_assertions": len(class_assertion),
        "n_factless_entities": n_factless, "n_type_dropped": n_type_dropped, "n_gci0": len(gci0), "n_gci3_range_domain": len(gci3), "range_axioms_unexpressed": gci3_unexpressed,
        "n_role_inclusion": len(role_inclusion), "inverse_method": "even-odd" if args.even_odd_inverses else args.inverse_method,
        "role_inclusions_with_inverse_counterpart": f"{incl_consistent}/{len(incl)}", "mined_counts": {k: len(v) for k, v in tbox.items() if isinstance(v, list)},
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("relations", "inverse_match")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
