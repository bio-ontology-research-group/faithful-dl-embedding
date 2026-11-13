#!/usr/bin/env python3
"""Seeded sampling of EL instance-retrieval queries, filtered by a one-pass materialisation.

Inputs
------
``<data-dir>/kbs/<kb>/<kb>_el.owl`` and ``<kb>.owl`` (see ``materialise.py``), ``--seed``,
``--per-type`` and ``--max-draws``.

Outputs
-------
``<data-dir>/kbs/<kb>/queries.json`` with ``sampler = "materialisation-filter"``, the
materialisation summary and, per query, ``query_id``, ``type``, ``expr`` and the filter's
answer set (``filter_answers``). These answers are NOT reference answers: ``ground_truth.py``
obtains the references from a reasoner on each class expression.

Method
------
The reasoner is called once per KB (ELK realisation of the EL projection, inside
``materialise.py``), never per candidate. For each query type, random candidates are drawn from
classes with at least one materialised instance and roles with at least one materialised edge;
a candidate is kept iff its filter answer set is non-empty, not the set of all individuals, and
different from every answer set already kept for that type. Conjunctions are canonicalised by
sorting their operands and ``A ⊓ A`` is not drawn. At most ``--max-draws`` candidates are drawn
per type (default ``100 * per_type``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import DATA_ROOT, NAMES, start_jvm

TYPES = ("A", "A_and_B", "some_r_A", "A_and_some_r_B", "some_r_and_B", "some_r_some_s_A", "some_inv_r_A")
SAMPLER = "materialisation-filter"


def query_id(kb: str, typ: str, expr: Any) -> str:
    """Stable 16-hex-digit identifier of a query (same formula as the C1 sampler)."""
    return hashlib.sha256((kb + "\0" + typ + "\0" + json.dumps(expr, separators=(",", ":"))).encode()).hexdigest()[:16]


def canonical(expr: Any) -> Any:
    """Sort conjunction operands so that syntactic permutations collapse."""
    if isinstance(expr, list) and expr[0] == "and":
        return ["and", *sorted(expr[1:], key=lambda x: json.dumps(x, sort_keys=True))]
    return expr


def templates(rng: random.Random, classes: list[str], roles: list[str]) -> dict[str, Callable[[], Any]]:
    """Random candidate generators for the seven query types."""
    c = lambda: rng.choice(classes)  # noqa: E731
    r = lambda: rng.choice(roles)  # noqa: E731

    def two() -> tuple[str, str]:
        a, b = rng.sample(classes, 2)
        return a, b

    return {
        "A": lambda: c(),
        "A_and_B": lambda: ["and", *two()],
        "some_r_A": lambda: ["some", r(), c()],
        "A_and_some_r_B": lambda: ["and", c(), ["some", r(), c()]],
        "some_r_and_B": lambda: ["some", r(), ["and", *two()]],
        "some_r_some_s_A": lambda: ["some", r(), ["some", r(), c()]],
        "some_inv_r_A": lambda: ["some", ["inv", r()], c()],
    }


def sample_from(mat, kb: str, seed: int = 0, per_type: int = 100, max_draws: int | None = None) -> dict:
    """Sample queries whose filter answer sets are non-empty, non-universal and new per type.

    Parameters
    ----------
    mat : materialise.Materialisation
        One-pass materialisation of the KB.
    kb : str
        KB name (enters the query id).
    seed : int
        Seed of the candidate generator.
    per_type : int
        Maximum number of queries kept per type.
    max_draws : int or None
        Maximum candidates drawn per type; default ``max(100 * per_type, 1000)``.

    Returns
    -------
    dict
        ``queries.json`` content.
    """
    max_draws = max_draws or max(per_type * 100, 1000)
    classes = [c for c in mat.classes if mat.members[c].any()]
    roles = [r for r in mat.roles if r in mat.edges and mat.edges[r].nnz]
    rng = random.Random(seed)
    gen = templates(rng, classes, roles)
    out, stats = [], {}
    for typ in TYPES:
        needs_roles = typ not in ("A", "A_and_B")
        if not classes or (needs_roles and not roles) or (typ in ("A_and_B", "some_r_and_B") and len(classes) < 2):
            stats[typ] = {"draws": 0, "kept": 0, "rejected_empty": 0, "rejected_universal": 0, "rejected_duplicate": 0, "repeated_syntax": 0}
            continue
        seen_sets: set[bytes] = set()
        seen_syntax: set[str] = set()
        counts = dict(draws=0, kept=0, rejected_empty=0, rejected_universal=0, rejected_duplicate=0, repeated_syntax=0)
        for _ in range(max_draws if typ != "A" else max(max_draws, len(classes) * 20)):
            expr = canonical(gen[typ]())
            counts["draws"] += 1
            syntax = json.dumps(expr, separators=(",", ":"), sort_keys=True)
            if syntax in seen_syntax:
                counts["repeated_syntax"] += 1
                continue
            seen_syntax.add(syntax)
            vec = mat.evaluate(expr)
            size = int(vec.sum())
            if size == 0:
                counts["rejected_empty"] += 1; continue
            if size == mat.n:
                counts["rejected_universal"] += 1; continue
            key = np.packbits(vec).tobytes()
            if key in seen_sets:
                counts["rejected_duplicate"] += 1; continue
            seen_sets.add(key)
            out.append({"query_id": query_id(kb, typ, expr), "type": typ, "expr": expr, "n_filter_answers": size,
                        "filter_answers": [mat.individuals[k] for k in np.flatnonzero(vec)]})
            counts["kept"] += 1
            if counts["kept"] >= per_type:
                break
        stats[typ] = counts
    return {"kb": kb, "seed": seed, "per_type": per_type, "max_draws_per_type": max_draws, "sampler": SAMPLER,
            "filter_note": "filter_answers come from materialise.py (ELK class realisation of the EL projection + closed asserted role "
                           "edges); they select queries but are not reference answers",
            "materialisation": mat.summary, "sampling_stats": stats, "queries": out}


def sample(kb: str, root: Path, seed: int = 0, per_type: int = 100, max_draws: int | None = None) -> dict:
    """Materialise ``kb`` once and sample its queries (see :func:`sample_from`)."""
    from materialise import materialise_kb
    started = time.monotonic()
    mat = materialise_kb(kb, root)
    result = sample_from(mat, kb, seed, per_type, max_draws)
    result["seconds_total"] = round(time.monotonic() - started, 2)
    return result


def main() -> None:
    """Run the query sampler CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", choices=NAMES, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-type", type=int, default=100)
    p.add_argument("--max-draws", type=int, default=None, help="candidates drawn per type (default 100*per-type)")
    p.add_argument("--heap-gb", type=int, default=None, help="JVM heap (GiB); capped by free memory and 20")
    a = p.parse_args()
    start_jvm(a.heap_gb)
    result = sample(a.kb, a.data_dir, a.seed, a.per_type, a.max_draws)
    m = result["materialisation"]
    print(f"{a.kb}: materialised {m['individuals']} individuals, {m['class_memberships']} memberships, "
          f"{m['asserted_distinct_edges']} asserted -> {m['closed_edges']} closed edges in {m['seconds']['total']} s")
    for typ, s in result["sampling_stats"].items():
        print(f"{a.kb}/{typ}: {s['kept']} accepted ({s['draws']} draws; rejected empty={s['rejected_empty']} "
              f"universal={s['rejected_universal']} duplicate={s['rejected_duplicate']} repeated={s['repeated_syntax']})")
    path = a.data_dir / "kbs" / a.kb / "queries.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {len(result['queries'])} queries to {path} ({result['seconds_total']} s)", file=sys.stdout)


if __name__ == "__main__":
    main()
