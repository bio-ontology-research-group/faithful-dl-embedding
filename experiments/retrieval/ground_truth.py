#!/usr/bin/env python3
"""Reasoner reference answers for sampled retrieval queries (EL projection and full KB).

Inputs
------
``<data-dir>/kbs/<kb>/queries.json`` written by ``sample_queries.py`` (materialisation-filter
sampler), ``<kb>.owl`` and ``<kb>_el.owl``.

Outputs
-------
``<data-dir>/kbs/<kb>/ground_truth.json``: per query the EL-projection reference
(``el_answers``), the full-KB reference (``full_answers``, ``null`` when not obtained), timings,
timeout flags, the plain asserted-ABox lookup, and the comparison of the sampling filter's
answer set with each reference; plus a summary block (also printed).

Reasoners
---------
* EL reference, non-inverse types: ELK 0.4.3 on the EL projection.
* EL reference, ``some_inv_r_A``: HermiT on the EL projection (ELK has no inverse roles).
* Full-KB reference, all types: HermiT on the full KB.
* Check (``--no-elk-check`` disables): HermiT also answers the non-inverse queries on the EL
  projection (``el_hermit_answers``, within the EL budget). ELK 0.4.3 ignores
  ``ObjectPropertyRange`` (and ``DifferentIndividuals``) axioms that the projection keeps, so
  ``elk_vs_hermit`` records where the ELK reference is incomplete. ``el_answers`` stays ELK's.

Every reasoner call runs under a watchdog thread that calls ``reasoner.interrupt()`` (repeatedly,
every 0.25 s, until the call returns) when the per-query timeout (``--timeout``) or the remaining
budget is reached. HermiT work on the full KB (setup + queries) has a global budget
(``--budget-minutes``, default 90); once it is spent the remaining full-KB answers are ``null``
with ``full_timeout = true`` and ``full_skipped = "budget"``. HermiT work for inverse queries on
the projection has its own budget (``--el-budget-minutes``). Full-KB queries are processed
round-robin over query types so that a partially spent budget covers every type.

HermiT query method (``--hermit-mode``): by default each class expression ``CE`` is given a
fresh name by adding ``CE ⊑ Q`` (one fresh ``Q`` per query, ``--chunk-size`` queries per
reasoner), and the guarded call is ``getInstances(Q)``; because ``Q`` is fresh this returns exactly
the instances of ``CE``, while letting HermiT use its ABox instance manager instead of one
tableau test per individual. ``--hermit-mode per-query`` calls ``getInstances(CE)`` directly.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from common import DATA_ROOT, NAMES, build_index, eval_structural, iri, java_expr_owl, load_ontology, start_jvm

INVERSE_TYPE = "some_inv_r_A"
SAMPLER = "materialisation-filter"


class ReasonerTimeout(Exception):
    """Raised when the watchdog interrupted a reasoner call."""


def call_with_watchdog(reasoner: Any, fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``fn()`` while a timer thread calls ``reasoner.interrupt()`` after ``timeout_s``.

    Parameters
    ----------
    reasoner : OWLReasoner
        Reasoner whose ``interrupt()`` aborts the running call.
    fn : callable
        The reasoner call (JPype releases the GIL while Java runs, so the timer can fire).
    timeout_s : float
        Seconds before the interrupt; non-positive values raise immediately.

    Returns
    -------
    Any
        ``fn()``'s result if it returned before the interrupt took effect.

    Raises
    ------
    ReasonerTimeout
        If the watchdog fired and ``fn`` raised (normally ``ReasonerInterruptedException``).
    """
    if timeout_s <= 0:
        raise ReasonerTimeout("no time left")
    fired, done = threading.Event(), threading.Event()

    def fire() -> None:
        # HermiT resets its interrupt flag whenever an internal task starts, so one call can be
        # lost between tasks; keep interrupting until the guarded call has returned.
        fired.set()
        while not done.is_set():
            reasoner.interrupt()
            done.wait(0.25)

    timer = threading.Timer(timeout_s, fire)
    timer.daemon = True
    timer.start()
    try:
        return fn()
    except Exception as exc:
        if fired.is_set():
            raise ReasonerTimeout(f"interrupted after {timeout_s:.1f} s") from exc
        raise
    finally:
        done.set()
        timer.cancel()


def _instances(reasoner: Any, ce: Any, timeout_s: float) -> tuple[list[str], float]:
    """Return sorted instance IRIs of ``ce`` and the elapsed seconds, under a watchdog."""
    started = time.monotonic()
    node_set = call_with_watchdog(reasoner, lambda: reasoner.getInstances(ce, False), timeout_s)
    elapsed = time.monotonic() - started
    return sorted({iri(ind) for node in node_set.getNodes() for ind in node.getEntities()}), elapsed


def round_robin(queries: list[dict]) -> list[dict]:
    """Interleave queries by type (A, A_and_B, ... , A, A_and_B, ...) preserving within-type order."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_type[q["type"]].append(q)
    order, k = [], 0
    while any(k < len(v) for v in by_type.values()):
        order.extend(v[k] for v in by_type.values() if k < len(v))
        k += 1
    return order


def compare(filter_answers: list[str] | None, reference: list[str] | None) -> dict | None:
    """Counts of reference answers missing from the filter and filter answers not in the reference."""
    if filter_answers is None or reference is None:
        return None
    f, r = set(filter_answers), set(reference)
    return {"missing_from_filter": len(r - f), "extra_in_filter": len(f - r), "equal": f == r}


def load_queries(directory: Path) -> tuple[dict, str]:
    """Read ``queries.json``; refuse files that were not written by the materialisation sampler."""
    path = directory / "queries.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run sample_queries.py first")
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("sampler") != SAMPLER:
        raise SystemExit(f"{path} was not written by the materialisation-filter sampler; re-run sample_queries.py")
    return data, hashlib.sha256(raw).hexdigest()


class HermitSession:
    """A HermiT reasoner created lazily, with setup and queries charged to one time budget."""

    def __init__(self, ontology: Any, budget_s: float, setup_timeout_s: float, label: str):
        self.ontology, self.label = ontology, label
        self.deadline = time.monotonic() + budget_s
        self.setup_timeout_s = setup_timeout_s
        self.reasoner = None
        self.status = "not_started"
        self.error: str | None = None
        self.setup_seconds: float | None = None

    def remaining(self) -> float:
        """Seconds left in the budget."""
        return self.deadline - time.monotonic()

    def ready(self) -> bool:
        """Create the reasoner and check consistency once; return whether queries may run."""
        if self.status == "not_started":
            from org.semanticweb.HermiT import ReasonerFactory
            started = time.monotonic()
            try:
                self.reasoner = ReasonerFactory().createReasoner(self.ontology)
                consistent = call_with_watchdog(self.reasoner, self.reasoner.isConsistent,
                                                min(self.setup_timeout_s, self.remaining()))
                self.status = "ok" if consistent else "inconsistent"
            except ReasonerTimeout as exc:
                self.status, self.error = "setup_timeout", str(exc)
            except Exception as exc:  # noqa: BLE001 - recorded in the output
                self.status, self.error = "unavailable", f"{type(exc).__name__}: {exc}"
            self.setup_seconds = round(time.monotonic() - started, 2)
            print(f"[{self.label}] HermiT setup: {self.status} in {self.setup_seconds} s", flush=True)
        return self.status == "ok"

    def dispose(self) -> None:
        """Release the reasoner."""
        if self.reasoner is not None:
            self.reasoner.dispose()
            self.reasoner = None


QUERY_CLASS_NS = "urn:gfe-retrieval:query-class#"


def _store(rec: dict, prefix: str, outcome: dict) -> None:
    """Copy a :func:`hermit_references` outcome into a record under ``el_*`` or ``full_*`` keys."""
    rec[f"{prefix}_answers"] = outcome.get("answers")
    rec[f"{prefix}_time_seconds"] = outcome.get("seconds")
    rec[f"{prefix}_timeout"] = bool(outcome.get("timeout"))
    rec[f"{prefix}_method"] = outcome.get("method")
    if outcome.get("skipped"):
        rec[f"{prefix}_skipped"] = outcome["skipped"]
    if outcome.get("error"):
        rec[f"{prefix}_error"] = outcome["error"]


def hermit_references(session: "HermitSession", queries: list[dict], df: Any, timeout: float,
                      mode: str = "fresh-class", chunk_size: int = 100) -> dict[str, dict]:
    """HermiT instances of each query's class expression, under watchdog and budget.

    Parameters
    ----------
    session : HermitSession
        Ready session (consistency checked); its deadline is the budget.
    queries : list of dict
        Queries in processing order.
    df : OWLDataFactory
        Data factory for building class expressions.
    timeout : float
        Per-query timeout (seconds) for each guarded ``getInstances`` call.
    mode : {"fresh-class", "per-query"}
        ``per-query``: ``getInstances(CE)`` on the session's reasoner (HermiT then tests every
        individual separately, which takes minutes per query on the large KBs).
        ``fresh-class``: per chunk of ``chunk_size`` queries, add ``CE_i ⊑ Q_i`` with fresh named
        classes ``Q_i`` to the ontology, create one HermiT reasoner, and call
        ``getInstances(Q_i)`` for each query under its own watchdog; the axioms are removed
        afterwards. Since ``Q_i`` is fresh, ``K ∪ {CE_i ⊑ Q_i} ⊨ Q_i(a)`` iff ``K ⊨ CE_i(a)``,
        so the answers are exactly the instances of ``CE_i``.
    chunk_size : int
        Queries per reasoner in ``fresh-class`` mode.

    Returns
    -------
    dict
        Query id to ``{"answers", "seconds", "timeout", "skipped", "error", "method"}``.
    """
    from java.util import HashSet
    from org.semanticweb.HermiT import ReasonerFactory
    from org.semanticweb.owlapi.model import IRI
    outcomes: dict[str, dict] = {}

    def guarded(reasoner: Any, ce: Any, method: str) -> dict:
        left = session.remaining()
        if left <= 0:
            return {"timeout": True, "skipped": "budget", "method": method}
        try:
            answers, seconds = _instances(reasoner, ce, min(timeout, left))
            return {"answers": answers, "seconds": round(seconds, 3), "method": method}
        except ReasonerTimeout as exc:
            return {"timeout": True, "error": str(exc), "method": method}
        except Exception as exc:  # noqa: BLE001 - recorded per query
            return {"error": f"{type(exc).__name__}: {exc}", "method": method}

    if mode == "per-query":
        for n, q in enumerate(queries, 1):
            outcomes[q["query_id"]] = guarded(session.reasoner, java_expr_owl(q["expr"], df), "getInstances(CE)")
            if n % 25 == 0:
                print(f"[{session.label}] {n}/{len(queries)} queries, {session.remaining() / 60:.1f} min budget left", flush=True)
        return outcomes
    if mode != "fresh-class":
        raise ValueError(f"unknown HermiT mode {mode!r}")
    session.dispose()  # chunks use their own reasoners
    ontology = session.ontology
    manager = ontology.getOWLOntologyManager()
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start:start + chunk_size]
        if session.remaining() <= 0:
            for q in chunk:
                outcomes[q["query_id"]] = {"timeout": True, "skipped": "budget", "method": None}
            continue
        method = f"getInstances(Q) with fresh Q, CE ⊑ Q; chunk of {len(chunk)}"
        names = {q["query_id"]: df.getOWLClass(IRI.create(QUERY_CLASS_NS + q["query_id"])) for q in chunk}
        axioms = HashSet()
        for q in chunk:
            axioms.add(df.getOWLSubClassOfAxiom(java_expr_owl(q["expr"], df), names[q["query_id"]]))
        manager.addAxioms(ontology, axioms)
        reasoner = None
        try:
            reasoner = ReasonerFactory().createReasoner(ontology)
            for q in chunk:
                outcomes[q["query_id"]] = guarded(reasoner, names[q["query_id"]], method)
        finally:
            if reasoner is not None:
                reasoner.dispose()
            manager.removeAxioms(ontology, axioms)
        got = sum(outcomes[q["query_id"]].get("answers") is not None for q in chunk)
        print(f"[{session.label}] chunk {start // chunk_size + 1}: {got}/{len(chunk)} answered, "
              f"{session.remaining() / 60:.1f} min budget left", flush=True)
    return outcomes


def run(kb: str, root: Path, timeout: float = 60.0, budget_minutes: float = 90.0, el_budget_minutes: float = 30.0,
        setup_timeout: float = 600.0, hermit_mode: str = "fresh-class", chunk_size: int = 100,
        elk_check: bool = True) -> dict:
    """Compute EL-projection and full-KB references for every query of ``kb``.

    Parameters
    ----------
    kb : str
        KB name.
    root : pathlib.Path
        Data directory (contains ``kbs/<kb>``).
    timeout : float
        Per-query timeout in seconds (watchdog).
    budget_minutes : float
        Global budget for HermiT on the full KB (setup + queries).
    el_budget_minutes : float
        Budget for HermiT on the EL projection (inverse queries only).
    setup_timeout : float
        Timeout in seconds for HermiT's initial consistency check (charged to the budget).
    hermit_mode : {"fresh-class", "per-query"}
        How HermiT answers a query (see :func:`hermit_references`).
    chunk_size : int
        Queries per HermiT reasoner in ``fresh-class`` mode.
    elk_check : bool
        Also answer non-inverse queries with HermiT on the projection (within the EL budget) and
        record ELK/HermiT differences (``el_hermit_answers``, ``elk_vs_hermit``).

    Returns
    -------
    dict
        ``ground_truth.json`` content.
    """
    directory = root / "kbs" / kb
    data, queries_sha = load_queries(directory)
    queries = data["queries"]
    started = time.monotonic()
    _, full = load_ontology(directory / f"{kb}.owl")
    _, projection = load_ontology(directory / f"{kb}_el.owl")
    index = build_index(full, False)
    df = full.getOWLOntologyManager().getOWLDataFactory()
    records = {q["query_id"]: {
        "query_id": q["query_id"], "type": q["type"], "expr": q["expr"],
        "abox_lookup": sorted(eval_structural(q["expr"], index)), "filter_answers": q.get("filter_answers"),
        "el_answers": None, "el_reasoner": "HermiT" if q["type"] == INVERSE_TYPE else "ELK", "el_time_seconds": None,
        "el_timeout": False, "full_answers": None, "full_time_seconds": None, "full_timeout": False,
        "full_skipped": None, "needs_tbox": None} for q in queries}

    # 1. EL projection, non-inverse types: ELK.
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    elk = ElkReasonerFactory().createReasoner(projection)
    el_status = "ok" if elk.isConsistent() else "inconsistent"
    if el_status == "ok":
        for q in queries:
            if q["type"] == INVERSE_TYPE:
                continue
            rec = records[q["query_id"]]
            try:
                rec["el_answers"], t = _instances(elk, java_expr_owl(q["expr"], df), timeout)
                rec["el_time_seconds"] = round(t, 3)
            except ReasonerTimeout as exc:
                rec["el_timeout"], rec["el_error"] = True, str(exc)
            except Exception as exc:  # noqa: BLE001
                rec["el_error"] = f"{type(exc).__name__}: {exc}"
    elk.dispose()
    print(f"[{kb}] ELK on projection: {el_status}, {time.monotonic() - started:.1f} s", flush=True)

    # 2. EL projection, inverse type: HermiT with its own budget. With ``elk_check`` the same
    #    session then answers the non-inverse queries too, to measure ELK's incompleteness
    #    (ELK 0.4.3 ignores ObjectPropertyRange axioms that the projection keeps).
    inverse = [q for q in queries if q["type"] == INVERSE_TYPE]
    others = round_robin([q for q in queries if q["type"] != INVERSE_TYPE]) if elk_check else []
    el_hermit = HermitSession(projection, el_budget_minutes * 60, setup_timeout, f"{kb}/EL")
    if (inverse or others) and el_status == "ok" and el_hermit.ready():
        outcomes = hermit_references(el_hermit, inverse + others, df, timeout, hermit_mode, chunk_size)
        for q in inverse:
            _store(records[q["query_id"]], "el", outcomes[q["query_id"]])
        for q in others:
            rec, out = records[q["query_id"]], outcomes[q["query_id"]]
            rec["el_hermit_answers"], rec["el_hermit_timeout"] = out.get("answers"), bool(out.get("timeout"))
            if rec["el_answers"] is not None and out.get("answers") is not None:
                elk, hermit = set(rec["el_answers"]), set(out["answers"])
                rec["elk_vs_hermit"] = {"missing_in_elk": len(hermit - elk), "extra_in_elk": len(elk - hermit), "equal": elk == hermit}
    el_hermit.dispose()
    print(f"[{kb}] HermiT on projection (inverse queries{' + ELK check' if elk_check else ''}): "
          f"{time.monotonic() - started:.1f} s", flush=True)

    # 3. Full KB: HermiT, global budget, round-robin over types.
    full_hermit = HermitSession(full, budget_minutes * 60, setup_timeout, f"{kb}/full")
    if full_hermit.ready():
        outcomes = hermit_references(full_hermit, round_robin(queries), df, timeout, hermit_mode, chunk_size)
        for q in queries:
            _store(records[q["query_id"]], "full", outcomes[q["query_id"]])
    else:
        for q in queries:
            rec = records[q["query_id"]]
            rec["full_timeout"] = full_hermit.status == "setup_timeout"
            rec["full_skipped"] = full_hermit.status
    full_hermit.dispose()
    print(f"[{kb}] HermiT on full KB: {time.monotonic() - started:.1f} s", flush=True)

    # 4. Derived fields and comparisons.
    individuals = {iri(i) for i in projection.getIndividualsInSignature()}
    seen_el: dict[tuple[str, tuple[str, ...]], str] = {}
    for q in queries:
        rec = records[q["query_id"]]
        reference = rec["full_answers"] if rec["full_answers"] is not None else rec["el_answers"]
        if reference is not None:
            rec["needs_tbox"] = bool(set(reference) - set(rec["abox_lookup"]))
        rec["filter_vs_el"] = compare(rec["filter_answers"], rec["el_answers"])
        rec["filter_vs_full"] = compare(rec["filter_answers"], rec["full_answers"])
        if rec["el_answers"] is not None:
            key = (rec["type"], tuple(rec["el_answers"]))
            rec["el_empty"] = not rec["el_answers"]
            rec["el_universal"] = individuals <= set(rec["el_answers"])
            first = seen_el.setdefault(key, rec["query_id"])
            rec["el_duplicate_of"] = first if first != rec["query_id"] else None
    result = {"kb": kb, "queries_sha256": queries_sha, "sampler": data.get("sampler"),
              "reasoners": {"el_non_inverse": "ELK 0.4.3 on the EL projection", "el_inverse": "HermiT 1.3.8.413 on the EL projection",
                            "full": "HermiT 1.3.8.413 on the full KB"},
              "timeout_seconds_per_query": timeout, "budget_minutes_full": budget_minutes, "budget_minutes_el_inverse": el_budget_minutes,
              "setup_timeout_seconds": setup_timeout, "hermit_mode": hermit_mode, "chunk_size": chunk_size, "elk_check": elk_check,
              "el_status": el_status, "el_hermit_status": el_hermit.status, "el_hermit_setup_seconds": el_hermit.setup_seconds,
              "full_status": full_hermit.status, "full_error": full_hermit.error, "full_setup_seconds": full_hermit.setup_seconds,
              "seconds_total": None, "summary": None, "queries": [records[q["query_id"]] for q in queries]}
    result["seconds_total"] = round(time.monotonic() - started, 1)
    result["summary"] = summarise(result)
    return result


def summarise(gt: dict) -> dict:
    """Per-type counts: references obtained, timeouts and filter/reasoner disagreements."""
    rows: dict[str, Counter] = defaultdict(Counter)
    for q in gt["queries"]:
        c = rows[q["type"]]
        c["queries"] += 1
        c["el_reference"] += q["el_answers"] is not None
        c["el_timeout"] += bool(q["el_timeout"])
        c["full_reference"] += q["full_answers"] is not None
        c["full_timeout"] += bool(q["full_timeout"])
        c["full_skipped_budget"] += q.get("full_skipped") == "budget"
        for target in ("el", "full"):
            cmp_ = q.get(f"filter_vs_{target}")
            if cmp_ is not None:
                c[f"filter_vs_{target}_compared"] += 1
                c[f"filter_vs_{target}_disagree"] += not cmp_["equal"]
                c[f"filter_vs_{target}_missing_queries"] += cmp_["missing_from_filter"] > 0
                c[f"filter_vs_{target}_extra_queries"] += cmp_["extra_in_filter"] > 0
                c[f"filter_vs_{target}_missing_answers"] += cmp_["missing_from_filter"]
                c[f"filter_vs_{target}_extra_answers"] += cmp_["extra_in_filter"]
        if q.get("elk_vs_hermit") is not None:
            c["elk_vs_hermit_compared"] += 1
            c["elk_vs_hermit_disagree"] += not q["elk_vs_hermit"]["equal"]
            c["elk_missing_answers"] += q["elk_vs_hermit"]["missing_in_elk"]
            c["elk_extra_answers"] += q["elk_vs_hermit"]["extra_in_elk"]
        c["el_empty"] += bool(q.get("el_empty"))
        c["el_universal"] += bool(q.get("el_universal"))
        c["el_duplicate"] += q.get("el_duplicate_of") is not None
    total = Counter()
    for c in rows.values():
        total.update(c)
    return {"by_type": {t: dict(c) for t, c in sorted(rows.items())}, "total": dict(total)}


def main() -> None:
    """Run the ground-truth CLI for one KB."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", choices=NAMES, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    p.add_argument("--timeout", type=float, default=60.0, help="per-query timeout in seconds")
    p.add_argument("--budget-minutes", type=float, default=90.0, help="HermiT budget on the full KB")
    p.add_argument("--el-budget-minutes", type=float, default=30.0, help="HermiT budget for inverse queries on the projection")
    p.add_argument("--setup-timeout", type=float, default=600.0, help="timeout (s) of HermiT's consistency check")
    p.add_argument("--hermit-mode", choices=("fresh-class", "per-query"), default="fresh-class",
                   help="HermiT query method (see hermit_references)")
    p.add_argument("--chunk-size", type=int, default=100, help="queries per HermiT reasoner in fresh-class mode")
    p.add_argument("--no-elk-check", action="store_true", help="skip HermiT on the projection for non-inverse queries")
    p.add_argument("--heap-gb", type=int, default=None, help="JVM heap (GiB); capped by free memory and 20")
    a = p.parse_args()
    directory = a.data_dir / "kbs" / a.kb
    load_queries(directory)  # fail fast, before starting the JVM
    with (directory / ".ground_truth.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another ground_truth.py run holds {directory / '.ground_truth.lock'}")
        start_jvm(a.heap_gb)
        res = run(a.kb, a.data_dir, a.timeout, a.budget_minutes, a.el_budget_minutes, a.setup_timeout, a.hermit_mode, a.chunk_size,
                  not a.no_elk_check)
        out = directory / "ground_truth.json"
        out.write_text(json.dumps(res, indent=1) + "\n")
    tot = res["summary"]["total"]
    print(f"full={res['full_status']} projection={res['el_status']} queries={tot['queries']} "
          f"el_refs={tot['el_reference']} full_refs={tot['full_reference']} full_timeouts={tot['full_timeout']} "
          f"(budget-skipped {tot['full_skipped_budget']}) seconds={res['seconds_total']} -> {out}")
    for typ, c in res["summary"]["by_type"].items():
        print(f"  {typ}: n={c['queries']} el={c['el_reference']} full={c['full_reference']} "
              f"filter!=el {c.get('filter_vs_el_disagree', 0)}/{c.get('filter_vs_el_compared', 0)} "
              f"(missing {c.get('filter_vs_el_missing_queries', 0)}, extra {c.get('filter_vs_el_extra_queries', 0)}) "
              f"filter!=full {c.get('filter_vs_full_disagree', 0)}/{c.get('filter_vs_full_compared', 0)} "
              f"ELK!=HermiT(EL) {c.get('elk_vs_hermit_disagree', 0)}/{c.get('elk_vs_hermit_compared', 0)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
