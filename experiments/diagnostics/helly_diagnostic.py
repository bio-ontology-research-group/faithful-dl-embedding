#!/usr/bin/env python3
"""Helly-satisfiability diagnostic on EL++ ontologies using mowl.geometric.

Runs:
1. EL++ normalisation (`mowl.geometric.normalise`) and axiom/signature counting.
2. Canonical model construction (`mowl.geometric.CanonicalModel.from_normalised`).
3. Bottom-relevant satisfiable concept name counting (`_bottom_relevant_names`).
4. Full Helly check (`mowl.geometric.helly_check`) in a spawned child process
   with a hard wall-clock timeout.
5. Targeted triangle search (`mowl.geometric.certificate_triangles`) plus
   direct NF2 named-conjunction graph and triangle counts.
6. Cross-check between `helly_check` singleton non-degenerate certificates and
   `certificate_triangles`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


def _local_name(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def _lookup_labels(ontology: Any, iris: Iterable[str]) -> Dict[str, Optional[str]]:
    """Fetch rdfs:label annotations via OWLAPI for the given IRIs."""
    from org.semanticweb.owlapi.model import IRI, OWLLiteral

    out: Dict[str, Optional[str]] = {}
    closure = list(ontology.getImportsClosure())
    for iri_str in sorted(set(iris)):
        if not iri_str:
            continue
        best_label: Optional[str] = None
        best_priority = -1
        iri_obj = IRI.create(iri_str)
        for ont in closure:
            for ax in ont.getAnnotationAssertionAxioms(iri_obj):
                if ax.getProperty().isLabel():
                    val = ax.getValue()
                    if isinstance(val, OWLLiteral):
                        lit = str(val.getLiteral())
                        lang = str(val.getLang()).lower() if val.hasLang() else ""
                        if lang == "en":
                            priority = 3
                        elif lang == "":
                            priority = 2
                        else:
                            priority = 1
                        if priority > best_priority:
                            best_priority = priority
                            best_label = lit
        out[iri_str] = best_label
    return out


def _compute_nf2_graph_stats(cm: Any) -> Tuple[int, int]:
    """Count named binary conjunctions (direct NF2 equivalences) and graph triangles.

    Follows the exact edge construction of `mowl.geometric.certificate_triangles`:
    an undirected edge {A, B} exists iff there is an NF2(A, B, N) axiom in
    `cm.axioms` with A != B, N != BOTTOM, A, B, N != TOP, all three satisfiable
    in `cm`, and `cm.entails(N, A) and cm.entails(N, B)`.
    """
    from mowl.geometric.normal_forms import BOTTOM, NF2, TOP

    edge_map: Dict[FrozenSet[str], str] = {}
    for ax in cm.axioms:
        if not isinstance(ax, NF2):
            continue
        a, b, n = ax.sub1, ax.sub2, ax.sup
        if a == b or n == BOTTOM or TOP in (a, b, n):
            continue
        if (
            cm.is_satisfiable(a)
            and cm.is_satisfiable(b)
            and cm.is_satisfiable(n)
            and cm.entails(n, a)
            and cm.entails(n, b)
        ):
            key: FrozenSet[str] = frozenset((a, b))
            if key not in edge_map or n < edge_map[key]:
                edge_map[key] = n

    adj: Dict[str, Set[str]] = defaultdict(set)
    for pair in edge_map:
        u, v = tuple(pair)
        adj[u].add(v)
        adj[v].add(u)

    graph_triangles = 0
    for a in sorted(adj.keys()):
        higher = sorted(v for v in adj[a] if v > a)
        m = len(higher)
        for i in range(m):
            b = higher[i]
            adj_b = adj[b]
            for j in range(i + 1, m):
                c = higher[j]
                if c in adj_b:
                    graph_triangles += 1

    return len(edge_map), graph_triangles


def _helly_child_worker(
    in_pkl_path: str,
    out_pkl_path: str,
    k_max: int,
    max_certificates: int,
    jvm_memory: str,
) -> None:
    """Child worker for `helly_check` with isolated JVM startup."""
    import mowl

    mowl.init_jvm(jvm_memory)
    from mowl.geometric.helly import helly_check

    with open(in_pkl_path, "rb") as f:
        cm = pickle.load(f)

    t0 = time.perf_counter()
    report = helly_check(cm, k_max=k_max, max_certificates=max_certificates)
    elapsed = time.perf_counter() - t0

    size_dist = Counter(
        len(cert.c1) + len(cert.c2) + len(cert.c3)
        for cert in report.certificates_strong
    )
    singleton_nondeg = [
        (cert.c1, cert.c2, cert.c3, cert.pairwise_names)
        for cert in report.certificates_nondegenerate
        if len(cert.c1) == 1 and len(cert.c2) == 1 and len(cert.c3) == 1
    ]
    non_singleton_nondeg = [
        (cert.c1, cert.c2, cert.c3, cert.pairwise_names)
        for cert in report.certificates_nondegenerate
        if not (len(cert.c1) == 1 and len(cert.c2) == 1 and len(cert.c3) == 1)
    ]

    payload = {
        "wall_seconds": elapsed,
        "complete": report.complete,
        "has_roles": report.has_roles,
        "k_max": report.k_max,
        "box_representable_strong": report.box_representable_strong,
        "box_representable_nondegenerate": report.box_representable_nondegenerate,
        "strong_certificates_count": len(report.certificates_strong),
        "nondegenerate_certificates_count": len(report.certificates_nondegenerate),
        "minimal_unsatisfiable_set_size_distribution": {
            str(k): int(v) for k, v in sorted(size_dist.items())
        },
        "singleton_nondegenerate_certificates": singleton_nondeg,
        "non_singleton_nondegenerate_certificates": non_singleton_nondeg,
    }
    tmp_path = out_pkl_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_pkl_path)


def _format_certificate(
    cert: Any, labels: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    c1 = cert.c1[0] if len(cert.c1) == 1 else list(cert.c1)
    c2 = cert.c2[0] if len(cert.c2) == 1 else list(cert.c2)
    c3 = cert.c3[0] if len(cert.c3) == 1 else list(cert.c3)
    p12, p13, p23 = cert.pairwise_names

    def _lbl(iri: Optional[str]) -> Optional[str]:
        if iri is None:
            return None
        return labels.get(iri) or _local_name(iri)

    return {
        "c1": list(cert.c1),
        "c2": list(cert.c2),
        "c3": list(cert.c3),
        "pairwise_names": {
            "p12": p12,
            "p13": p13,
            "p23": p23,
        },
        "labels": {
            "c1": _lbl(cert.c1[0]) if len(cert.c1) == 1 else [_lbl(x) for x in cert.c1],
            "c2": _lbl(cert.c2[0]) if len(cert.c2) == 1 else [_lbl(x) for x in cert.c2],
            "c3": _lbl(cert.c3[0]) if len(cert.c3) == 1 else [_lbl(x) for x in cert.c3],
            "p12": _lbl(p12),
            "p13": _lbl(p13),
            "p23": _lbl(p23),
        },
        "rdfs_labels": {
            "c1": labels.get(cert.c1[0]) if len(cert.c1) == 1 else [labels.get(x) for x in cert.c1],
            "c2": labels.get(cert.c2[0]) if len(cert.c2) == 1 else [labels.get(x) for x in cert.c2],
            "c3": labels.get(cert.c3[0]) if len(cert.c3) == 1 else [labels.get(x) for x in cert.c3],
            "p12": labels.get(p12) if p12 else None,
            "p13": labels.get(p13) if p13 else None,
            "p23": labels.get(p23) if p23 else None,
        },
    }


SUMMARY_HEADER = (
    "| Ontology | Concepts | Roles | NF1 | NF2 | NF3 | NF4 | NF6 | NF7 | "
    "⊥-axioms | Dropped | CM names | Unsat | CM time (s) | ⊥-rel sat | "
    "Helly status | Strong certs | Nondeg certs | Helly time (s) | "
    "Named conj | Graph △ | Cert △ | △ time (s) | Discrepancies |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
)


def _format_summary_row(record: Dict[str, Any]) -> str:
    norm = record["normalise"]
    nf = norm["axioms_by_normal_form"]
    cm = record["canonical_model"]
    hc = record["helly_check"]
    tri = record["targeted_triangle_search"]

    if hc.get("status") == "running":
        h_status = f"RUNNING (timeout {hc['timeout_seconds']}s)"
        strong_s = "…"
        nondeg_s = "…"
        hc_wall_s = "…"
    elif hc["timed_out"]:
        h_status = f"TIMEOUT ({hc['timeout_seconds']}s)"
        strong_s = "—"
        nondeg_s = "—"
        hc_wall_s = f"{hc['wall_seconds']:.2f}"
    else:
        h_status = "complete" if hc["complete"] else ("incomplete(roles)" if hc["has_roles"] else "incomplete(cutoff)")
        strong_s = str(hc["strong_certificates_count"])
        nondeg_s = str(hc["nondegenerate_certificates_count"])
        hc_wall_s = f"{hc['wall_seconds']:.2f}"

    return (
        f"| {record['name']} | {norm['concept_names_count']} | {norm['role_names_count']} | "
        f"{nf['NF1']} | {nf['NF2']} | {nf['NF3']} | {nf['NF4']} | {nf['NF6']} | {nf['NF7']} | "
        f"{norm['bottom_axioms_count']} | {norm['dropped_axioms']} | "
        f"{cm['names_count']} | {cm['unsatisfiable_names_count']} | {cm['wall_seconds']:.2f} | "
        f"{record['bottom_relevant_satisfiable_names_count']} | "
        f"{h_status} | {strong_s} | {nondeg_s} | {hc_wall_s} | "
        f"{tri['named_conjunctions_count']} | {tri['graph_triangles_count']} | "
        f"{tri['certificate_triangles_count']} | {tri['wall_seconds']:.2f} | "
        f"{len(record['discrepancies'])} |\n"
    )


def _update_summary_md(out_dir: Path, work_dir: Path, record: Dict[str, Any]) -> None:
    summary_path = out_dir / "summary.md"
    lock_path = work_dir / ".summary.lock"
    row = _format_summary_row(record)
    prefix = f"| {record['name']} |"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if not summary_path.exists():
                summary_path.write_text(SUMMARY_HEADER + row, encoding="utf-8")
                return
            lines = summary_path.read_text(encoding="utf-8").splitlines(keepends=True)
            replaced = False
            new_lines: List[str] = []
            for line in lines:
                if line.startswith(prefix):
                    new_lines.append(row)
                    replaced = True
                else:
                    new_lines.append(line)
            if not replaced:
                new_lines.append(row)
            summary_path.write_text("".join(new_lines), encoding="utf-8")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Helly-satisfiability diagnostic and targeted triangle search on an OWL ontology."
    )
    parser.add_argument("--ontology", required=True, help="Path to OWL ontology file.")
    parser.add_argument("--name", required=True, help="Short ontology identifier (e.g. pizza, go).")
    parser.add_argument("--out", required=True, help="Output directory for JSON and summary.md.")
    parser.add_argument("--k-max", type=int, default=4, help="Maximum minimal unsatisfiable set size for helly_check.")
    parser.add_argument("--max-certificates", type=int, default=100000, help="Maximum strong certificates for helly_check.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Hard wall-clock timeout in seconds for helly_check.")
    parser.add_argument("--jvm-memory", default="24g", help="JVM max heap size (e.g. 4g, 24g).")
    args = parser.parse_args()

    ont_path = Path(args.ontology).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir.parent / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name}.json"

    print(f"[{args.name}] Starting JVM ({args.jvm_memory})...", flush=True)
    import mowl

    mowl.init_jvm(args.jvm_memory)

    from mowl.datasets import PathDataset
    from mowl.geometric import (
        BOTTOM,
        CanonicalModel,
        InconsistentOntology,
        NF1,
        NF2,
        NF3,
        NF4,
        NF6,
        NF7,
        TOP,
        certificate_triangles,
        normalise,
    )
    from mowl.geometric.helly import _bottom_relevant_names

    print(f"[{args.name}] Loading PathDataset from {ont_path}...", flush=True)
    t_load0 = time.perf_counter()
    ds = PathDataset(str(ont_path))
    _ = ds.ontology
    load_wall = time.perf_counter() - t_load0
    print(f"[{args.name}] Loaded dataset in {load_wall:.2f}s. Running normalise(ds)...", flush=True)

    # Step 1: Normalise
    t_norm0 = time.perf_counter()
    no = normalise(ds)
    norm_wall = time.perf_counter() - t_norm0

    nf_counts = {
        "NF1": sum(1 for ax in no.axioms if isinstance(ax, NF1)),
        "NF2": sum(1 for ax in no.axioms if isinstance(ax, NF2)),
        "NF3": sum(1 for ax in no.axioms if isinstance(ax, NF3)),
        "NF4": sum(1 for ax in no.axioms if isinstance(ax, NF4)),
        "NF5": sum(1 for ax in no.axioms if isinstance(ax, NF1) and ax.sup == BOTTOM),
        "NF6": sum(1 for ax in no.axioms if isinstance(ax, NF6)),
        "NF7": sum(1 for ax in no.axioms if isinstance(ax, NF7)),
    }
    bot_by_nf = {
        "NF1": sum(1 for ax in no.axioms if isinstance(ax, NF1) and ax.sup == BOTTOM),
        "NF2": sum(1 for ax in no.axioms if isinstance(ax, NF2) and ax.sup == BOTTOM),
        "NF4": sum(1 for ax in no.axioms if isinstance(ax, NF4) and ax.sup == BOTTOM),
    }
    bottom_axioms_count = sum(bot_by_nf.values())

    normalise_info = {
        "wall_seconds": norm_wall,
        "concept_names_count": len(no.concept_names),
        "concept_names_excluding_top_bottom": len(no.concept_names - {TOP, BOTTOM}),
        "role_names_count": len(no.role_names),
        "total_axioms": len(no.axioms),
        "axioms_by_normal_form": nf_counts,
        "bottom_axioms_count": bottom_axioms_count,
        "bottom_axioms_by_nf": bot_by_nf,
        "dropped_axioms": no.dropped_axioms,
    }
    print(
        f"[{args.name}] Normalised in {norm_wall:.2f}s: "
        f"concepts={len(no.concept_names)}, roles={len(no.role_names)}, "
        f"axioms={len(no.axioms)}, bottom_axioms={bottom_axioms_count}, "
        f"dropped={no.dropped_axioms}",
        flush=True,
    )

    # Step 2: CanonicalModel construction
    # CanonicalModel.from_dataset(ds) is CanonicalModel.from_normalised(normalise(ds));
    # we build from `no` to avoid running ELNormalizer twice and report both completion
    # time and total from_dataset time (normalise + completion).
    print(f"[{args.name}] Building CanonicalModel...", flush=True)
    t_cm0 = time.perf_counter()
    inconsistent = False
    cm: Optional[CanonicalModel] = None
    try:
        cm = CanonicalModel.from_normalised(no)
    except InconsistentOntology:
        inconsistent = True
    cm_completion_wall = time.perf_counter() - t_cm0
    cm_total_wall = norm_wall + cm_completion_wall

    if inconsistent or cm is None:
        record = {
            "name": args.name,
            "ontology_path": str(ont_path),
            "status": "inconsistent",
            "load_wall_seconds": load_wall,
            "normalise": normalise_info,
            "canonical_model": {
                "wall_seconds": cm_total_wall,
                "completion_wall_seconds": cm_completion_wall,
                "inconsistent": True,
                "names_count": 0,
                "unsatisfiable_names_count": len(no.concept_names - {BOTTOM}),
            },
            "bottom_relevant_satisfiable_names_count": 0,
            "helly_check": None,
            "targeted_triangle_search": None,
            "cross_check": None,
            "discrepancies": [],
        }
        _write_json_atomic(json_path, record)
        print(f"[{args.name}] Ontology is inconsistent! Saved {json_path}.", flush=True)
        return

    all_non_bottom = cm.concept_names - {BOTTOM}
    unsat_names = sorted(c for c in all_non_bottom if not cm.is_satisfiable(c))
    cm_info = {
        "wall_seconds": cm_total_wall,
        "completion_wall_seconds": cm_completion_wall,
        "inconsistent": False,
        "names_count": len(cm.names),
        "unsatisfiable_names_count": len(unsat_names),
    }
    print(
        f"[{args.name}] CanonicalModel built in {cm_total_wall:.2f}s "
        f"(completion={cm_completion_wall:.2f}s): names={len(cm.names)}, "
        f"unsatisfiable={len(unsat_names)}",
        flush=True,
    )

    # Step 3: Bottom-relevant satisfiable concept names
    relevant_all = _bottom_relevant_names(cm.axioms)
    relevant_sat = [name for name in cm.names if name != TOP and name in relevant_all]
    print(
        f"[{args.name}] Bottom-relevant satisfiable names: {len(relevant_sat)} "
        f"(total bottom-relevant: {len(relevant_all)})",
        flush=True,
    )

    # Step 5: Targeted triangle search (run before Step 4 so fast results are saved immediately)
    print(f"[{args.name}] Running targeted triangle search (certificate_triangles)...", flush=True)
    t_tri0 = time.perf_counter()
    cert_tris = certificate_triangles(cm, max_triangles=None)
    tri_wall = time.perf_counter() - t_tri0

    t_graph0 = time.perf_counter()
    named_conj_count, graph_tri_count = _compute_nf2_graph_stats(cm)
    graph_stats_wall = time.perf_counter() - t_graph0

    sample_tris = cert_tris[:50]
    iris_for_labels: Set[str] = set()
    for cert in sample_tris:
        iris_for_labels.update(cert.c1)
        iris_for_labels.update(cert.c2)
        iris_for_labels.update(cert.c3)
        for p in cert.pairwise_names:
            if p is not None:
                iris_for_labels.add(p)

    labels_map = _lookup_labels(ds.ontology, iris_for_labels)
    formatted_sample = [_format_certificate(c, labels_map) for c in sample_tris]

    triangle_info = {
        "wall_seconds": tri_wall,
        "graph_stats_wall_seconds": graph_stats_wall,
        "named_conjunctions_count": named_conj_count,
        "graph_triangles_count": graph_tri_count,
        "certificate_triangles_count": len(cert_tris),
        "certificate_triangles_sample": formatted_sample,
    }
    print(
        f"[{args.name}] Targeted triangle search finished in {tri_wall:.2f}s: "
        f"named_conjunctions={named_conj_count}, graph_triangles={graph_tri_count}, "
        f"certificate_triangles={len(cert_tris)}",
        flush=True,
    )

    # Save partial JSON before launching full helly_check child process
    partial_record = {
        "name": args.name,
        "ontology_path": str(ont_path),
        "status": "running_helly_check",
        "load_wall_seconds": load_wall,
        "normalise": normalise_info,
        "canonical_model": cm_info,
        "bottom_relevant_all_names_count": len(relevant_all),
        "bottom_relevant_satisfiable_names_count": len(relevant_sat),
        "helly_check": {
            "status": "running",
            "timeout_seconds": args.timeout_seconds,
            "k_max": args.k_max,
            "max_certificates": args.max_certificates,
        },
        "targeted_triangle_search": triangle_info,
        "cross_check": None,
        "discrepancies": [],
    }
    _write_json_atomic(json_path, partial_record)
    _update_summary_md(out_dir, work_dir, partial_record)

    # Step 4: Full Helly check in a child process with hard wall-clock timeout
    in_pkl = work_dir / f".helly_in_{args.name}_{os.getpid()}.pkl"
    out_pkl = work_dir / f".helly_out_{args.name}_{os.getpid()}.pkl"
    has_roles = any(isinstance(ax, (NF3, NF4, NF6, NF7)) for ax in cm.axioms)

    print(
        f"[{args.name}] Launching full helly_check in spawned child process "
        f"(k_max={args.k_max}, max_certificates={args.max_certificates}, "
        f"timeout={args.timeout_seconds}s)...",
        flush=True,
    )
    with open(in_pkl, "wb") as f:
        pickle.dump(cm, f, protocol=pickle.HIGHEST_PROTOCOL)

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_helly_child_worker,
        args=(str(in_pkl), str(out_pkl), args.k_max, args.max_certificates, "2g"),
    )
    t_hc0 = time.perf_counter()
    proc.start()
    proc.join(timeout=args.timeout_seconds)
    hc_wall_outer = time.perf_counter() - t_hc0

    timed_out = False
    child_payload: Optional[Dict[str, Any]] = None
    if proc.is_alive():
        timed_out = True
        print(
            f"[{args.name}] helly_check timed out after {hc_wall_outer:.2f}s; killing child PID {proc.pid}...",
            flush=True,
        )
        proc.kill()
        proc.join()
    else:
        if out_pkl.exists():
            with open(out_pkl, "rb") as f:
                child_payload = pickle.load(f)
        else:
            raise RuntimeError(
                f"helly_check child process exited with code {proc.exitcode} without producing output."
            )

    for p in (in_pkl, out_pkl):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    discrepancies: List[Dict[str, Any]] = []
    cross_check_info: Dict[str, Any]

    if timed_out or child_payload is None:
        helly_info = {
            "timed_out": True,
            "timeout_seconds": args.timeout_seconds,
            "wall_seconds": hc_wall_outer,
            "complete": None,
            "has_roles": has_roles,
            "k_max": args.k_max,
            "max_certificates": args.max_certificates,
            "box_representable_strong": None,
            "box_representable_nondegenerate": None,
            "strong_certificates_count": None,
            "nondegenerate_certificates_count": None,
            "minimal_unsatisfiable_set_size_distribution": None,
        }
        cross_check_info = {
            "checked": False,
            "reason": "helly_check timed out",
        }
    else:
        helly_info = {
            "timed_out": False,
            "timeout_seconds": args.timeout_seconds,
            "wall_seconds": child_payload["wall_seconds"],
            "spawn_and_wall_seconds": hc_wall_outer,
            "complete": child_payload["complete"],
            "has_roles": child_payload["has_roles"],
            "k_max": child_payload["k_max"],
            "max_certificates": args.max_certificates,
            "box_representable_strong": child_payload["box_representable_strong"],
            "box_representable_nondegenerate": child_payload["box_representable_nondegenerate"],
            "strong_certificates_count": child_payload["strong_certificates_count"],
            "nondegenerate_certificates_count": child_payload["nondegenerate_certificates_count"],
            "minimal_unsatisfiable_set_size_distribution": child_payload[
                "minimal_unsatisfiable_set_size_distribution"
            ],
        }
        print(
            f"[{args.name}] helly_check completed in {helly_info['wall_seconds']:.2f}s: "
            f"complete={helly_info['complete']}, has_roles={helly_info['has_roles']}, "
            f"strong={helly_info['strong_certificates_count']}, "
            f"nondegenerate={helly_info['nondegenerate_certificates_count']}",
            flush=True,
        )

        # Step 6: Cross-check singleton non-degenerate certificates vs certificate_triangles
        tri_triples_set = {(t.c1, t.c2, t.c3) for t in cert_tris}
        singleton_nondeg = child_payload["singleton_nondegenerate_certificates"]
        helly_singleton_set = {(c1, c2, c3) for (c1, c2, c3, _pw) in singleton_nondeg}

        missing_from_triangles = []
        for c1, c2, c3, pw in singleton_nondeg:
            if (c1, c2, c3) not in tri_triples_set:
                item = {
                    "type": "in_helly_check_singleton_nondegenerate_but_missing_in_certificate_triangles",
                    "c1": list(c1),
                    "c2": list(c2),
                    "c3": list(c3),
                    "pairwise_names": list(pw),
                }
                missing_from_triangles.append(item)
                discrepancies.append(item)

        extra_in_triangles = []
        for t in cert_tris:
            if (t.c1, t.c2, t.c3) not in helly_singleton_set:
                item = {
                    "type": "in_certificate_triangles_but_missing_in_helly_check_singleton_nondegenerate",
                    "c1": list(t.c1),
                    "c2": list(t.c2),
                    "c3": list(t.c3),
                    "pairwise_names": list(t.pairwise_names),
                }
                extra_in_triangles.append(item)
                discrepancies.append(item)

        cross_check_info = {
            "checked": True,
            "helly_singleton_nondegenerate_count": len(singleton_nondeg),
            "helly_non_singleton_nondegenerate_count": len(
                child_payload["non_singleton_nondegenerate_certificates"]
            ),
            "certificate_triangles_count": len(cert_tris),
            "missing_from_triangles_count": len(missing_from_triangles),
            "extra_in_triangles_count": len(extra_in_triangles),
            "missing_from_triangles": missing_from_triangles[:50],
            "extra_in_triangles": extra_in_triangles[:50],
        }
        if discrepancies:
            print(
                f"[{args.name}] WARNING: {len(discrepancies)} discrepancy(ies) found during cross-check!",
                flush=True,
            )
        else:
            print(f"[{args.name}] Cross-check passed with 0 discrepancies.", flush=True)

    final_record = {
        "name": args.name,
        "ontology_path": str(ont_path),
        "status": "completed",
        "load_wall_seconds": load_wall,
        "normalise": normalise_info,
        "canonical_model": cm_info,
        "bottom_relevant_all_names_count": len(relevant_all),
        "bottom_relevant_satisfiable_names_count": len(relevant_sat),
        "helly_check": helly_info,
        "targeted_triangle_search": triangle_info,
        "cross_check": cross_check_info,
        "discrepancies": discrepancies,
    }
    _write_json_atomic(json_path, final_record)
    _update_summary_md(out_dir, work_dir, final_record)
    print(f"[{args.name}] Wrote {json_path} and updated {out_dir / 'summary.md'}.", flush=True)


if __name__ == "__main__":
    main()
