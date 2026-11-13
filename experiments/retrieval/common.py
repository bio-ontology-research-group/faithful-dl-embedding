"""Shared OWLAPI utilities for instance-retrieval benchmarks.

Inputs are OWL files and JSON query expressions; outputs are OWLAPI-backed
indexes, class expressions, and stable JSON-compatible identifiers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

NAMES = ("carcinogenesis", "mutagenesis", "semantic_bible", "vicodi", "family", "father")
DATA_ROOT = Path(os.environ.get("GFE_RETRIEVAL_DATA", Path.home() / "tmp/gfe/retrieval")).expanduser()


#: Hard ceiling for the whole process (heap + JVM native + Python), in GiB.
TOTAL_MEMORY_CAP_GB = 24
#: Memory kept for JVM native structures, Python and numpy, in GiB.
NATIVE_RESERVE_GB = 4
DEFAULT_HEAP_GB = 16


def mem_available_gb() -> float:
    """Return ``MemAvailable`` from ``/proc/meminfo`` in GiB (``inf`` if unreadable)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("inf")


def choose_heap_gb(requested: int | None = None) -> int:
    """Choose a JVM heap that fits both the free memory and the 24 GB process cap.

    Parameters
    ----------
    requested : int or None
        Desired heap in GiB; defaults to ``$GFE_RETRIEVAL_HEAP_GB`` or 16.

    Returns
    -------
    int
        Heap size in GiB, at most ``TOTAL_MEMORY_CAP_GB - NATIVE_RESERVE_GB`` and at most
        ``MemAvailable - NATIVE_RESERVE_GB``.

    Raises
    ------
    MemoryError
        If less than 1 GiB of heap would fit.
    """
    want = requested or int(os.environ.get("GFE_RETRIEVAL_HEAP_GB", DEFAULT_HEAP_GB))
    available = mem_available_gb()
    heap = int(min(want, TOTAL_MEMORY_CAP_GB - NATIVE_RESERVE_GB, available - NATIVE_RESERVE_GB))
    if heap < 1:
        raise MemoryError(f"only {available:.1f} GiB available; refusing to start the JVM")
    return heap


def jvm_options(heap_gb: int) -> list[str]:
    """JVM flags: bounded heap/native areas and silenced slf4j-simple logging."""
    return [
        f"-Xmx{heap_gb}g", "-Xms512m", "-Xss16m",
        "-XX:ParallelGCThreads=8", "-XX:ConcGCThreads=2", "-XX:CICompilerCount=4",
        "-XX:ReservedCodeCacheSize=256m", "-XX:MaxMetaspaceSize=512m", "-XX:CompressedClassSpaceSize=256m",
        "-XX:MaxDirectMemorySize=512m",
        "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
        "-Dorg.slf4j.simpleLogger.logFile=System.err",
    ]


def silence_java_logging() -> None:
    """Set log4j (used by ELK) and java.util.logging to ERROR, with a stderr appender."""
    from org.apache.log4j import BasicConfigurator, Level, Logger, LogManager
    root = Logger.getRootLogger()
    if not root.getAllAppenders().hasMoreElements():
        BasicConfigurator.configure()
    root.setLevel(Level.ERROR)
    for name in ("org.semanticweb.elk", "org.semanticweb.owlapi", "uk.ac.manchester.cs"):
        LogManager.getLogger(name).setLevel(Level.ERROR)
    from java.util.logging import Level as JulLevel, Logger as JulLogger
    JulLogger.getLogger("").setLevel(JulLevel.SEVERE)


def start_jvm(heap_gb: int | None = None) -> None:
    """Start the JVM once on mOWL's jar directory (OWLAPI, ELK, HermiT) with a checked heap.

    The classpath is the one ``mowl.init_jvm`` uses; the JVM is started here so that heap and
    logging flags can be set. If a JVM is already running (e.g. started by ``mowl.init_jvm``
    in a test) only the logging levels are adjusted.
    """
    import jpype
    if not jpype.isJVMStarted():
        import importlib.util
        import sys
        spec = importlib.util.find_spec("mowl")
        jars_dir = Path(spec.origin).parent / "lib"
        jars = sorted(str(p) for p in jars_dir.iterdir() if p.suffix == ".jar")
        heap = choose_heap_gb(heap_gb)
        print(f"[jvm] MemAvailable={mem_available_gb():.1f} GiB, heap={heap} GiB", file=sys.stderr, flush=True)
        import jpype.imports  # noqa: F401  (enables `from org... import`)
        jpype.startJVM(jpype.getDefaultJVMPath(), *jvm_options(heap), classpath=jars, convertStrings=False)
    silence_java_logging()


def load_ontology(path: Path):
    """Load an ontology from a local OWL document using OWLAPI."""
    start_jvm()
    from org.semanticweb.owlapi.apibinding import OWLManager
    from java.io import File
    manager = OWLManager.createOWLOntologyManager()
    ontology = manager.loadOntologyFromOntologyDocument(File(str(path.resolve())))
    return manager, ontology


def iri(entity: Any) -> str:
    """Return an OWL entity's IRI as a Python string."""
    return str(entity.getIRI().toString())


def expr_to_owl(expr: Any, df: Any):
    """Convert the benchmark's small s-expression DSL into an OWL class expression."""
    if isinstance(expr, str):
        return df.getOWLClass(_java_iri(expr))
    op = expr[0]
    if op == "and":
        return df.getOWLObjectIntersectionOf(expr_to_owl(x, df) for x in expr[1:])
    if op == "some":
        role, filler = expr[1], expr[2]
        if isinstance(role, list) and role[0] == "inv":
            prop = df.getOWLObjectInverseOf(df.getOWLObjectProperty(_java_iri(role[1])))
        else:
            prop = df.getOWLObjectProperty(_java_iri(role))
        return df.getOWLObjectSomeValuesFrom(prop, expr_to_owl(filler, df))
    raise ValueError(f"Unknown expression operator: {op!r}")


def _java_iri(value: str):
    from org.semanticweb.owlapi.model import IRI
    return IRI.create(value)


def expression_key(expr: Any) -> str:
    """Canonical compact representation of an expression."""
    return json.dumps(expr, sort_keys=False, separators=(",", ":"))


def build_index(ontology: Any, told_hierarchy: bool = False) -> dict[str, Any]:
    """Extract named individuals, asserted class/role facts and named subclass edges."""
    individuals = {iri(x) for x in ontology.getIndividualsInSignature()}
    types: dict[str, set[str]] = {x: set() for x in individuals}
    roles: dict[str, dict[str, set[str]]] = {}
    hierarchy: dict[str, set[str]] = {}
    from org.semanticweb.owlapi.model.parameters import Imports
    from org.semanticweb.owlapi.model import AxiomType
    for ax in ontology.getABoxAxioms(Imports.EXCLUDED):
        if ax.isOfType(__import__("org.semanticweb.owlapi.model", fromlist=["AxiomType"]).AxiomType.CLASS_ASSERTION):
            ce = ax.getClassExpression()
            if not ce.isAnonymous():
                types.setdefault(iri(ax.getIndividual()), set()).add(iri(ce.asOWLClass()))
        elif ax.isOfType(__import__("org.semanticweb.owlapi.model", fromlist=["AxiomType"]).AxiomType.OBJECT_PROPERTY_ASSERTION):
            p = ax.getProperty()
            subject, obj = iri(ax.getSubject()), iri(ax.getObject())
            if p.isAnonymous():
                p = p.getInverseProperty()
                subject, obj = obj, subject
            roles.setdefault(iri(p.asOWLObjectProperty()), {}).setdefault(subject, set()).add(obj)
    if told_hierarchy:
        for ax in ontology.getAxioms(AxiomType.SUBCLASS_OF, Imports.EXCLUDED):
            if not ax.getSubClass().isAnonymous() and not ax.getSuperClass().isAnonymous():
                hierarchy.setdefault(iri(ax.getSubClass().asOWLClass()), set()).add(iri(ax.getSuperClass().asOWLClass()))
        changed = True
        while changed:
            changed = False
            for c in list(hierarchy):
                expanded = set().union(*(hierarchy.get(x, set()) for x in hierarchy[c]))
                if not expanded <= hierarchy[c]:
                    hierarchy[c] |= expanded
                    changed = True
        for ind, cs in types.items():
            types[ind] |= set().union(*(hierarchy.get(c, set()) for c in tuple(cs))) if cs else set()
    return {"individuals": individuals, "types": types, "roles": roles, "hierarchy": hierarchy}


def eval_structural(expr: Any, index: dict[str, Any]) -> set[str]:
    """Evaluate an expression over asserted facts in a finite closed-world index."""
    if isinstance(expr, str):
        return {i for i, cs in index["types"].items() if expr in cs}
    op = expr[0]
    if op == "and":
        sets = [eval_structural(x, index) for x in expr[1:]]
        return set.intersection(*sets) if sets else set(index["individuals"])
    if op == "some":
        role, filler = expr[1], expr[2]
        fill = eval_structural(filler, index)
        if isinstance(role, list) and role[0] == "inv":
            edges = index["roles"].get(role[1], {})
            return {obj for subject, objs in edges.items() if subject in fill for obj in objs}
        edges = index["roles"].get(role, {})
        return {subject for subject, objs in edges.items() if objs & fill}
    raise ValueError(f"Unknown expression operator: {op!r}")


def java_expr_owl(expr: Any, df: Any):
    """Build class expression; unlike s-expression parsing, handles generated Python lists safely."""
    if isinstance(expr, str):
        return df.getOWLClass(_java_iri(expr))
    if expr[0] == "and":
        from java.util import HashSet
        operands = HashSet()
        for item in expr[1:]:
            operands.add(java_expr_owl(item, df))
        return df.getOWLObjectIntersectionOf(operands)
    if expr[0] == "some":
        role, filler = expr[1:]
        prop = df.getOWLObjectProperty(_java_iri(role[1])) if isinstance(role, list) else df.getOWLObjectProperty(_java_iri(role))
        if isinstance(role, list):
            prop = df.getOWLObjectInverseOf(prop)
        return df.getOWLObjectSomeValuesFrom(prop, java_expr_owl(filler, df))
    raise ValueError(expr)


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning a default only when the file is absent."""
    return json.loads(path.read_text()) if path.exists() else default
