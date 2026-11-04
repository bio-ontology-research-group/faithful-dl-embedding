#!/usr/bin/env python3
"""Synthetic theory builders for GroundedEL Experiment E1 (design §9 E1).

Builds and persists OWL ontologies under ``$TMPDIR/gfe-grounded/`` (defaulting to
``~/tmp/gfe-grounded/``, never ``/tmp``) for:

- ``tk3`` .. ``tk7``: Helly stress-suite theories :math:`T_3 \\dots T_7` (reusing
  :func:`experiments.data.make_synthetic.build_tk_ontology`).
- ``box2el_t1``: :math:`\\{A \\sqsubseteq \\exists r.B,\\; \\exists r.C \\sqsubseteq D,\\; \\exists r.C \\sqsubseteq E,\\; D \\sqcap E \\sqsubseteq \\bot\\}` with :math:`C` declared; all named classes (:math:`A, B, C, D, E`) are satisfiable.
- ``box2el_c4``: 4-cycle intersection obstruction :math:`\\{M_1 \\sqsubseteq A_1, M_1 \\sqsubseteq A_2, M_2 \\sqsubseteq A_2, M_2 \\sqsubseteq A_3, M_3 \\sqsubseteq A_3, M_3 \\sqsubseteq A_4, M_4 \\sqsubseteq A_4, M_4 \\sqsubseteq A_1, A_1 \\sqcap A_3 \\sqsubseteq \\bot, A_2 \\sqcap A_4 \\sqsubseteq \\bot\\}`; all named classes are satisfiable.
- ``role_functional``: :math:`\\{A \\sqsubseteq \\exists r.B,\\; A \\sqsubseteq \\exists r.C,\\; B \\sqcap C \\sqsubseteq \\bot\\}`; all named classes (:math:`A, B, C`) are satisfiable.
- ``role_cycle``: :math:`\\{A \\sqsubseteq \\exists r.B,\\; B \\sqsubseteq \\exists r.A,\\; A \\sqcap B \\sqsubseteq \\bot\\}`; all named classes (:math:`A, B`) are satisfiable.
- ``role_chain``: :math:`\\{A \\sqsubseteq \\exists r.B,\\; B \\sqsubseteq \\exists s.C,\\; r \\circ s \\sqsubseteq t,\\; \\exists t.C \\sqsubseteq D,\\; s \\circ r \\sqsubseteq u,\\; \\exists u.A \\sqsubseteq E,\\; D \\sqcap E \\sqsubseteq \\bot,\\; F \\sqsubseteq \\exists s.G,\\; G \\sqsubseteq \\exists r.A\\}`. Both role chains fire (:math:`A \\sqsubseteq D` via :math:`r \\circ s \\sqsubseteq t` and :math:`F \\sqsubseteq E` via :math:`s \\circ r \\sqsubseteq u`) and all named classes (:math:`A, B, C, D, E, F, G`) are satisfiable (verified with :class:`~mowl.geometric.CanonicalModel`; no axiom needed to be dropped).
- ``unsat_mix``: :math:`T_3` augmented with :math:`\\{Q \\sqsubseteq A_1,\\; Q \\sqsubseteq A_2\\}` so that :math:`Q` is unsatisfiable while :math:`X_1, X_2, X_3, A_1, A_2, A_3, Z` are satisfiable (tests :math:`\\delta`).

Guarantees
----------
- Every builder writes a valid OWL file under ``$TMPDIR/gfe-grounded/`` atomically
  (via a process-local temporary file and ``os.replace``) that loads cleanly via
  :class:`mowl.datasets.PathDataset`.
- Never writes under ``/tmp``.
- All named classes in ``tk3``..``tk7``, ``box2el_t1``, ``box2el_c4``,
  ``role_functional``, ``role_cycle``, and ``role_chain`` are satisfiable; in
  ``unsat_mix``, ``Q`` is the unique unsatisfiable named class.

Non-guarantees
--------------
- Requires the JVM to be started (automatically initialised to ``4g`` by the
  file-writing helpers if not already running).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mowl

from experiments.config import SYNTHETIC_NS, SYNTHETIC_ONT_BASE
from experiments.data.make_synthetic import build_tk_ontology

THEORIES: Tuple[str, ...] = (
    "tk3",
    "tk4",
    "tk5",
    "tk6",
    "tk7",
    "box2el_t1",
    "box2el_c4",
    "role_functional",
    "role_cycle",
    "role_chain",
    "unsat_mix",
)


def default_grounded_tmp_dir() -> Path:
    """Resolve ``$TMPDIR/gfe-grounded`` (falling back to ``~/tmp/gfe-grounded``; never ``/tmp``).

    Returns
    -------
    Path
        Resolved directory path under ``$TMPDIR`` or ``~/tmp``.

    Guarantees
    ----------
    - Raises :class:`ValueError` if ``$TMPDIR`` resolves to ``/tmp`` or a
      subdirectory of ``/tmp`` without being symlinked outside ``/tmp``.

    Non-guarantees
    --------------
    - Does not create the directory on disk.
    """
    raw_tmp = os.environ.get("TMPDIR", "").strip()
    if not raw_tmp or raw_tmp == "/tmp" or raw_tmp.startswith("/tmp/"):
        base = Path("~/tmp").expanduser()
    else:
        base = Path(raw_tmp).expanduser()
    resolved_base = base.resolve()
    if str(resolved_base) == "/tmp" or str(resolved_base).startswith("/tmp/"):
        raise ValueError(
            f"Refusing to write under /tmp (resolved base={resolved_base}); set TMPDIR=~/tmp"
        )
    return resolved_base / "gfe-grounded"


def _ensure_jvm(mem: str = "4g") -> None:
    """Start the mOWL JVM if not already started."""
    mowl.init_jvm(mem)


def build_box2el_t1_ontology():
    """Construct the ``box2el_t1`` OWL ontology :math:`\\{A \\sqsubseteq \\exists r.B, \\exists r.C \\sqsubseteq D, \\exists r.C \\sqsubseteq E, D \\sqcap E \\sqsubseteq \\bot\\}`.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Declares classes ``A, B, C, D, E`` and object property ``r``.
    - All five named classes ``A, B, C, D, E`` are satisfiable.

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{SYNTHETIC_ONT_BASE}/box2el_t1")
    bot = adapter.create_class(BOT)

    cls = {
        name: adapter.create_class(f"{SYNTHETIC_NS}{name}")
        for name in ("A", "B", "C", "D", "E")
    }
    r = adapter.create_object_property(f"{SYNTHETIC_NS}r")

    for c_obj in cls.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(r))

    # A <= exists r.B
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["A"], adapter.create_object_some_values_from(r, cls["B"])
        ),
    )
    # exists r.C <= D
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(r, cls["C"]), cls["D"]
        ),
    )
    # exists r.C <= E
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(r, cls["C"]), cls["E"]
        ),
    )
    # D cap E <= bot
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["D"], cls["E"]), bot
        ),
    )
    return ont


def build_box2el_c4_ontology():
    """Construct the ``box2el_c4`` 4-cycle OWL ontology.

    Contains :math:`\\{M_1 \\sqsubseteq A_1, M_1 \\sqsubseteq A_2, M_2 \\sqsubseteq A_2, M_2 \\sqsubseteq A_3, M_3 \\sqsubseteq A_3, M_3 \\sqsubseteq A_4, M_4 \\sqsubseteq A_4, M_4 \\sqsubseteq A_1, A_1 \\sqcap A_3 \\sqsubseteq \\bot, A_2 \\sqcap A_4 \\sqsubseteq \\bot\\}`.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Declares classes ``M1..M4`` and ``A1..A4``, all satisfiable.

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{SYNTHETIC_ONT_BASE}/box2el_c4")
    bot = adapter.create_class(BOT)

    ms = {i: adapter.create_class(f"{SYNTHETIC_NS}M{i}") for i in range(1, 5)}
    as_ = {i: adapter.create_class(f"{SYNTHETIC_NS}A{i}") for i in range(1, 5)}

    for c_obj in list(ms.values()) + list(as_.values()):
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))

    pairs = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 1)]
    for m_idx, a_idx in pairs:
        mgr.addAxiom(ont, adapter.create_subclass_of(ms[m_idx], as_[a_idx]))

    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(as_[1], as_[3]), bot
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(as_[2], as_[4]), bot
        ),
    )
    return ont


def build_role_functional_ontology():
    """Construct the ``role_functional`` OWL ontology :math:`\\{A \\sqsubseteq \\exists r.B, A \\sqsubseteq \\exists r.C, B \\sqcap C \\sqsubseteq \\bot\\}`.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Declares classes ``A, B, C`` and role ``r``, with ``A, B, C`` all satisfiable.

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{SYNTHETIC_ONT_BASE}/role_functional")
    bot = adapter.create_class(BOT)

    cls = {
        name: adapter.create_class(f"{SYNTHETIC_NS}{name}")
        for name in ("A", "B", "C")
    }
    r = adapter.create_object_property(f"{SYNTHETIC_NS}r")

    for c_obj in cls.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(r))

    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["A"], adapter.create_object_some_values_from(r, cls["B"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["A"], adapter.create_object_some_values_from(r, cls["C"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["B"], cls["C"]), bot
        ),
    )
    return ont


def build_role_cycle_ontology():
    """Construct the ``role_cycle`` OWL ontology :math:`\\{A \\sqsubseteq \\exists r.B, B \\sqsubseteq \\exists r.A, A \\sqcap B \\sqsubseteq \\bot\\}`.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Declares classes ``A, B`` and role ``r``, with ``A, B`` both satisfiable.

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{SYNTHETIC_ONT_BASE}/role_cycle")
    bot = adapter.create_class(BOT)

    cls = {
        name: adapter.create_class(f"{SYNTHETIC_NS}{name}")
        for name in ("A", "B")
    }
    r = adapter.create_object_property(f"{SYNTHETIC_NS}r")

    for c_obj in cls.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(r))

    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["A"], adapter.create_object_some_values_from(r, cls["B"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["B"], adapter.create_object_some_values_from(r, cls["A"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["A"], cls["B"]), bot
        ),
    )
    return ont


def build_role_chain_ontology():
    """Construct the ``role_chain`` OWL ontology with non-commuting chains :math:`r \\circ s \\sqsubseteq t` and :math:`s \\circ r \\sqsubseteq u`.

    Contains:
    :math:`\\{A \\sqsubseteq \\exists r.B,\\; B \\sqsubseteq \\exists s.C,\\; r \\circ s \\sqsubseteq t,\\; \\exists t.C \\sqsubseteq D,\\; s \\circ r \\sqsubseteq u,\\; \\exists u.A \\sqsubseteq E,\\; D \\sqcap E \\sqsubseteq \\bot,\\; F \\sqsubseteq \\exists s.G,\\; G \\sqsubseteq \\exists r.A\\}`.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Declares classes ``A, B, C, D, E, F, G`` and roles ``r, s, t, u``.
    - All seven named classes ``A..G`` are satisfiable, and both role chains fire
      (:math:`A \\sqsubseteq D` and :math:`F \\sqsubseteq E`).

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    from java.util import Arrays
    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT

    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology(f"{SYNTHETIC_ONT_BASE}/role_chain")
    bot = adapter.create_class(BOT)

    cls = {
        name: adapter.create_class(f"{SYNTHETIC_NS}{name}")
        for name in ("A", "B", "C", "D", "E", "F", "G")
    }
    rels = {
        name: adapter.create_object_property(f"{SYNTHETIC_NS}{name}")
        for name in ("r", "s", "t", "u")
    }

    for c_obj in cls.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
    for r_obj in rels.values():
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(r_obj))

    # A <= exists r.B, B <= exists s.C
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["A"], adapter.create_object_some_values_from(rels["r"], cls["B"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["B"], adapter.create_object_some_values_from(rels["s"], cls["C"])
        ),
    )
    # r o s <= t, exists t.C <= D
    mgr.addAxiom(
        ont,
        df.getOWLSubPropertyChainOfAxiom(
            Arrays.asList([rels["r"], rels["s"]]), rels["t"]
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(rels["t"], cls["C"]), cls["D"]
        ),
    )
    # s o r <= u, exists u.A <= E
    mgr.addAxiom(
        ont,
        df.getOWLSubPropertyChainOfAxiom(
            Arrays.asList([rels["s"], rels["r"]]), rels["u"]
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_some_values_from(rels["u"], cls["A"]), cls["E"]
        ),
    )
    # D cap E <= bot
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_object_intersection_of(cls["D"], cls["E"]), bot
        ),
    )
    # F <= exists s.G, G <= exists r.A so s o r <= u also fires
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["F"], adapter.create_object_some_values_from(rels["s"], cls["G"])
        ),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            cls["G"], adapter.create_object_some_values_from(rels["r"], cls["A"])
        ),
    )
    return ont


def build_unsat_mix_ontology():
    """Construct the ``unsat_mix`` OWL ontology (:math:`T_3` plus :math:`Q \\sqsubseteq A_1, Q \\sqsubseteq A_2`).

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology in namespace ``http://mowl.borg/geo#``.

    Guarantees
    ----------
    - Contains all axioms of :math:`T_3` plus ``Q <= A1`` and ``Q <= A2``.
    - ``Q`` is unsatisfiable while ``X1, X2, X3, A1, A2, A3, Z`` are satisfiable.

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    from mowl.owlapi import OWLAPIAdapter

    ont = build_tk_ontology(3)
    adapter = OWLAPIAdapter()
    mgr = ont.getOWLOntologyManager()
    df = mgr.getOWLDataFactory()

    q_cls = adapter.create_class(f"{SYNTHETIC_NS}Q")
    a1_cls = adapter.create_class(f"{SYNTHETIC_NS}A1")
    a2_cls = adapter.create_class(f"{SYNTHETIC_NS}A2")

    mgr.addAxiom(ont, df.getOWLDeclarationAxiom(q_cls))
    mgr.addAxiom(ont, adapter.create_subclass_of(q_cls, a1_cls))
    mgr.addAxiom(ont, adapter.create_subclass_of(q_cls, a2_cls))
    return ont


def build_theory_ontology(name: str):
    """Build the OWLAPI ``OWLOntology`` for synthetic theory ``name``.

    Parameters
    ----------
    name : str
        Theory identifier in ``THEORIES``.

    Returns
    -------
    org.semanticweb.owlapi.model.OWLOntology
        Constructed OWL ontology.

    Guarantees
    ----------
    - Raises :class:`ValueError` if ``name`` is not in ``THEORIES``.

    Non-guarantees
    --------------
    - Requires the JVM to be started before calling.
    """
    if name not in THEORIES:
        raise ValueError(f"Unknown theory {name!r}; expected one of {THEORIES}")
    if name.startswith("tk"):
        k = int(name[2:])
        return build_tk_ontology(k)
    if name == "box2el_t1":
        return build_box2el_t1_ontology()
    if name == "box2el_c4":
        return build_box2el_c4_ontology()
    if name == "role_functional":
        return build_role_functional_ontology()
    if name == "role_cycle":
        return build_role_cycle_ontology()
    if name == "role_chain":
        return build_role_chain_ontology()
    if name == "unsat_mix":
        return build_unsat_mix_ontology()
    raise ValueError(f"Unhandled theory {name!r}")  # pragma: no cover


def verify_theory_ontology(name: str, ont) -> Dict[str, List[str]]:
    """Verify satisfiability invariants of ``ont`` using :class:`~mowl.geometric.CanonicalModel`.

    Parameters
    ----------
    name : str
        Theory identifier in ``THEORIES``.
    ont : org.semanticweb.owlapi.model.OWLOntology
        Constructed ontology to check.

    Returns
    -------
    Dict[str, List[str]]
        ``{"satisfiable": [...], "unsatisfiable": [...]}`` over non-top/bottom
        classes in ``ont``.

    Guarantees
    ----------
    - Asserts that ``normalise(ont).dropped_axioms == 0``.
    - For ``unsat_mix``, asserts that ``Q`` is the unique unsatisfiable class and
      all ``T_3`` classes are satisfiable.
    - For every other theory in ``THEORIES``, asserts that all declared classes
      are satisfiable (and for ``role_chain``, asserts that both chains fire:
      :math:`A \\sqsubseteq D` and :math:`F \\sqsubseteq E`).

    Non-guarantees
    --------------
    - Uses :class:`~mowl.geometric.CanonicalModel` (for verification/testing only).
    """
    from mowl.geometric import CanonicalModel, normalise
    from mowl.owlapi.defaults import BOT, TOP

    no = normalise(ont)
    assert no.dropped_axioms == 0, (
        f"Expected 0 dropped axioms on {name}, got {no.dropped_axioms}"
    )
    cm = CanonicalModel.from_normalised(no)

    sat_names: List[str] = []
    unsat_names: List[str] = []
    for cls_obj in ont.getClassesInSignature():
        iri = str(cls_obj.toStringID())
        if iri in (TOP, BOT):
            continue
        if cm.is_satisfiable(iri):
            sat_names.append(iri)
        else:
            unsat_names.append(iri)

    sat_names.sort()
    unsat_names.sort()

    if name == "unsat_mix":
        expected_unsat = [f"{SYNTHETIC_NS}Q"]
        assert unsat_names == expected_unsat, (
            f"Expected unsat={expected_unsat} on unsat_mix, got {unsat_names}"
        )
        assert len(sat_names) >= 7, (
            f"Expected >= 7 satisfiable classes on unsat_mix, got {sat_names}"
        )
    else:
        assert unsat_names == [], (
            f"Expected all named classes satisfiable on {name}, got unsat={unsat_names}"
        )
        assert len(sat_names) >= 2, (
            f"Expected >= 2 satisfiable classes on {name}, got {sat_names}"
        )

    if name == "role_chain":
        # Verify both role chains fire: A <= D (via r o s <= t) and F <= E (via s o r <= u)
        a_iri = f"{SYNTHETIC_NS}A"
        d_iri = f"{SYNTHETIC_NS}D"
        f_iri = f"{SYNTHETIC_NS}F"
        e_iri = f"{SYNTHETIC_NS}E"
        assert cm.extension(a_iri).issubset(cm.extension(d_iri)), (
            "Expected r o s <= t chain to entail A <= D in role_chain"
        )
        assert cm.extension(f_iri).issubset(cm.extension(e_iri)), (
            "Expected s o r <= u chain to entail F <= E in role_chain"
        )

    return {"satisfiable": sat_names, "unsatisfiable": unsat_names}


def build_theory(
    name: str,
    out_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``<out_dir>/<name>.owl`` for synthetic theory ``name`` and return its path.

    Parameters
    ----------
    name : str
        Theory identifier in ``THEORIES``.
    out_dir : Path or str or None, optional
        Output directory; defaults to ``$TMPDIR/gfe-grounded/``.
    overwrite : bool, default False
        If ``False`` and a non-empty ``<name>.owl`` already exists on disk, reuses
        it without rewriting.

    Returns
    -------
    Path
        Resolved path to the saved ``<name>.owl`` file (loadable via
        :class:`mowl.datasets.PathDataset`).

    Guarantees
    ----------
    - Writes via a process-local temporary file and atomic ``os.replace`` so
      concurrent calls across worker processes cannot corrupt ``<name>.owl``.
    - Never writes under ``/tmp``.

    Non-guarantees
    --------------
    - Starts the JVM if not already running.
    """
    if name not in THEORIES:
        raise ValueError(f"Unknown theory {name!r}; expected one of {THEORIES}")

    target_dir = (
        default_grounded_tmp_dir()
        if out_dir is None
        else Path(out_dir).expanduser().resolve()
    )
    if str(target_dir) == "/tmp" or str(target_dir).startswith("/tmp/"):
        raise ValueError(f"Refusing to write under /tmp: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / f"{name}.owl"
    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    _ensure_jvm("4g")
    from org.semanticweb.owlapi.model import IRI

    ont = build_theory_ontology(name)
    tmp_path = target_dir / f".{name}.{os.getpid()}.tmp.owl"
    ont.getOWLOntologyManager().saveOntology(
        ont, IRI.create("file:" + str(tmp_path.resolve()))
    )
    os.replace(tmp_path, out_path)
    return out_path


def load_theory_dataset(
    name: str,
    out_dir: Path | str | None = None,
    overwrite: bool = False,
):
    """Build (if needed) and load synthetic theory ``name`` as a :class:`mowl.datasets.PathDataset`.

    Parameters
    ----------
    name : str
        Theory identifier in ``THEORIES``.
    out_dir : Path or str or None, optional
        Output directory for ``<name>.owl``; defaults to ``$TMPDIR/gfe-grounded/``.
    overwrite : bool, default False
        Whether to force regeneration of ``<name>.owl``.

    Returns
    -------
    mowl.datasets.PathDataset
        Loaded dataset wrapping ``<out_dir>/<name>.owl``.

    Guarantees
    ----------
    - Starts the JVM if not already running and returns a ready ``PathDataset``.

    Non-guarantees
    --------------
    - Validation and testing ontologies on the returned ``PathDataset`` are ``None``.
    """
    _ensure_jvm("4g")
    from mowl.datasets import PathDataset

    owl_path = build_theory(name, out_dir=out_dir, overwrite=overwrite)
    return PathDataset(str(owl_path))


def generate_all_theories(
    out_dir: Path | str | None = None,
    theories: Iterable[str] = THEORIES,
    verify: bool = True,
    overwrite: bool = True,
) -> Dict[str, Path]:
    """Generate, optionally verify with :class:`~mowl.geometric.CanonicalModel`, and reload all requested theories.

    Parameters
    ----------
    out_dir : Path or str or None, optional
        Output directory; defaults to ``$TMPDIR/gfe-grounded/``.
    theories : Iterable[str], default THEORIES
        Theory names to build.
    verify : bool, default True
        Whether to run :func:`verify_theory_ontology` on each reloaded
        :class:`~mowl.datasets.PathDataset`.
    overwrite : bool, default True
        Whether to overwrite existing ``.owl`` files.

    Returns
    -------
    Dict[str, Path]
        Mapping ``theory_name -> owl_path``.

    Guarantees
    ----------
    - Reloads every generated file via :class:`~mowl.datasets.PathDataset` to
      confirm compatibility.

    Non-guarantees
    --------------
    - Overwrites existing ``.owl`` files when ``overwrite=True``.
    """
    _ensure_jvm("4g")
    from mowl.datasets import PathDataset

    saved: Dict[str, Path] = {}
    for name in theories:
        path = build_theory(name, out_dir=out_dir, overwrite=overwrite)
        ds = PathDataset(str(path))
        if verify:
            verify_theory_ontology(name, ds.ontology)
        saved[name] = path
    return saved


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point to generate and verify E1 synthetic theory OWL files."""
    parser = argparse.ArgumentParser(
        description="Build and verify GroundedEL E1 synthetic theories."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for <theory>.owl files (default: $TMPDIR/gfe-grounded/).",
    )
    parser.add_argument(
        "--theories",
        type=str,
        nargs="+",
        default=list(THEORIES),
        choices=list(THEORIES),
        help="Theories to generate.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip CanonicalModel verification.",
    )
    args = parser.parse_args(argv)
    saved = generate_all_theories(
        out_dir=args.out_dir,
        theories=args.theories,
        verify=not args.no_verify,
        overwrite=True,
    )
    for name, path in saved.items():
        print(f"Generated {name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
