"""Fast OWLAPI regression tests for EL instance retrieval and metric conventions."""
from __future__ import annotations

from common import build_index, eval_structural, java_expr_owl, load_ontology, iri
from evaluate import scores


def _ontology(tmp_path):
    """Build the deterministic ten-individual fixture with asserted and TBox facts."""
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import IRI
    manager=OWLManager.createOWLOntologyManager();df=manager.getOWLDataFactory()
    ont=manager.createOntology(IRI.create("urn:retrieval-test"))
    cls=lambda name:df.getOWLClass(IRI.create("urn:retrieval#"+name))
    ind=lambda n:df.getOWLNamedIndividual(IRI.create("urn:retrieval#i"+str(n)))
    role=lambda name:df.getOWLObjectProperty(IRI.create("urn:retrieval#"+name))
    A,B,C=cls("A"),cls("B"),cls("C");r,s=role("r"),role("s")
    for n in (0,1,2,3):manager.addAxiom(ont,df.getOWLClassAssertionAxiom(A,ind(n)))
    for n in (1,2,4):manager.addAxiom(ont,df.getOWLClassAssertionAxiom(B,ind(n)))
    for n in (3,4):manager.addAxiom(ont,df.getOWLClassAssertionAxiom(C,ind(n)))
    for x,y in ((0,1),(1,2),(2,4),(5,0)):manager.addAxiom(ont,df.getOWLObjectPropertyAssertionAxiom(r,ind(x),ind(y)))
    for x,y in ((1,3),(2,3)):manager.addAxiom(ont,df.getOWLObjectPropertyAssertionAxiom(s,ind(x),ind(y)))
    manager.addAxiom(ont,df.getOWLSubClassOfAxiom(A,C))
    path=tmp_path/"fixture.owl";manager.saveOntology(ont,IRI.create(path.resolve().as_uri()))
    return path


def test_query_forms_lookup_and_reasoner(tmp_path):
    """All seven EL expression forms agree with known HermiT answers; told closure adds entailments."""
    import mowl  # ensures OWLAPI jars are present; JVM starts through mOWL
    mowl.init_jvm("2g")
    from org.semanticweb.HermiT import ReasonerFactory
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import IRI
    path=_ontology(tmp_path)
    _,ont=load_ontology(path);index=build_index(ont);told=build_index(ont,True)
    df=ont.getOWLOntologyManager().getOWLDataFactory();reasoner=ReasonerFactory().createReasoner(ont)
    prefix="urn:retrieval#"
    cases=[
      (prefix+"A",{prefix+f"i{n}" for n in (0,1,2,3)}),
      (["and",prefix+"A",prefix+"B"],{prefix+"i1",prefix+"i2"}),
      (["some",prefix+"r",prefix+"B"],{prefix+"i0",prefix+"i1",prefix+"i2"}),
      (["and",prefix+"A",["some",prefix+"r",prefix+"B"]],{prefix+"i0",prefix+"i1",prefix+"i2"}),
      (["some",prefix+"r",["and",prefix+"A",prefix+"B"]],{prefix+"i0",prefix+"i1"}),
      (["some",prefix+"r",["some",prefix+"s",prefix+"A"]],{prefix+"i0",prefix+"i1"}),
      (["some",["inv",prefix+"r"],prefix+"A"],{prefix+"i1",prefix+"i2",prefix+"i4"}),
    ]
    for expr,expected in cases:
        assert eval_structural(expr,index)==expected
        nodes=reasoner.getInstances(java_expr_owl(expr,df),False).getNodes()
        actual={iri(i) for node in nodes for i in node.getEntities()}
        assert actual==expected
    c_expr=prefix+"C"
    assert eval_structural(c_expr,index)=={prefix+"i3",prefix+"i4"}
    assert eval_structural(c_expr,told)=={prefix+f"i{n}" for n in (0,1,2,3,4)}
    assert {iri(i) for node in reasoner.getInstances(java_expr_owl(c_expr,df),False).getNodes() for i in node.getEntities()}=={prefix+f"i{n}" for n in (0,1,2,3,4)}
    reasoner.dispose()


def test_jaccard_and_classification_edge_cases():
    """Both-empty convention is one; one-sided empty predictions score zero."""
    assert scores([],[])=={"jaccard":1.0,"precision":1.0,"recall":1.0,"f1":1.0}
    assert scores([], ["x"])["jaccard"]==0.0
    assert scores(["x"], [])["recall"]==0.0
    assert scores(["x","y"],["y","z"])["jaccard"]==1/3


# --- Materialisation filter, sampler and ground-truth driver (task C2) -----------------------

P = "urn:retrieval#"


def _role_ontologies(tmp_path):
    """Write an EL file and a full file that adds inverse/symmetric RBox axioms.

    Both contain the ten-individual fixture plus ``r ⊑ t``, ``Transitive(t)`` and one ``u``
    edge ``i3 -> i4``; the full file adds ``InverseObjectProperties(r, rinv)`` and
    ``Symmetric(u)``.
    """
    from org.semanticweb.owlapi.model import IRI
    paths = {}
    for variant in ("el", "full"):
        sub = tmp_path / variant
        sub.mkdir(parents=True)
        _, ont = load_ontology(_ontology(sub))
        manager = ont.getOWLOntologyManager(); df = manager.getOWLDataFactory()
        role = lambda name: df.getOWLObjectProperty(IRI.create(P + name))  # noqa: E731
        ind = lambda n: df.getOWLNamedIndividual(IRI.create(P + f"i{n}"))  # noqa: E731
        manager.addAxiom(ont, df.getOWLSubObjectPropertyOfAxiom(role("r"), role("t")))
        manager.addAxiom(ont, df.getOWLTransitiveObjectPropertyAxiom(role("t")))
        manager.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(role("u"), ind(3), ind(4)))
        if variant == "full":
            manager.addAxiom(ont, df.getOWLInverseObjectPropertiesAxiom(role("r"), role("rinv")))
            manager.addAxiom(ont, df.getOWLSymmetricObjectPropertyAxiom(role("u")))
            manager.addAxiom(ont, df.getOWLDeclarationAxiom(role("rinv")))
        path = tmp_path / f"fixture_{variant}.owl"
        manager.saveOntology(ont, IRI.create(path.resolve().as_uri()))
        paths[variant] = path
    return paths["el"], paths["full"]


def _hermit_answers(ontology, expr):
    from org.semanticweb.HermiT import ReasonerFactory
    reasoner = ReasonerFactory().createReasoner(ontology)
    df = ontology.getOWLOntologyManager().getOWLDataFactory()
    try:
        return sorted(iri(i) for node in reasoner.getInstances(java_expr_owl(expr, df), False).getNodes() for i in node.getEntities())
    finally:
        reasoner.dispose()


def test_materialisation_filter_matches_reasoner_on_fixture(tmp_path):
    """ELK membership + asserted edges reproduce HermiT's answers on the TBox-light fixture."""
    from common import start_jvm
    start_jvm(2)
    from materialise import materialise
    _, ont = load_ontology(_ontology(tmp_path))
    mat = materialise(ont, ont)
    assert mat.n == 6 and mat.summary["asserted_distinct_edges"] == 6 and mat.summary["closed_edges"] == 6
    # A ⊑ C is applied by ELK: C = asserted {3,4} plus A's {0,1,2,3}.
    assert mat.answers(P + "C") == [P + f"i{n}" for n in (0, 1, 2, 3, 4)]
    exprs = [P + "A", ["and", P + "A", P + "B"], ["some", P + "r", P + "B"], ["and", P + "A", ["some", P + "r", P + "B"]],
             ["some", P + "r", ["and", P + "A", P + "B"]], ["some", P + "r", ["some", P + "s", P + "A"]],
             ["some", ["inv", P + "r"], P + "A"], ["some", P + "r", P + "C"], ["some", ["inv", P + "s"], P + "C"]]
    for expr in exprs:
        assert mat.answers(expr) == _hermit_answers(ont, expr), expr
    assert mat.answers(["some", P + "r", P + "C"]) == [P + f"i{n}" for n in (0, 1, 2, 5)]
    assert mat.answers(["some", P + "nosuchrole", P + "A"]) == []


def test_materialisation_closes_role_edges(tmp_path):
    """Sub-property, transitive, inverse and symmetric axioms of the full KB close the edges."""
    from common import start_jvm
    start_jvm(2)
    from materialise import materialise
    el_path, full_path = _role_ontologies(tmp_path)
    _, el = load_ontology(el_path); _, full = load_ontology(full_path)
    mat = materialise(el, full)
    pairs = lambda role: {(mat.individuals[a][len(P):], mat.individuals[b][len(P):]) for a, b in zip(*mat.edges[role].nonzero())}  # noqa: E731
    r_edges = {("i0", "i1"), ("i1", "i2"), ("i2", "i4"), ("i5", "i0")}
    assert pairs(P + "r") == r_edges
    assert pairs(P + "rinv") == {(b, a) for a, b in r_edges}
    assert pairs(P + "u") == {("i3", "i4"), ("i4", "i3")}
    chain = ["i5", "i0", "i1", "i2", "i4"]
    assert pairs(P + "t") == {(chain[i], chain[j]) for i in range(5) for j in range(i + 1, 5)}
    used = mat.summary["rbox_axioms_used"]
    assert used["SubObjectPropertyOf"] == 1 and used["InverseObjectProperties"] == 1
    assert used["SymmetricObjectProperty"] == 1 and used["TransitiveObjectProperty"] == 1
    for expr in (["some", P + "t", P + "B"], ["some", P + "rinv", P + "A"], ["some", ["inv", P + "t"], P + "A"],
                 ["some", P + "u", P + "B"], ["some", P + "t", ["some", P + "s", P + "A"]]):
        assert mat.answers(expr) == _hermit_answers(full, expr), expr


def test_sampler_keeps_only_nonempty_nonuniversal_new_sets(tmp_path):
    """Kept candidates satisfy the filter conditions and sampling is deterministic given the seed."""
    from common import start_jvm
    start_jvm(2)
    from materialise import materialise
    from sample_queries import TYPES, sample_from
    _, ont = load_ontology(_ontology(tmp_path))
    mat = materialise(ont, ont)
    first = sample_from(mat, "fixture", seed=3, per_type=4, max_draws=300)
    again = sample_from(mat, "fixture", seed=3, per_type=4, max_draws=300)
    assert first["queries"] == again["queries"] and first["queries"]
    assert first["sampler"] == "materialisation-filter"
    seen = set()
    for q in first["queries"]:
        assert q["type"] in TYPES
        assert 0 < len(q["filter_answers"]) < mat.n
        assert q["filter_answers"] == mat.answers(q["expr"])
        key = (q["type"], tuple(q["filter_answers"]))
        assert key not in seen
        seen.add(key)
    assert len({q["query_id"] for q in first["queries"]}) == len(first["queries"])
    assert sum(s["kept"] for s in first["sampling_stats"].values()) == len(first["queries"])


def test_ground_truth_pipeline_and_budget(tmp_path):
    """sample -> ground truth on the fixture: references match HermiT; zero budget gives nulls."""
    import json
    import shutil
    from common import start_jvm
    start_jvm(2)
    from ground_truth import run
    from sample_queries import sample
    kb_dir = tmp_path / "kbs" / "fixture"; kb_dir.mkdir(parents=True)
    source = _ontology(tmp_path)
    shutil.copy(source, kb_dir / "fixture.owl"); shutil.copy(source, kb_dir / "fixture_el.owl")
    queries = sample("fixture", tmp_path, seed=1, per_type=3, max_draws=200)
    (kb_dir / "queries.json").write_text(json.dumps(queries))
    gt = run("fixture", tmp_path, timeout=30, budget_minutes=5, el_budget_minutes=5)
    _, ont = load_ontology(source)
    assert gt["full_status"] == "ok" and gt["el_status"] == "ok"
    for q in gt["queries"]:
        expected = _hermit_answers(ont, q["expr"])
        assert q["el_answers"] == expected and q["full_answers"] == expected
        assert q["filter_vs_el"]["equal"] and q["filter_vs_full"]["equal"]
        assert q["el_reasoner"] == ("HermiT" if q["type"] == "some_inv_r_A" else "ELK")
        if q["type"] != "some_inv_r_A":
            assert q["elk_vs_hermit"]["equal"] and q["el_hermit_answers"] == expected
    assert gt["summary"]["total"]["full_reference"] == len(queries["queries"])
    per_query = run("fixture", tmp_path, timeout=30, budget_minutes=5, el_budget_minutes=5, hermit_mode="per-query")
    assert [q["full_answers"] for q in per_query["queries"]] == [q["full_answers"] for q in gt["queries"]]
    assert [q["el_answers"] for q in per_query["queries"]] == [q["el_answers"] for q in gt["queries"]]
    starved = run("fixture", tmp_path, timeout=30, budget_minutes=0, el_budget_minutes=5)
    assert all(q["full_answers"] is None and q["full_timeout"] for q in starved["queries"])
    assert all(q["el_answers"] is not None for q in starved["queries"])


def test_watchdog_interrupts_blocking_call():
    """The watchdog thread calls interrupt(); fast calls return normally."""
    import threading
    import time
    from ground_truth import ReasonerTimeout, call_with_watchdog, round_robin

    class FakeReasoner:
        def __init__(self):
            self.flag = threading.Event()

        def interrupt(self):
            self.flag.set()

        def slow(self):
            if self.flag.wait(5):
                raise RuntimeError("interrupted")
            return "finished"

    fake = FakeReasoner()
    started = time.monotonic()
    try:
        call_with_watchdog(fake, fake.slow, 0.2)
        raise AssertionError("watchdog did not fire")
    except ReasonerTimeout:
        pass
    assert time.monotonic() - started < 2
    assert call_with_watchdog(FakeReasoner(), lambda: 42, 5) == 42
    order = round_robin([{"type": t, "i": i} for t, i in (("a", 0), ("a", 1), ("b", 0), ("c", 0), ("b", 1))])
    assert [(q["type"], q["i"]) for q in order] == [("a", 0), ("b", 0), ("c", 0), ("a", 1), ("b", 1)]
