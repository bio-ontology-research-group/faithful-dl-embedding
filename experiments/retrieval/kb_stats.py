#!/usr/bin/env python3
"""Count OWL signature and axioms, classifying TBox axioms by EL++ support.

Input: fetched OWL documents. Outputs: ``stats.json`` and ``STATS.md`` under
``$GFE_RETRIEVAL_DATA``; ontology parsing is performed only through OWLAPI.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from common import DATA_ROOT, NAMES, load_ontology, start_jvm

PAPER = {"carcinogenesis": (22372,142,4,144,74223), "mutagenesis": (14145,86,5,82,47722), "semantic_bible": (724,48,29,56,3106), "vicodi": (33238,194,10,204,116181), "family": (202,18,4,26,472), "father": (6,4,1,3,4)}
TBOX_TYPES={"SubClassOf","EquivalentClasses","DisjointClasses","DisjointUnion","SubObjectPropertyOf","EquivalentObjectProperties","DisjointObjectProperties","ObjectPropertyDomain","ObjectPropertyRange","InverseObjectProperties","FunctionalObjectProperty","InverseFunctionalObjectProperty","SymmetricObjectProperty","AsymmetricObjectProperty","TransitiveObjectProperty","ReflexiveObjectProperty","IrreflexiveObjectProperty","SubDataPropertyOf","EquivalentDataProperties","DisjointDataProperties","DataPropertyDomain","DataPropertyRange","FunctionalDataProperty","DatatypeDefinition"}
ABOX_PREFIXES=("ClassAssertion","ObjectPropertyAssertion","NegativeObjectPropertyAssertion","DataPropertyAssertion","NegativeDataPropertyAssertion","SameIndividual","DifferentIndividuals")


def _expr_ok(expr) -> bool:
    """Check named/⊤/⊥, intersection and existential EL class expressions."""
    if not expr.isAnonymous() or expr.isOWLThing() or expr.isOWLNothing(): return True
    kind=str(expr.getClassExpressionType())
    if kind=="ObjectIntersectionOf": return all(_expr_ok(x) for x in expr.asConjunctSet())
    if kind=="ObjectSomeValuesFrom": return _expr_ok(expr.getFiller())
    return False


def _constructors(expr, counts: Counter) -> None:
    """Count non-EL class expression constructors recursively."""
    if not expr.isAnonymous() or expr.isOWLThing() or expr.isOWLNothing(): return
    kind=str(expr.getClassExpressionType())
    if kind=="ObjectIntersectionOf": children=list(expr.asConjunctSet())
    elif kind in ("ObjectUnionOf","ObjectOneOf"): children=list(expr.asDisjunctSet()) if kind=="ObjectUnionOf" else []
    elif kind in ("ObjectSomeValuesFrom","ObjectAllValuesFrom"): children=[expr.getFiller()]
    elif kind in ("ObjectComplementOf",): children=[expr.getOperand()]
    elif "Cardinality" in kind: children=[expr.getFiller()]
    else: children=[]
    counts[kind]+=1
    for child in children: _constructors(child,counts)


def inspect(path: Path) -> dict:
    """Return signature, axiom-type, fragment and paper-comparison counts."""
    start_jvm()
    from org.semanticweb.owlapi.model.parameters import Imports
    _,ont=load_ontology(path); axs=list(ont.getAxioms(Imports.EXCLUDED))
    tb=[ax for ax in axs if str(ax.getAxiomType().getName()) in TBOX_TYPES]
    ab=[ax for ax in axs if str(ax.getAxiomType().getName()).startswith(ABOX_PREFIXES)]
    tbox_types=Counter(str(ax.getAxiomType().getName()) for ax in tb)
    abox_types=Counter(str(ax.getAxiomType().getName()) for ax in ab)
    inside=[]; outside=[]; offenders=Counter()
    for ax in tb:
        typ=str(ax.getAxiomType().getName()); exprs=[]
        if typ=="SubClassOf": exprs=[ax.getSubClass(),ax.getSuperClass()]
        elif typ in ("EquivalentClasses","DisjointClasses","DisjointUnion"): exprs=list(ax.getClassExpressions())
        elif typ=="ObjectPropertyDomain": exprs=[ax.getDomain()]
        elif typ=="ObjectPropertyRange": exprs=[ax.getRange()]
        ok=typ in {"SubClassOf","EquivalentClasses","DisjointClasses","ObjectPropertyDomain","ObjectPropertyRange","SubObjectPropertyOf","TransitiveObjectProperty"} and all(_expr_ok(e) for e in exprs)
        (inside if ok else outside).append(ax)
        if not ok:
            offenders[typ]+=1
            for e in exprs:
                if not _expr_ok(e): _constructors(e,offenders)
            if typ in {"SubObjectPropertyOf","TransitiveObjectProperty","ObjectPropertyDomain","ObjectPropertyRange"}:
                prop=ax.getProperty() if hasattr(ax,"getProperty") else None
                if prop is not None and prop.isAnonymous(): offenders["InverseObjectPropertyUse"]+=1
    fields=("individuals","named_classes","object_properties","tbox_axioms","abox_assertions")
    values=(len(ont.getIndividualsInSignature()),len(ont.getClassesInSignature()),len(ont.getObjectPropertiesInSignature()),len(tb),len(ab))
    return {"path":str(path),"individuals":values[0],"named_classes":values[1],"object_properties":values[2],"data_properties":len(ont.getDataPropertiesInSignature()),"tbox_axioms":len(tb),"tbox_axioms_by_type":dict(sorted(tbox_types.items())),"abox_assertions":len(ab),"abox_assertions_by_type":dict(sorted(abox_types.items())),"el_plus_plus_tbox_axioms":len(inside),"non_el_plus_plus_tbox_axioms":len(outside),"non_el_offenders":dict(sorted(offenders.items())),"inverse_property_axioms":tbox_types.get("InverseObjectProperties",0),"paper_table3":dict(zip(fields,PAPER[path.parent.name])),"differences_from_paper":dict(zip(fields,[v-p for v,p in zip(values,PAPER[path.parent.name])]))}


def main() -> None:
    """Run stats for one KB or all fetched KBs."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kb",choices=NAMES);p.add_argument("--data-dir",type=Path,default=DATA_ROOT);a=p.parse_args()
    names=[a.kb] if a.kb else NAMES;results={}
    for name in names:
        f=a.data_dir/"kbs"/name/f"{name}.owl"
        if not f.exists(): results[name]={"status":"not_found","path":str(f)};print(f"{name}: missing");continue
        print(f"Inspecting {name}...");results[name]=inspect(f);r=results[name];print(f"{name}: individuals={r['individuals']} classes={r['named_classes']} TBox={r['tbox_axioms']} ABox={r['abox_assertions']}")
    (a.data_dir/"stats.json").write_text(json.dumps(results,indent=2)+"\n")
    rows=["| KB | Individuals | Classes | Obj. props | TBox | ABox | EL++ TBox | Non-EL++ | Δ vs Ebr (I/C/P/T/A) |","|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for n,r in results.items():
        if r.get("status")=="not_found": rows.append(f"| {n} | — | — | — | — | — | — | — | not fetched |")
        else:
            d=r["differences_from_paper"];delta="/".join(str(d[k]) for k in ("individuals","named_classes","object_properties","tbox_axioms","abox_assertions"));rows.append(f"| {n} | {r['individuals']} | {r['named_classes']} | {r['object_properties']} | {r['tbox_axioms']} | {r['abox_assertions']} | {r['el_plus_plus_tbox_axioms']} | {r['non_el_plus_plus_tbox_axioms']} | {delta} |")
    (a.data_dir/"STATS.md").write_text("# Retrieval KB statistics\n\nCounts use OWLAPI over fetched documents (imports excluded). Ebr Table 3 is the comparison reference; differences are local count minus paper count.\n\n"+"\n".join(rows)+"\n")

if __name__=="__main__":main()
