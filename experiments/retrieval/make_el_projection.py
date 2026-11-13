#!/usr/bin/env python3
"""Create EL++ TBox projections while preserving class/object ABox assertions.

Input is a fetched OWL ontology. Outputs ``<name>_el.owl``, ``inverses.json``
and a drop report beneath ``$GFE_RETRIEVAL_DATA/kbs/<name>``.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from common import DATA_ROOT, NAMES, load_ontology, start_jvm
from kb_stats import TBOX_TYPES


def is_el_tbox(ax) -> bool:
    """Check the supported EL++ TBox fragment, intentionally excluding inverse roles."""
    typ=str(ax.getAxiomType().getName())
    allowed={"SubClassOf","EquivalentClasses","DisjointClasses","ObjectPropertyDomain","ObjectPropertyRange","SubObjectPropertyOf","TransitiveObjectProperty"}
    if typ not in allowed: return False
    exprs=[]
    if typ=="SubClassOf": exprs=[ax.getSubClass(),ax.getSuperClass()]
    elif typ in ("EquivalentClasses","DisjointClasses"): exprs=list(ax.getClassExpressions())
    elif typ=="ObjectPropertyDomain": exprs=[ax.getDomain()]
    elif typ=="ObjectPropertyRange": exprs=[ax.getRange()]
    def ok(e):
        if not e.isAnonymous() or e.isOWLThing() or e.isOWLNothing(): return True
        t=str(e.getClassExpressionType())
        if t=="ObjectIntersectionOf": return all(ok(x) for x in e.asConjunctSet())
        if t=="ObjectSomeValuesFrom": return ok(e.getFiller())
        return False
    return all(ok(e) for e in exprs)


def project(kb: str, root: Path) -> dict:
    """Write the projection and return a precise summary of removed axioms."""
    start_jvm()
    from org.semanticweb.owlapi.model.parameters import Imports
    source=root/"kbs"/kb/f"{kb}.owl"; manager, ont=load_ontology(source)
    removed=Counter(); inverses=[]; to_remove=[]
    for ax in ont.getAxioms(Imports.EXCLUDED):
        typ=str(ax.getAxiomType().getName())
        if typ=="InverseObjectProperties":
            inverses.append({"first": str(ax.getFirstProperty()), "second": str(ax.getSecondProperty()), "axiom": str(ax)})
            to_remove.append(ax); removed["InverseObjectProperties (side-file)"]+=1
        elif typ.startswith("DataProperty") or typ in {"FunctionalDataProperty","SubDataPropertyOf","EquivalentDataProperties","DisjointDataProperties","DatatypeDefinition"} or (typ=="Declaration" and ax.getEntity().isOWLDataProperty()):
            to_remove.append(ax); removed["data_property_axiom"]+=1
        elif typ in TBOX_TYPES:
            if not is_el_tbox(ax): to_remove.append(ax); removed[typ]+=1
    # OWLAPI set-based remove is stable; it does not touch the source file.
    from java.util import HashSet
    removals=HashSet()
    for ax in to_remove: removals.add(ax)
    manager.removeAxioms(ont, removals)
    target=source.parent/f"{kb}_el.owl"; manager.saveOntology(ont, __import__("org.semanticweb.owlapi.model",fromlist=["IRI"]).IRI.create(target.resolve().as_uri()))
    inv=source.parent/"inverses.json"; inv.write_text(json.dumps(inverses,indent=2)+"\n")
    result={"source":str(source),"projection":str(target),"dropped_axioms":dict(sorted(removed.items())),"dropped_total":sum(removed.values()),"inverse_axioms_in_side_file":len(inverses),"projection_axiom_count":len(ont.getAxioms(Imports.EXCLUDED))}
    (source.parent/"projection_report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    """Run projection CLI."""
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--kb",choices=NAMES,required=True); p.add_argument("--data-dir",type=Path,default=DATA_ROOT); a=p.parse_args()
    try: print(json.dumps(project(a.kb,a.data_dir),indent=2))
    except FileNotFoundError as e: p.error(str(e))
if __name__=="__main__": main()
