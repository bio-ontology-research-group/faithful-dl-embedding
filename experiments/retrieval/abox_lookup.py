#!/usr/bin/env python3
"""Closed-world structural query evaluation over asserted ABox facts.

Library API: ``evaluate(expr, kb_index) -> set[str]``. CLI reads one KB's
query file and emits a query-id to answer-IRI mapping, optionally adding told
named-class hierarchy closure.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from common import DATA_ROOT, NAMES, load_ontology, build_index, eval_structural


def evaluate(expr, kb_index) -> set[str]:
    """Evaluate class expression over asserted types and object-role edges only."""
    return eval_structural(expr,kb_index)


def main():
    """Run asserted-ABox lookup for a KB."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kb",choices=NAMES,required=True);p.add_argument("--data-dir",type=Path,default=DATA_ROOT);p.add_argument("--told-hierarchy",action="store_true");p.add_argument("--output",type=Path);a=p.parse_args()
    d=a.data_dir/"kbs"/a.kb;_,ont=load_ontology(d/f"{a.kb}.owl");index=build_index(ont,a.told_hierarchy);queries=json.loads((d/"queries.json").read_text())["queries"]
    result={q["query_id"]:sorted(evaluate(q["expr"],index)) for q in queries};out=a.output or d/("predictions_told.json" if a.told_hierarchy else "predictions_abox.json");out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(f"wrote {len(result)} predictions to {out}")
if __name__=="__main__":main()
