"""Check that a ``build_eomqa.py --tbox-json`` build differs from the build without the option only by the dropped
axioms: ``gci0`` / ``gci3`` / ``role_inclusion`` of B are those of A minus the dropped axioms (mapped to table ids
with A's index dicts) plus any added ones, and every other entry is identical.

    python -m experiments.omqa.hybrid.check_tbox_json_tables A/ B/      (directories with tables.pt + build_report.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def rows(t: torch.Tensor) -> set:
    return {tuple(x) for x in t.tolist()}


def check(a_dir: str, b_dir: str) -> int:
    a = torch.load(Path(a_dir) / "tables.pt", weights_only=False)
    b = torch.load(Path(b_dir) / "tables.pt", weights_only=False)
    rep_a = json.loads((Path(a_dir) / "build_report.json").read_text())
    tj = json.loads((Path(b_dir) / "build_report.json").read_text())["tbox_json"]
    cid, rid, top = a["entity_to_class"], a["relation_to_role"], a["top_id"]
    inv = {int(k): v for k, v in rep_a["inverse"].items()}
    bad = 0
    for k in sorted(a):
        if k in ("gci0", "gci3", "role_inclusion"):
            continue
        same = torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k]
        print(f"{'same' if same else 'DIFF'}  {k}")
        bad += not same
    m_ci = lambda ax: {(cid[x], cid[y]) for x, y in ax}
    m_rng = lambda ax: {(rid[inv[r]], top, cid[A]) for r, A in ax}
    m_ri = lambda ax: {(rid[r], rid[s]) for r, s in ax}
    for key, kind, f in (("gci0", "class_inclusion", m_ci), ("gci3", "range_axiom", m_rng), ("role_inclusion", "role_inclusion", m_ri)):
        drop, add = f(tj[kind]["dropped_axioms"]), f(tj[kind]["added_axioms"])
        ra, rb = rows(a[key]), rows(b[key])
        ok = rb == (ra - drop) | add and len(ra - rb) == len(drop) and len(rb) == len(ra) - len(drop) + len(add)
        print(f"{'ok  ' if ok else 'FAIL'}  {key}: {len(ra)} -> {len(rb)} rows; dropped axioms {len(drop)}, added {len(add)}")
        bad += not ok
    print("CHECK PASSED" if bad == 0 else f"CHECK FAILED ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(check(sys.argv[1], sys.argv[2]))
