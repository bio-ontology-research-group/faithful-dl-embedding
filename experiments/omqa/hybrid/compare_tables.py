"""Compare two GroundedEL ``tables.pt`` files entry by entry (tensors with torch.equal, everything else with ==).

    python -m experiments.omqa.hybrid.compare_tables A/tables.pt B/tables.pt
"""
from __future__ import annotations

import sys

import torch


def compare(a_path: str, b_path: str) -> int:
    a = torch.load(a_path, weights_only=False)
    b = torch.load(b_path, weights_only=False)
    bad = 0
    if a.keys() != b.keys():
        print(f"key sets differ: only in A {sorted(set(a) - set(b))}, only in B {sorted(set(b) - set(a))}")
        bad += 1
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            same = isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
            desc = f"tensor {tuple(x.shape)} {x.dtype}"
        else:
            same = x == y
            desc = f"{type(x).__name__} (len {len(x)})" if hasattr(x, "__len__") else repr(x)
        print(f"{'same' if same else 'DIFF'}  {k}: {desc}")
        bad += not same
    print("IDENTICAL" if bad == 0 else f"{bad} entries differ")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1], sys.argv[2]))
