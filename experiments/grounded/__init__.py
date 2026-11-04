"""GroundedEL experiment suite (design §9 E1 synthetic sweep and E2 GO rate-distortion).

Guarantees
----------
- Exposes synthetic theory builders in :mod:`experiments.grounded.theories`,
  the single-cell synthetic runner in :mod:`experiments.grounded.run_synthetic`,
  the Markdown aggregator in :mod:`experiments.grounded.aggregate`, and the GO
  rate-distortion runner in :mod:`experiments.grounded.run_go`.

Non-guarantees
--------------
- Does not import or start the JVM at package import time.
"""
from __future__ import annotations

__all__ = [
    "theories",
    "run_synthetic",
    "aggregate",
    "run_go",
]
