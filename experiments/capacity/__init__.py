"""Capacity (rate-distortion) experiments: exactness, interval budgets, single-box baselines.

Modules
-------
- :mod:`experiments.capacity.common` -- dataset loading, least-model tables, reference entailments.
- :mod:`experiments.capacity.exact` -- B1, exactness table and intervals per class.
- :mod:`experiments.capacity.budget` -- B2, sound lossy compression under an interval budget.
- :mod:`experiments.capacity.baselines` -- B3, single-box baselines at equal parameter count.
- :mod:`experiments.capacity.aggregate` -- B4, ``CAPACITY.md`` and ``capacity_curve.csv``.
"""
