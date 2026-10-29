# Experiment Results Tables (`experiments/results/tables/`)

This directory contains aggregated Markdown tables produced by `python -m experiments.aggregate` from the per-run JSON outputs in `experiments/results/raw/` (or `experiments/results/smoke/` for the smoke sweep in `experiments/results/smoke/tables/`).

## Generated Files

- `{dataset}.md` (`tk3.md`, `tk4.md`, `tk5.md`, `tk6.md`, `pizza.md`, `go.md`, `foodon.md`):
  Per-dataset summary tables comparing all 11 methods in `config.METHODS` across embedding dimensions `d` (`d ∈ {2, 10}` for `tk3..tk6`; `d = 50` for `pizza`, `go`, `foodon`), reporting `mean ± sd` across seeds and the seed count `n`.
- `multibox_curve.md`:
  Relaxation curve comparing `multibox-k1`, `multibox-k2`, `multibox-k3`, `multibox-k4`, and `multibox-k2-init` across datasets and dimensions.
- `../all_runs.csv` (or `../smoke/all_runs.csv` for the smoke sweep):
  Flat CSV table with one row per run JSON.

## Reproducing the Tables

```bash
# Aggregate smoke-sweep results (epochs = 3, seed = 0)
~/.venvs/mowl-gpu/bin/python -m experiments.aggregate \
    --raw experiments/results/smoke \
    --out experiments/results/smoke/tables

# Aggregate full-sweep results
~/.venvs/mowl-gpu/bin/python -m experiments.aggregate \
    --raw experiments/results/raw \
    --out experiments/results/tables
```

## Notes on Seed Coverage and Protocol

- **Smoke sweep (`results/smoke/`):** Complete for all 11 methods × 7 datasets across all configured dimensions (`121` cells total: `4 × 2 × 11 = 88` synthetic cells + `3 × 1 × 11 = 33` subsumption cells) at `seed = 0`, `epochs = 3` on CUDA (`NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`).
- **Full sweep (`results/raw/`):** Configured in `config.py` with seeds `{0, 1, 2}` for `tk3..tk6` and `pizza` (`epochs = 2000`), and seed `{0}` for `go` and `foodon` (`epochs = 300`, `validate_every = 10`).
- **MILP initialisation (`pizza-box2el-milp`):** Per Spec §5.2, the Helly diagnostic on `pizza` (`experiments/diagnostics/results/pizza.json`) reports `0` non-degenerate certificates (and `incomplete(roles)` status due to role axioms).
