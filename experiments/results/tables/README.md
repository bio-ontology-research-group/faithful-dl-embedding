# Experiment Results Tables (`experiments/results/tables/`)

This directory contains aggregated Markdown tables produced by `python -m experiments.aggregate` from the per-run JSON outputs in `experiments/results/raw/` (or `experiments/results/smoke/` for the smoke sweep in `experiments/results/smoke/tables/`).

## Generated Files

- `{dataset}.md` (`tk3.md`, `tk4.md`, `tk5.md`, `tk6.md`, `pizza.md`, `go.md`, `foodon.md`):
  Per-dataset summary tables comparing all configured methods in `config.METHODS` (`11` methods on `tk3..tk6`, `go`, `foodon`; `12` methods including `box2el-milp` on `pizza`) across embedding dimensions `d` (`d ∈ {2, 10}` for `tk3..tk6`; `d = 50` for `pizza`, `go`, `foodon`), reporting `mean ± sd` across seeds and the seed count `n` as `k/n` (non-null over total). Both satisfaction and subsumption tables include the `Certs` (`n_certificates`) column.
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

## Notes on Seed Coverage, Aggregation, and Protocol

- **Aggregation, `k/n` reporting, and `sd` (`ddof = 1`):** `aggregate.py` never silently drops `None` or `NaN` values. The `n` column reports `"k/n"` (non-null runs `k` over total runs `n`), any metric cell with `0 < k < n` appends `(k/n)`, and any metric cell where all runs are null (`k == 0`) is rendered as `"—"`. When `k >= 2`, the standard deviation `sd` is the sample standard deviation computed with **`ddof = 1`** (`np.std(..., ddof=1)`).
- **Helly metrics when `n_certificates == 0`:** When `n_certificates == 0` (as on `pizza`, `go`, and `foodon`), `helly_violation_rate` (`HVR`) and `degeneracy_rate` (`DegRate`) are `null` (`None` in JSON, rendered as `"—"` in Markdown tables), and `Certs` (`n_certificates`) is included in both satisfaction and subsumption tables.
- **Filtered ranking protocol (deliberate deviation from upstream `BaseRankingEvaluator`):** First, `ontology_deductive_closure.owl` for `pizza` is the deductive closure of the *training* ontology `ontology.owl` (matching mOWL's `go` and `foodon` datasets; `224` non-trivial atomic subsumptions, containing `0 / 16` held-out validation and `0 / 16` held-out test axioms). Second, in `_compute_fast_head_centric_ranking_metrics` (`experiments/metrics.py`), before multiplying candidate tail scores by `filtering_labels[head]`, the filter label of the test tuple `(head, tail)` being ranked is reset to `1.0` on a cloned row copy (`batch_filter[row, tail] = 1.0`). This is a deliberate deviation from upstream `BaseRankingEvaluator.compute_ranking_metrics` (which multiplies by `self.filtering_labels[head]` without un-masking the target tail itself) so that standard filtered ranking removes only *other* known positives and never masks the target test tuple being ranked. Unfiltered ranking metrics (`mr`, `mrr`, `auc`, `hits@k`) are kept alongside the filtered metrics.
- **MILP initialisation (`box2el-milp` on `pizza`):** Per Spec §5.2, the Helly diagnostic on `pizza` (`experiments/diagnostics/results/pizza.json`) reports `0` non-degenerate certificates, so `box2el-milp` (`BoxSquaredEL`, `init="milp"`, `milp_time_limit=120.0`, `anchor_weight=0.0`) is included in the `pizza` suite (`12` methods total for `pizza`). When `ExactBoxInit` falls back to `"projected"` (because `pizza` contains role axioms `NF3`/`NF4`/`NF6`/`NF7` outside `BoxModelBuilder`'s propositional `NF1`/`NF2` fragment), `config.init_report.fallback` records the fallback reason in the output JSON.
- **OOM handling and forward chunking (`max_oom_retries = 3`):** On `torch.cuda.OutOfMemoryError`, `experiments/run.py` halves `effective_batch_size` and retries up to `3` times (`max_oom_retries = 3`). When `effective_batch_size` is halved below the full-batch default, `_chunked_module_forward` slices 2-D GCI inputs inside `module.forward` into chunks of `effective_batch_size` rows. Because upstream `EmbeddingELModel.train` calls `backward()` once on the concatenated full-batch GCI loss without gradient checkpointing, forward chunking bounds only peak *non-saved* intermediate tensor allocations inside each chunk's forward pass, not autograd saved activations across chunks. No OOM occurred during the smoke sweep (maximum GPU memory across all cells was `3.7 GB`).
- **Smoke sweep (`results/smoke/`):** Complete for all 7 datasets across all configured dimensions (`122` cells total: `4 × 2 × 11 = 88` synthetic cells + `1 × 1 × 12 = 12` `pizza` cells + `2 × 1 × 11 = 22` `go`/`foodon` cells) at `seed = 0`, `epochs = 3` on CUDA (`NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`).
- **Full sweep (`results/raw/`):** Configured in `config.py` with seeds `{0, 1, 2}` for `tk3..tk6` (`11` methods) and `pizza` (`12` methods, `epochs = 2000`), and seed `{0}` for `go` and `foodon` (`11` methods, `epochs = 300`, `validate_every = 10`).
