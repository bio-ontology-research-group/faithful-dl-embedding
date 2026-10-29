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
- **Full sweep (`results/raw/`):** Complete for all `366 / 366` planned cells (`0` missing cells; `3` seeds `{0, 1, 2}` per dataset across all three suites — `3 / 3 / 3` seeds for `synthetic` / `pizza` / `real`):
  - `synthetic` (`tk3`, `tk4`, `tk5`, `tk6`): `4 datasets × 2 dims (2, 10) × 3 seeds (0, 1, 2) × 11 methods = 264` cells at `epochs = 2000` (`1416.75s` sweep wall-clock; per-dataset cell wall-clock totals: `tk3 = 358.96s`, `tk4 = 384.50s`, `tk5 = 347.12s`, `tk6 = 320.84s`; total training time `1409.58s`).
  - `pizza`: `1 dataset × 1 dim (50) × 3 seeds (0, 1, 2) × 12 methods (incl. box2el-milp) = 36` cells at `epochs = 2000`, `validate_every = 10` (`336.34s` sweep wall-clock; cell wall-clock total `334.91s`; total training time `333.44s`).
  - `real` (`go`, `foodon`): `2 datasets × 1 dim (50) × 3 seeds (0, 1, 2) × 11 methods = 66` cells at `epochs = 300`, `validate_every = 10` (`3213.82s` sweep wall-clock across seed-0 and seeds-1–2 passes; per-dataset cell wall-clock totals: `go = 2445.93s` across `33` cells — mean `74.12s`/cell, median `22.79s`/cell, range `16.89s–325.44s`, with `431.88s` total training time (`13.09s` mean, `6.40s` median); `foodon = 713.68s` across `33` cells — mean `21.63s`/cell, median `8.33s`/cell, range `5.71s–93.72s`, with `162.81s` total training time (`4.93s` mean, `3.46s` median)).
  - **Grand totals & OOM / batch-size halvings:** `4966.91s` (`82.8 min`, `1.38 GPU-hours`) total sweep wall-clock (`4905.94s` cell wall-clock sum), `2337.71s` (`39.0 min`) total training time, **`0` OOM retries / batch-size halvings** (`effective_batch_size = 32768` across all `366` runs), and peak GPU memory `1968.13 MB` (`go__multibox-k4__d50__s2.json`).

## Margin conventions

Upstream mOWL embedding models use opposite sign conventions for the positive training margin hyperparameter (`config.margin = 0.02`):

- **Box²EL and MultiBoxEL (`config.margin_semantics = "relaxing"`):** In `BoxSquaredELModule` (and `MultiBoxELModule`, which inherits its per-component GCI losses), the margin parameter $\gamma$ (`module.gamma`) enters positive containment and disjointness losses as $\operatorname{relu}(\dots - \gamma)$ — specifically $\operatorname{relu}(\operatorname{dist}(A, B) + 2\,\text{offset}_A - \gamma)$ for inclusion (`gci0`, `gci1`, `gci2`, `gci3`) and $\operatorname{relu}(-\operatorname{dist}(A, B) - \gamma)$ for disjointness (`gci1_bot`). A positive $\gamma > 0$ therefore **relaxes** containment and disjointness by $\gamma$: any box configuration that satisfies inclusion or disjointness up to $\gamma = 0.02$ coordinate slack already attains zero training loss and receives zero gradient.
- **ELBE (`config.margin_semantics = "tightening"`):** In `ELBEModule`, the margin parameter $m$ (`module.margin`) enters positive containment and disjointness losses as $\operatorname{relu}(\dots + m)$. A positive $m > 0$ therefore **tightens** containment and disjointness, requiring strict interior containment or separation by $m = 0.02$ to reach zero training loss.
- **Pre-registered sweep defaults:** For comparability with upstream mOWL baselines, the pre-registered sweep trained all models with the upstream default `margin = 0.02` (`DEFAULT_HYPERPARAMS["margin"]`). Consequently, the Box²EL/MultiBoxEL family is trained with $0.02$ slack whereas ELBE is trained with $0.02$ strict tightening.
- **What `Modelness` and `Mod@γ` measure:**
  - `Modelness` (`metrics.modelness_all`, per-GCI `metrics.modelness_<gci>`): Fraction of normalised training axioms with loss $\le 10^{-6}$ evaluated at **strict margin $\gamma = 0$** (temporarily zeroing `module.gamma` / `module.margin`). By construction, this penalises the relaxed Box²EL/MultiBoxEL family whenever optimisation halts on the zero-loss boundary of the $\gamma = 0.02$-relaxed objective (where strict $\gamma = 0$ loss is up to $\gamma^2 = 4 \times 10^{-4} > 10^{-6}$, scoring `0`).
  - `Mod@γ` (`metrics.modelness_train_margin`, per-GCI `metrics.modelness_train_margin_<gci>`): Fraction of normalised training axioms with loss $\le 10^{-6}$ evaluated at the **training margin** (`config.margin = 0.02`, without zeroing the module's margin), measuring whether the model satisfies the training objective it was optimised on.
  - **Lower-bound relationship for the Box²EL family:** Because $\operatorname{relu}(x - \gamma) \le \operatorname{relu}(x)$ for all $\gamma \ge 0$, strict `Modelness` ($\gamma = 0$) is a lower bound on `Mod@γ` (`modelness_train_margin >= modelness_all`) for every Box²EL and MultiBoxEL run, and serves as a lower bound on geometric satisfaction up to the $0.02$ training slack. Conversely, for ELBE ($\operatorname{relu}(x + m) \ge \operatorname{relu}(x)$ for $m \ge 0$), `Mod@γ` is a lower bound on strict `Modelness`.

