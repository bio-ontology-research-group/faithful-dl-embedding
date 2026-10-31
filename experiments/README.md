# Learnable Geometric Embedding Experiments

This directory contains the dataset builders, loaders, evaluation harnesses, sweep orchestrator, diagnostics, and test suite for comparing learnable geometric DL embedding models across synthetic Helly stress benchmarks (`satisfaction`), TBox subsumption prediction benchmarks (`subsumption`), and ABox assertion prediction benchmarks (`abox`).

## Datasets (9)

Configured in `config.DATASETS` and loaded via `datasets.load_dataset(name) -> DatasetBundle`:

- **Synthetic Helly suite (`task = "satisfaction"`, `d ∈ {2, 10}`, `epochs = 2000`, `seeds = {0, 1, 2}`):**
  `tk3`, `tk4`, `tk5`, `tk6` (`T_k` for `k ∈ {3, 4, 5, 6}` under `experiments/data/synthetic/tk{k}.owl`).
- **TBox subsumption benchmarks (`task = "subsumption"`, `d = 50`, `seeds = {0, 1, 2}`):**
  - `pizza`: Pizza-EL subsumption split (`epochs = 2000`, committed mirror in `experiments/data/pizza-el/`).
  - `go`: Gene Ontology subsumption benchmark (`epochs = 300`, `mowl.datasets.builtin.GOSubsumptionDataset`).
  - `foodon`: FoodOn subsumption benchmark (`epochs = 300`, `mowl.datasets.builtin.FoodOnSubsumptionDataset`).
- **ABox assertion prediction benchmarks (`task = "abox"`, `d = 50`, `epochs = 2000`, `seeds = {0, 1, 2}`):**
  - `yeast-function` (B1): Yeast protein function prediction (`Aux_G(p)` with `Aux_G ≡ ∃has_function.G`) over the normalised GO EL++ TBox and training protein-protein interactions `interacts_with(p, q)`, built from mOWL's `PPIYeastDataset`.
  - `helis` (B3): HeLiS food/nutrient class membership benchmark following the OWL2Vec\* case-study protocol (`KRR-Oxford/OWL2Vec-Star`), with sound `ObjectPropertyRange` materialisation and both KB-entailed (`label`) and gold-conditional (`label_gold`) three-way evaluation labels.

## Suites (6)

Configured in `sweep.SUITE_DATASETS`:

- `synthetic`: `["tk3", "tk4", "tk5", "tk6"]` (baseline methods).
- `pizza`: `["pizza"]` (baseline methods including `box2el-milp`).
- `real`: `["go", "foodon"]` (baseline methods).
- `objectives`: `["tk3", "tk4", "tk5", "tk6"]` across `lr ∈ {0.001, 0.01}` for the 36 multi-component objective methods plus `prism` and `prism-init`.
- `abox`: `["yeast-function", "helis"]` (baseline + selected objective/prism methods evaluated on ABox prediction).
- `all`: all 9 datasets in `DATASETS`.

## Method Families (52 Method IDs)

Configured in `config.METHODS`:

- **Single-box baselines (`ELBE`, `BoxSquaredEL`):**
  `elbe`, `elbe-init`, `elbe-anchor`, `box2el`, `box2el-init`, `box2el-anchor`, and `box2el-milp` (`pizza` only, MILP initialisation with 120 s limit).
- **Multi-box relaxation baselines (`MultiBoxEL`):**
  `multibox-k1`, `multibox-k2`, `multibox-k3`, `multibox-k4`, and `multibox-k2-init` (projected `ExactBoxInit` initialisation).
- **Ball baseline (`ELEm`):**
  `elem` (`pizza`, `go`, `foodon`, `yeast-function`, `helis`).
- **Canonical-model mixture (`CanonicalMixture`):**
  `canonical-mixture` (`yeast-function`, `helis`).
- **Multi-component objective grid (`MultiBoxEL` and `MultiELBE`, 36 method IDs):**
  `{multibox,multielbe}-k{1,2,3}-{axiom,soft,softdiv,point,mixed,hybrid}`, combining backbone (`multibox` vs `multielbe`), component count `k ∈ {1, 2, 3}`, and training objective (`axiom` hard-min region loss, `soft` temperature-annealed softmin aggregation, `softdiv` softmin + diversity regularisation, `point` canonical-element point membership loss, `mixed` point + region loss, `hybrid` region + point loss).
- **Triangular-prism embeddings (`PrismEL`):**
  `prism` (random init) and `prism-init` (`projected` init; requires `d ≥ 3`, so nominal `d = 2` cells run at effective `d = 3`).

## Layout

- `config.py`: Shared dataset identifiers (`DATASETS`), suite lists, filesystem paths (`<data-dir>/gfe-data/`, `<data-dir>/gfe-runs/`, defaulting to `~/tmp/gfe-data/` and `~/tmp/gfe-runs/`), and method/hyperparameter configurations (`METHODS`, `OBJECTIVE_METHODS`, `PRISM_METHODS`).
- `datasets.py`: Unified dataset loader `load_dataset(name) -> DatasetBundle` supporting all 9 datasets (`tk3`–`tk6`, `pizza`, `go`, `foodon`, `yeast-function`, `helis`).
- `run.py` & `sweep.py`: Single-cell runner and sequential sweep orchestrator (`--suite {synthetic,objectives,pizza,real,abox,all}`).
- `metrics.py`, `abox_eval.py`, & `posthoc.py`: TBox/ABox model-ness, point model-ness, Helly violation/degeneracy diagnostics, filtered subsumption and ABox candidate ranking, protein-centric $F_{\max}$, hierarchy violation rate, and three-way faithfulness (`KB` and `gold`).
- `aggregate.py`: Aggregates raw JSON outputs in `results/raw/` into Markdown tables in `results/tables/` and `results/all_runs.csv`.
- `data/`:
  - `make_synthetic.py`: Generates synthetic Helly ontologies `T_k` (`k ∈ {3, 4, 5, 6}`) under `experiments/data/synthetic/tk{k}.owl`.
  - `make_pizza.py`: Builds the `pizza` EL++ subsumption benchmark under `<data-dir>/gfe-data/pizza-el/` (mirrored in `experiments/data/pizza-el/`).
  - `make_yeast_function.py`: Builds the B1 `yeast-function` ABox benchmark under `<data-dir>/gfe-data/yeast-function/` (manifest mirrored in `experiments/data/yeast-function/manifest.json`).
  - `fetch_helis.py` & `make_helis.py`: Downloads/verifies the OWL2Vec\* HeLiS tarballs and builds the B3 `helis` ABox benchmark under `<data-dir>/gfe-data/helis/` (manifest mirrored in `experiments/data/helis/manifest.json`).
  - `README.md`: Documentation of dataset generation protocols, input SHA-256 digests, and file layouts.
- `diagnostics/`: Learning-free Helly-satisfiability and certificate-triangle diagnostics (`helly_diagnostic.py`).
- `results/`:
  - [`OBJECTIVES.md`](results/OBJECTIVES.md): Synthetic objective sweep (`tk3`–`tk6`), `prism`/`prism-init` evaluation, and Pizza sanity check verdict.
  - [`ABOX.md`](results/ABOX.md): B1 `yeast-function` and B3 `helis` ABox benchmark results and diagnostics.
  - [`dataset_stats.md`](results/dataset_stats.md): Summary statistics for all 9 datasets (including Sections 4 and 5 for B1 `yeast-function` and B3 `helis`).
  - [`tables/`](results/tables/README.md): Per-dataset Markdown tables (`{dataset}.md`, `objectives_{dataset}.md`, `multibox_curve.md`).
- `tests/`: Pytest suite covering dataset builders, loaders, metrics, ABox evaluation, objectives configuration, post-hoc analysis, and sweep orchestration.

## Usage

All commands run with the project Python environment (with `mowl` installed from `bio-ontology-research-group/mowl` branch `abox-geometric`, PR #172):

```bash
# Regenerate synthetic T_k ontologies (tk3..tk6)
python -m experiments.data.make_synthetic

# Regenerate Pizza-EL subsumption split
python -m experiments.data.make_pizza

# Build B1 yeast-function ABox benchmark (after fetching PPIYeastDataset via mOWL)
python -m experiments.data.make_yeast_function \
    --input <data-dir>/gfe-data/yeast/ppi_yeast/ontology.owl \
    --out <data-dir>/gfe-data/yeast-function \
    --seed 0

# Fetch/verify OWL2Vec* HeLiS tarballs and build B3 helis ABox benchmark
python -m experiments.data.fetch_helis --dl-dir <data-dir>/gfe-data/helis-dl
python -m experiments.data.make_helis \
    --dl-dir <data-dir>/gfe-data/helis-dl \
    --out <data-dir>/gfe-data/helis

# Dry-run or execute experiment sweeps
python -m experiments.sweep --suite {synthetic,objectives,pizza,real,abox,all} --dry-run

# Aggregate raw JSON runs into Markdown tables
python -m experiments.aggregate --raw experiments/results/raw --out experiments/results/tables

# Run fast test suite
python -m pytest experiments/tests -q -m "not slow"
```
