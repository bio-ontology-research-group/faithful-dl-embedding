# Learnable Geometric Embedding Experiments

This directory contains the dataset builders, loaders, diagnostics, and test suite for comparing learnable geometric DL embedding models (`ELBE`, `BoxSquaredEL`, `MultiBoxEL`, with and without exact-model initialisation `ExactBoxInit` and anchor regularisation) on synthetic Helly stress benchmarks and subsumption prediction benchmarks.

## Layout

- `config.py`: Shared dataset identifiers (`DATASETS`), filesystem paths (`~/tmp/gfe-data/`, `~/tmp/gfe-runs/`), and fixed hyperparameter / method configurations (Spec §5.2).
- `datasets.py`: Unified dataset loader `load_dataset(name) -> DatasetBundle` supporting all 7 datasets (`tk3`, `tk4`, `tk5`, `tk6`, `pizza`, `go`, `foodon`) and providing the `PathDataset` + `deductive_closure_ontology` / `evaluation_classes` shim for local subsumption splits.
- `data/`:
  - `make_synthetic.py`: Generates synthetic Helly ontologies `T_k` (`k in {3, 4, 5, 6}`) under `experiments/data/synthetic/tk{k}.owl`.
  - `make_pizza.py`: Builds the `pizza` EL++ subsumption benchmark (`ontology.owl`, `valid.owl`, `test.owl`, `ontology_deductive_closure.owl`) under `~/tmp/gfe-data/pizza-el/` (and mirrors into `experiments/data/pizza-el/` since the total size is `< 1 MB`).
  - `README.md`: Documentation of dataset generation protocols and file layouts.
- `diagnostics/`: Learning-free Helly-satisfiability and certificate-triangle diagnostics (`helly_diagnostic.py`).
- `results/dataset_stats.md`: Summary statistics for all 7 datasets (concept/role counts, NF1–NF7 axiom counts, `⊥`-axioms, certificate triangles, `helly_check` certificates, split sizes, download times, and file sizes).
- `tests/test_datasets.py`: Pytest suite verifying synthetic certificate structure, Pizza-EL split disjointness and deductive closure, and `load_dataset` across all 7 dataset identifiers.

## Usage

All commands use the `mowl-gpu` virtual environment (`~/.venvs/mowl-gpu/bin/python`):

```bash
# Regenerate synthetic T_k ontologies (tk3..tk6)
~/.venvs/mowl-gpu/bin/python -m experiments.data.make_synthetic

# Regenerate Pizza-EL subsumption split
~/.venvs/mowl-gpu/bin/python -m experiments.data.make_pizza

# Run dataset tests
~/.venvs/mowl-gpu/bin/python -m pytest experiments/tests -q
```
