"""Shared experiment configuration: dataset identifiers, filesystem paths, and grid settings.

Guarantees
----------
- Exposes the canonical ordered list ``DATASETS`` of all 7 experiment dataset
  identifiers (`tk3`, `tk4`, `tk5`, `tk6`, `pizza`, `go`, `foodon`).
- Resolves all raw data and checkpoint directories under ``~/tmp/`` (never ``/tmp``)
  and all committed code/results directories inside ``experiments/``.
- Documents the fixed hyperparameter grid and method specifications of Spec §5.2.

Non-guarantees
--------------
- Does not create directories or download files at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = EXPERIMENTS_DIR / "data"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
PIZZA_REPO_DIR: Path = DATA_DIR / "pizza-el"

RESULTS_DIR: Path = EXPERIMENTS_DIR / "results"
RAW_RESULTS_DIR: Path = RESULTS_DIR / "raw"
SMOKE_RESULTS_DIR: Path = RESULTS_DIR / "smoke"
TABLES_DIR: Path = RESULTS_DIR / "tables"
DATASET_STATS_PATH: Path = RESULTS_DIR / "dataset_stats.md"
LOGS_DIR: Path = EXPERIMENTS_DIR / "logs"

RAW_DATA_ROOT: Path = Path("~/tmp/gfe-data").expanduser()
MOWL_DATA_ROOT: Path = RAW_DATA_ROOT / "mowl"
PIZZA_RAW_PATH: Path = RAW_DATA_ROOT / "pizza" / "pizza.owl"
PIZZA_EL_DIR: Path = RAW_DATA_ROOT / "pizza-el"
RUNS_DIR: Path = Path("~/tmp/gfe-runs").expanduser()

SYNTHETIC_NS: str = "http://mowl.borg/geo#"
SYNTHETIC_ONT_BASE: str = "http://mowl.borg/geo"

DATASETS: List[str] = ["tk3", "tk4", "tk5", "tk6", "pizza", "go", "foodon"]
SYNTHETIC_DATASETS: List[str] = ["tk3", "tk4", "tk5", "tk6"]
SUBSUMPTION_DATASETS: List[str] = ["pizza", "go", "foodon"]

DATASET_TASKS: Dict[str, str] = {
    "tk3": "satisfaction",
    "tk4": "satisfaction",
    "tk5": "satisfaction",
    "tk6": "satisfaction",
    "pizza": "subsumption",
    "go": "subsumption",
    "foodon": "subsumption",
}

# Spec §5.2 fixed hyperparameters
DATASET_DIMS: Dict[str, List[int]] = {
    "tk3": [2, 10],
    "tk4": [2, 10],
    "tk5": [2, 10],
    "tk6": [2, 10],
    "pizza": [50],
    "go": [50],
    "foodon": [50],
}

DATASET_EPOCHS: Dict[str, int] = {
    "tk3": 2000,
    "tk4": 2000,
    "tk5": 2000,
    "tk6": 2000,
    "pizza": 2000,
    "go": 300,
    "foodon": 300,
}

DATASET_SEEDS: Dict[str, List[int]] = {
    "tk3": [0, 1, 2],
    "tk4": [0, 1, 2],
    "tk5": [0, 1, 2],
    "tk6": [0, 1, 2],
    "pizza": [0, 1, 2],
    "go": [0, 1, 2],
    "foodon": [0, 1, 2],
}

# Spec §5.2 fixed hyperparameters.
# Note on Box²EL / MultiBoxEL `margin` (gamma): in Box²EL's upstream convention,
# gamma enters inside `relu(... - gamma)`, so a positive gamma RELAXES inclusion
# and disjointness constraints rather than enforcing a strict interior/separation
# margin. We keep the upstream default `margin=0.02` for comparability.
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "margin": 0.02,
    "delta": 2.5,
    "reg_factor": 0.2,
    "num_negs": 4,
    "learning_rate": 0.001,
    "batch_size": 4096 * 8,
    "validate_every": 10,
    "eval_gci_name": "gci0",
}

METHODS: Dict[str, Dict[str, Any]] = {
    "elbe": {
        "model": "ELBE",
        "n_components": 1,
        "init": None,
        "anchor_weight": 0.0,
    },
    "elbe-init": {
        "model": "ELBE",
        "n_components": 1,
        "init": "projected",
        "anchor_weight": 0.0,
    },
    "elbe-anchor": {
        "model": "ELBE",
        "n_components": 1,
        "init": "projected",
        "anchor_weight": 0.1,
    },
    "box2el": {
        "model": "BoxSquaredEL",
        "n_components": 1,
        "init": None,
        "anchor_weight": 0.0,
    },
    "box2el-init": {
        "model": "BoxSquaredEL",
        "n_components": 1,
        "init": "projected",
        "anchor_weight": 0.0,
    },
    "box2el-anchor": {
        "model": "BoxSquaredEL",
        "n_components": 1,
        "init": "projected",
        "anchor_weight": 0.1,
    },
    "box2el-milp": {
        "model": "BoxSquaredEL",
        "n_components": 1,
        "init": "milp",
        "milp_time_limit": 120.0,
        "anchor_weight": 0.0,
        "datasets": ["pizza"],
    },
    "multibox-k1": {
        "model": "MultiBoxEL",
        "n_components": 1,
        "init": None,
        "anchor_weight": 0.0,
    },
    "multibox-k2": {
        "model": "MultiBoxEL",
        "n_components": 2,
        "init": None,
        "anchor_weight": 0.0,
    },
    "multibox-k3": {
        "model": "MultiBoxEL",
        "n_components": 3,
        "init": None,
        "anchor_weight": 0.0,
    },
    "multibox-k4": {
        "model": "MultiBoxEL",
        "n_components": 4,
        "init": None,
        "anchor_weight": 0.0,
    },
    "multibox-k2-init": {
        "model": "MultiBoxEL",
        "n_components": 2,
        "init": "projected",
        "anchor_weight": 0.0,
    },
}


def methods_for_dataset(dataset: str) -> List[str]:
    """Return ordered method identifiers applicable to ``dataset``."""
    return [
        m_id
        for m_id, spec in METHODS.items()
        if spec.get("datasets") is None or dataset in spec["datasets"]
    ]
