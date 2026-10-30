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
YEAST_FUNCTION_DIR: Path = RAW_DATA_ROOT / "yeast-function"
HELIS_DIR: Path = RAW_DATA_ROOT / "helis"
RUNS_DIR: Path = Path("~/tmp/gfe-runs").expanduser()

SYNTHETIC_NS: str = "http://mowl.borg/geo#"
SYNTHETIC_ONT_BASE: str = "http://mowl.borg/geo"

DATASETS: List[str] = [
    "tk3",
    "tk4",
    "tk5",
    "tk6",
    "pizza",
    "go",
    "foodon",
    "yeast-function",
    "helis",
]
SYNTHETIC_DATASETS: List[str] = ["tk3", "tk4", "tk5", "tk6"]
SUBSUMPTION_DATASETS: List[str] = ["pizza", "go", "foodon"]
ABOX_DATASETS: List[str] = ["yeast-function", "helis"]
OBJECTIVE_DATASETS: List[str] = list(SYNTHETIC_DATASETS)
LR_GRID: List[float] = [0.001, 0.01]

DATASET_TASKS: Dict[str, str] = {
    "tk3": "satisfaction",
    "tk4": "satisfaction",
    "tk5": "satisfaction",
    "tk6": "satisfaction",
    "pizza": "subsumption",
    "go": "subsumption",
    "foodon": "subsumption",
    "yeast-function": "abox",
    "helis": "abox",
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
    "yeast-function": [50],
    "helis": [50],
}

DATASET_EPOCHS: Dict[str, int] = {
    "tk3": 2000,
    "tk4": 2000,
    "tk5": 2000,
    "tk6": 2000,
    "pizza": 2000,
    "go": 300,
    "foodon": 300,
    "yeast-function": 2000,
    "helis": 2000,
}

DATASET_SEEDS: Dict[str, List[int]] = {
    "tk3": [0, 1, 2],
    "tk4": [0, 1, 2],
    "tk5": [0, 1, 2],
    "tk6": [0, 1, 2],
    "pizza": [0, 1, 2],
    "go": [0, 1, 2],
    "foodon": [0, 1, 2],
    "yeast-function": [0, 1, 2],
    "helis": [0, 1, 2],
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
    "canonical-mixture": {
        "model": "CanonicalMixture",
        "n_components": 1,
        "init": None,
        "anchor_weight": 0.0,
        "datasets": ["yeast-function", "helis"],
    },
}

_OBJECTIVE_ALLOWED_DATASETS: List[str] = list(OBJECTIVE_DATASETS) + ["pizza"] + list(ABOX_DATASETS)

_BACKBONE_MODEL_MAP: Dict[str, str] = {
    "multibox": "MultiBoxEL",
    "multielbe": "MultiELBE",
}

_OBJECTIVE_KWARGS_MAP: Dict[str, Dict[str, Any]] = {
    "axiom": {},
    "soft": {"component_aggregation": "soft", "tau_anneal": True},
    "softdiv": {"component_aggregation": "soft", "tau_anneal": True, "diversity_reg": 0.1},
    "point": {"objective": "point"},
    "mixed": {"objective": "mixed"},
    "hybrid": {"objective": "hybrid"},
}

OBJECTIVE_METHODS: List[str] = []
for _backbone, _model_cls in _BACKBONE_MODEL_MAP.items():
    for _k in (1, 2, 3):
        for _obj_var, _obj_kw in _OBJECTIVE_KWARGS_MAP.items():
            _m_id = f"{_backbone}-k{_k}-{_obj_var}"
            OBJECTIVE_METHODS.append(_m_id)
            METHODS[_m_id] = {
                "model": _model_cls,
                "backbone": _backbone,
                "n_components": _k,
                "init": None,
                "anchor_weight": 0.0,
                "objective": _obj_kw.get("objective", "axiom"),
                "objective_variant": _obj_var,
                "component_aggregation": _obj_kw.get("component_aggregation", "hard"),
                "tau_anneal": bool(_obj_kw.get("tau_anneal", False)),
                "diversity_reg": float(_obj_kw.get("diversity_reg", 0.0)),
                "objective_kwargs": dict(_obj_kw),
                "datasets": list(_OBJECTIVE_ALLOWED_DATASETS),
            }

PRISM_METHODS: List[str] = ["prism", "prism-init"]
METHODS["prism"] = {
    "model": "PrismEL",
    "backbone": "prism",
    "n_components": 1,
    "init": None,
    "anchor_weight": 0.0,
    "datasets": list(_OBJECTIVE_ALLOWED_DATASETS),
}
METHODS["prism-init"] = {
    "model": "PrismEL",
    "backbone": "prism",
    "n_components": 1,
    "init": "projected",
    "anchor_weight": 0.0,
    "datasets": list(_OBJECTIVE_ALLOWED_DATASETS),
}


def methods_for_dataset(dataset: str) -> List[str]:
    """Return ordered method identifiers applicable to ``dataset``."""
    return [
        m_id
        for m_id, spec in METHODS.items()
        if spec.get("datasets") is None or dataset in spec["datasets"]
    ]
