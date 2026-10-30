"""Single-cell experiment runner (Task 8, Spec §5.5 & §7).

CLI usage
---------
::

    python -m experiments.run --dataset D --method M --dim d --seed s [--epochs E] [--smoke] [--out results/raw/]

Guarantees
----------
- Writes ``<out>/{dataset}__{method}__d{dim}__s{seed}.json`` atomically via a
  temporary file and ``os.replace`` only after all metrics succeed; if any metric
  raises, no JSON file is written and the runner exits non-zero.
- Seeds ``random.seed(seed)``, ``np.random.seed(seed)``, and ``torch.manual_seed(seed)``
  (plus ``torch.cuda.manual_seed_all(seed)``) before model construction and before each
  training attempt.
- Uses cached :class:`~mowl.geometric.CanonicalModel` via
  :func:`~experiments.cm_cache.get_canonical_model` for both ``ExactBoxInit``
  (passing ``model.class_index_dict`` so auxiliary normaliser classes are
  initialised) and evaluation metrics.
- For subsumption datasets (``pizza``, ``go``, ``foodon``), sets
  ``model.eval_gci_name = "gci0"``, validates every ``validate_every`` epochs,
  saves the best-validation state dict under ``~/tmp/gfe-runs/``, and reloads it
  via ``model.load_best_model()`` before evaluation.
- For satisfaction datasets (``tk3``..``tk6``), trains for the fixed number of
  epochs without validation and evaluates the final parameters.
- For ABox datasets (``yeast-function``, ``helis``, ``task == "abox"``), disables
  mOWL's loss-based validation checkpointing (``validate_every = epochs + 1``) and
  runs a 50-epoch validation-MRR callback (plus the final epoch) via
  :func:`~experiments.abox_eval.compute_validation_mrr` without early stopping,
  saving the state dict with the highest validation rankable-only MRR under
  ``~/tmp/gfe-runs/`` and reloading it via ``load_state_dict`` before test
  evaluation. Note that ``timings["total_train_time_s"]`` includes validation-MRR
  callback time whereas ``timings["mean_epoch_time_s"]`` measures per-epoch
  training steps only.
- Handles ``torch.cuda.OutOfMemoryError`` per Spec §7: halves ``batch_size`` and
  retries up to 3 times (``max_oom_retries = 3``), recording
  ``config["effective_batch_size"]`` and ``config["oom_retries"]``. When
  ``effective_batch_size`` has been halved below the full-batch default, chunks
  GCI scoring inside the module's ``forward`` call into slices of
  ``effective_batch_size`` rows. Note that because upstream
  ``EmbeddingELModel.train`` calls ``backward()`` once on the concatenated
  full-batch GCI loss without gradient checkpointing, forward chunking bounds
  only peak non-saved intermediate tensor allocations inside each chunk's
  forward pass, not autograd saved activations across chunks (no OOM occurred in
  the smoke sweep; max GPU memory was 3.7 GB).

ABox JSON Schema (``task == "abox"``)
-------------------------------------
Every output JSON has four top-level keys (``config``, ``metrics``, ``timings``,
``env``). For ``task == "abox"`` runs:

- ``config``:
  ``dataset``, ``task`` (``"abox"``), ``method``, ``model``, ``n_components``,
  ``init``, ``anchor_weight``, ``dim``, ``seed``, ``epochs``, ``validate_every``
  (``50``), ``eval_gci_name`` (``"class_assertion"``), ``margin``, ``delta``,
  ``reg_factor``, ``num_negs``, ``learning_rate``, ``batch_size``,
  ``effective_batch_size``, ``oom_retries``, ``device``, ``checkpoint_path``,
  ``init_report``.
- ``metrics``:
  - Filtered candidate ranking (rankable-only): ``rank_mrr``, ``rank_hits@1``,
    ``rank_hits@5``, ``rank_hits@10``, ``rank_auc``, and nested dict
    ``ranking_rankable`` (``{"mrr", "hits@1", "hits@5", "hits@10", "auc"}``).
  - Filtered candidate ranking (all-positives, where unrankable positives outside
    ``candidates.txt`` receive ``rank = n_candidates + 1``): ``rank_all_mrr``,
    ``rank_all_hits@1``, ``rank_all_hits@5``, ``rank_all_hits@10``,
    ``rank_all_auc``, and nested dict ``ranking_all``
    (``{"mrr", "hits@1", "hits@5", "hits@10", "auc"}``).
  - Ranking coverage accounting: ``rank_coverage``, ``n_positives_total``,
    ``n_positives_rankable``, ``n_positives_unrankable``.
  - CAFA-style protein-centric :math:`F_{\\max}`: ``fmax`` (float on
    ``yeast-function`` with KB-entailed cells excluded from both prediction and
    truth sets; ``null`` on ``helis``).
  - Hierarchy consistency: ``hierarchy_violation_rate`` (fraction of up to 10,000
    sampled ``(a, G ⊑ G')`` pairs with ``G != G'`` where
    ``score(a, G) < score(a, G') - 1e-6``, or ``null`` below 50 pairs).
  - ABox and TBox modelness: ``abox_modelness`` (fraction of training class and
    supported role assertions with strict ``margin=0`` loss ``<= 1e-6``, or
    ``null`` if none scored), plus ``modelness_all``, ``modelness_<gci>``,
    ``modelness_train_margin``, ``modelness_train_margin_<gci>``,
    ``margin_semantics``.
  - Three-way faithfulness on KB ``label``: nested dict ``faithfulness_kb``
    (``{"auc_pos_neg", "auc_pos_und", "pair_fvr", "fvr", "n_pos", "n_und", "n_neg"}``)
    and top-level keys ``auc_pos_neg``, ``auc_pos_und``, ``pair_fvr``, ``fvr``,
    ``n_pos``, ``n_und``, ``n_neg``. ``auc_pos_und`` (``AUC(+/?)``) is pooled
    across evaluation individuals.
  - Three-way faithfulness on ``label_gold`` (``helis`` only; ``null`` on
    ``yeast-function``): nested dict ``faithfulness_gold`` (same 7 keys or
    ``null``) and top-level keys ``auc_pos_neg_gold``, ``auc_pos_und_gold``,
    ``pair_fvr_gold``, ``fvr_gold``.
  - Missing entity counts: ``n_missing_individuals``, ``n_missing_candidates``.
  - Validation curve & best checkpoint (stored once in the JSON under ``metrics``):
    ``val_mrr_curve`` (list of ``{"epoch": int, "val_mrr": float}``),
    ``best_val_mrr``, ``best_val_epoch``.
  - Helly & cost metrics: ``helly_violation_rate``, ``degeneracy_rate``,
    ``n_certificates``, ``n_certificates_unmapped``, ``param_count``,
    ``mean_epoch_time_s``, ``total_train_time_s``, ``n_components``,
    ``mean_effective_components``, ``peak_gpu_memory_bytes``,
    ``peak_gpu_memory_mb``.
- ``timings``:
  ``dataset_load_time_s``, ``cm_load_time_s``, ``model_init_time_s``,
  ``total_train_time_s``, ``mean_epoch_time_s``, ``eval_time_s``,
  ``modelness_time_s``, ``helly_time_s``, ``faithfulness_time_s``,
  ``ranking_time_s``, ``wall_clock_s``, ``epochs``, ``peak_gpu_memory_bytes``.
- ``env``:
  ``git_hash_experiments``, ``git_hash_mowl``, ``git_dirty_experiments``,
  ``git_dirty_mowl``, ``torch_version``, ``cuda_version``, ``cuda_available``,
  ``device``, ``gpu_name``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import mowl
import numpy as np
import torch

from experiments.cm_cache import get_canonical_model
from experiments.config import (
    DATASETS,
    DATASET_DIMS,
    DATASET_EPOCHS,
    DEFAULT_HYPERPARAMS,
    METHODS,
    OBJECTIVE_METHODS,
    PRISM_METHODS,
    RAW_RESULTS_DIR,
    REPO_ROOT,
    RUNS_DIR,
    SMOKE_RESULTS_DIR,
)
from experiments.datasets import DatasetBundle, load_dataset
from experiments.metrics import (
    abox_metrics,
    cost_metrics,
    faithfulness_metrics,
    helly_metrics,
    modelness,
    point_modelness,
    ranking_metrics,
)

PLAIN_BACKBONES = ("ELBE", "BoxSquaredEL", "ELEm", "PrismEL", "CanonicalMixture")

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric import CanonicalModel, HellyCertificate


def _git_rev_parse(repo_dir: Path) -> str:
    """Return ``git rev-parse HEAD`` in ``repo_dir`` (or ``'unknown'`` on error)."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        return res.stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _git_is_dirty(repo_dir: Path) -> bool:
    """Return ``True`` iff ``git status --porcelain`` in ``repo_dir`` has entries outside untracked ``results/`` and ``logs/``."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            if line.startswith("?? "):
                rel = line[3:].strip().strip('"')
                norm = "/" + rel.lstrip("/")
                if (
                    "/results/" in norm
                    or "/logs/" in norm
                    or norm.endswith(("/results", "/logs"))
                ):
                    continue
            return True
        return False
    except Exception:  # pragma: no cover
        return True


def _mowl_repo_root() -> Path:
    """Resolve the git worktree root of the imported ``mowl`` package."""
    mowl_init = Path(mowl.__file__).resolve()
    return mowl_init.parent.parent


def collect_env_info(device: str) -> Dict[str, Any]:
    """Collect git hashes and dirty flags of both repositories, torch/CUDA versions, and GPU name."""
    cuda_avail = torch.cuda.is_available()
    gpu_name: Optional[str] = None
    if cuda_avail:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover
            gpu_name = None

    mowl_root = _mowl_repo_root()
    return {
        "git_hash_experiments": _git_rev_parse(REPO_ROOT),
        "git_hash_mowl": _git_rev_parse(mowl_root),
        "git_dirty_experiments": _git_is_dirty(REPO_ROOT),
        "git_dirty_mowl": _git_is_dirty(mowl_root),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_available": bool(cuda_avail),
        "device": str(device),
        "gpu_name": gpu_name,
    }


def _seed_everything(seed: int) -> None:
    """Seed Python ``random``, NumPy legacy RNG, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_filename(
    dataset: str,
    method: str,
    dim: int,
    seed: int,
    lr: Optional[float] = None,
    objective: Optional[str] = None,
) -> str:
    """Return canonical JSON filename ``{dataset}__{method}__d{dim}__s{seed}[__lr{lr}][__obj-{objective}].json``."""
    default_lr = float(DEFAULT_HYPERPARAMS["learning_rate"])
    stem = f"{dataset}__{method}__d{dim}__s{seed}"
    if lr is not None and abs(float(lr) - default_lr) > 1e-12:
        stem += f"__lr{float(lr):g}"
    if objective is not None:
        stem += f"__obj-{objective}"
    return f"{stem}.json"


def _resolve_certificates(
    bundle: DatasetBundle,
    cm: "CanonicalModel",
) -> List["HellyCertificate"]:
    """Obtain Helly certificates for ``bundle`` (cached on ``bundle._certificates``).

    Uses ``certificate_triangles(cm)`` first; for synthetic ``tk4``..``tk6`` where
    ``certificate_triangles(cm)`` is empty because only ``(k-1)``-wise conjunctions
    are named (Task 6 protocol ruling in ``dataset_stats.md``), falls back to
    ``helly_check(cm, k_max=k).certificates_nondegenerate``.
    """
    cached = getattr(bundle, "_certificates", None)
    if cached is not None:
        return cached

    from mowl.geometric import certificate_triangles, helly_check

    certs = list(certificate_triangles(cm))
    if not certs and bundle.name.startswith("tk"):
        k = int(bundle.name[2:])
        if k >= 4:
            rep = helly_check(cm, k_max=k, max_certificates=100)
            certs = list(rep.certificates_nondegenerate)

    setattr(bundle, "_certificates", certs)
    return certs


def _validate_method_objective(
    method_id: str,
    spec: Dict[str, Any],
    objective_override: Optional[str] = None,
) -> str:
    """Validate that plain backbones are not run with non-axiom objectives (and ELEm has no init/anchor); return effective objective."""
    model_cls_name = str(spec["model"])
    if model_cls_name == "ELEm" and (
        spec.get("init") is not None or float(spec.get("anchor_weight", 0.0)) != 0.0
    ):
        raise ValueError(
            f"ELEm (method {method_id!r}) does not support box initialization or anchor "
            f"regularization (got init={spec.get('init')!r}, anchor_weight={spec.get('anchor_weight', 0.0)!r})."
        )
    obj_kw = dict(spec.get("objective_kwargs", {}))
    effective_objective = str(
        objective_override
        if objective_override is not None
        else spec.get("objective", obj_kw.get("objective", "axiom"))
    )
    if model_cls_name in PLAIN_BACKBONES and effective_objective != "axiom":
        raise ValueError(
            f"Plain backbone {model_cls_name!r} (method {method_id!r}) does not support "
            f"objective={effective_objective!r}; only 'axiom' is supported."
        )
    return effective_objective


def _build_model(
    bundle: DatasetBundle,
    cm: Optional["CanonicalModel"],
    method_id: str,
    dim: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    delta: float,
    reg_factor: float,
    num_negs: int,
    device: str,
    checkpoint_path: Path,
    objective: Optional[str] = None,
):
    """Instantiate the requested mOWL model and apply ``ExactBoxInit`` when configured."""
    from mowl.geometric import ExactBoxInit
    from mowl.models import (
        BoxSquaredEL,
        CanonicalMixture,
        ELBE,
        ELEmbeddings,
        MultiBoxEL,
        MultiELBE,
        PrismEL,
    )

    if method_id not in METHODS:
        raise ValueError(
            f"Unknown method {method_id!r}; expected one of {list(METHODS)}"
        )
    spec = METHODS[method_id]
    model_cls_name = spec["model"]
    k = int(spec["n_components"])
    init_method = spec["init"]
    anchor_weight = float(spec["anchor_weight"])
    effective_objective = _validate_method_objective(method_id, spec, objective)
    obj_kw: Dict[str, Any] = dict(spec.get("objective_kwargs", {}))
    if objective is not None:
        obj_kw["objective"] = effective_objective

    effective_dim = max(int(dim), 3) if model_cls_name == "PrismEL" else int(dim)

    _seed_everything(seed)

    # Cache normalized GCIs in-memory on bundle to avoid reloading large cached OWL files
    # (e.g. GO's 105k normalized axioms) from disk across the 11 methods in a sweep.
    from mowl.ontology.normalize import ELNormalizer

    orig_normalize = ELNormalizer.normalize
    cached_norm_gcis: Dict[str, Any] = getattr(bundle, "_normalized_gcis_cache", None) or {}
    setattr(bundle, "_normalized_gcis_cache", cached_norm_gcis)

    def _cached_normalize(self_norm, ontology, load=False, ontology_path=None, use_cache=True):
        key = str(ontology_path) if ontology_path is not None else f"id:{id(ontology)}"
        if use_cache and not load and key in cached_norm_gcis:
            return cached_norm_gcis[key]
        res = orig_normalize(
            self_norm,
            ontology,
            load=load,
            ontology_path=ontology_path,
            use_cache=use_cache,
        )
        if use_cache and not load:
            cached_norm_gcis[key] = res
        return res

    # First construct with init_boxes=None so EmbeddingELModel normalises the
    # training ontology and populates model.class_index_dict (including auxiliary
    # normaliser classes). When init_method is set, build ExactBoxInit from the
    # cached CanonicalModel over model.class_index_dict and attach it via init_module().
    ckpt_str = str(checkpoint_path)
    ELNormalizer.normalize = _cached_normalize  # type: ignore[method-assign]
    try:
        if model_cls_name == "ELBE":
            model = ELBE(
                bundle.dataset,
                embed_dim=effective_dim,
                margin=margin,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_filepath=ckpt_str,
                device=device,
                init_boxes=None,
                anchor_weight=anchor_weight,
            )
        elif model_cls_name == "ELEm":
            model = ELEmbeddings(
                bundle.dataset,
                embed_dim=effective_dim,
                margin=margin,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_filepath=ckpt_str,
                device=device,
            )
        elif model_cls_name == "BoxSquaredEL":
            model = BoxSquaredEL(
                bundle.dataset,
                embed_dim=effective_dim,
                margin=margin,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta=delta,
                reg_factor=reg_factor,
                num_negs=num_negs,
                model_filepath=ckpt_str,
                device=device,
                init_boxes=None,
                anchor_weight=anchor_weight,
            )
        elif model_cls_name == "MultiBoxEL":
            model = MultiBoxEL(
                bundle.dataset,
                embed_dim=effective_dim,
                n_components=k,
                margin=margin,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta=delta,
                reg_factor=reg_factor,
                component_reg=0.0,
                num_negs=num_negs,
                model_filepath=ckpt_str,
                device=device,
                init_boxes=None,
                anchor_weight=anchor_weight,
                points_seed=seed,
                **obj_kw,
            )
        elif model_cls_name == "MultiELBE":
            model = MultiELBE(
                bundle.dataset,
                embed_dim=effective_dim,
                n_components=k,
                margin=margin,
                component_reg=0.0,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_filepath=ckpt_str,
                device=device,
                init_boxes=None,
                anchor_weight=anchor_weight,
                points_seed=seed,
                **obj_kw,
            )
        elif model_cls_name == "PrismEL":
            model = PrismEL(
                bundle.dataset,
                embed_dim=effective_dim,
                margin=margin,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_filepath=ckpt_str,
                device=device,
                init_boxes=None,
                anchor_weight=anchor_weight,
            )
        elif model_cls_name == "CanonicalMixture":
            model = CanonicalMixture(
                bundle.dataset,
                rank=effective_dim,
                lr=learning_rate,
                batch_size=min(batch_size, 1024),
                model_filepath=ckpt_str,
                device=device,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported model class {model_cls_name!r}")
    finally:
        ELNormalizer.normalize = orig_normalize  # type: ignore[method-assign]

    # Wrap _load_datasets so validation/testing ELNormalizer calls reuse the cache.
    orig_load_ds = model._load_datasets

    def _cached_load_datasets():
        ELNormalizer.normalize = _cached_normalize  # type: ignore[method-assign]
        try:
            return orig_load_ds()
        finally:
            ELNormalizer.normalize = orig_normalize  # type: ignore[method-assign]

    model._load_datasets = _cached_load_datasets

    init_report: Optional[Dict[str, Any]] = None
    if init_method is not None:
        from mowl.geometric.errors import UnsupportedAxiom

        if cm is None:
            cm = get_canonical_model(bundle)
        init_cache: Dict[Any, Any] = getattr(bundle, "_exact_box_init_cache", None) or {}
        setattr(bundle, "_exact_box_init_cache", init_cache)
        milp_time_limit = float(spec.get("milp_time_limit", 120.0))
        init_key = (
            int(effective_dim),
            int(k),
            str(init_method),
            int(seed),
            len(model.class_index_dict),
            milp_time_limit if init_method == "milp" else None,
        )
        if init_key in init_cache:
            init_boxes = init_cache[init_key]
        else:
            init_kwargs: Dict[str, Any] = {
                "embed_dim": effective_dim,
                "n_components": k,
                "method": init_method,
                "seed": seed,
            }
            if init_method == "milp":
                init_kwargs["milp_time_limit"] = milp_time_limit
            try:
                init_boxes = ExactBoxInit.from_canonical_model(
                    cm,
                    model.class_index_dict,
                    **init_kwargs,
                )
            except UnsupportedAxiom as exc:
                t_fb0 = time.monotonic()
                proj_boxes = ExactBoxInit.from_canonical_model(
                    cm,
                    model.class_index_dict,
                    embed_dim=effective_dim,
                    n_components=k,
                    method="projected",
                    seed=seed,
                )
                fb_report = dict(proj_boxes.report)
                fb_report["requested_method"] = init_method
                fb_report["method"] = "projected"
                fb_report["fallback"] = f"{type(exc).__name__}: {exc}"
                fb_report["elapsed_s"] = float(
                    proj_boxes.report.get("elapsed_s", time.monotonic() - t_fb0)
                )
                init_boxes = replace(proj_boxes, report=fb_report)
            init_cache[init_key] = init_boxes
        model.init_boxes = init_boxes
        init_boxes.apply_to_module(model.module)
        init_report = dict(init_boxes.report)

    return model, init_report


@contextmanager
def _chunked_module_forward(
    module: torch.nn.Module,
    chunk_size: int,
    enabled: bool,
) -> Iterator[None]:
    """When ``enabled``, chunk 2-D GCI tensor batches larger than ``chunk_size`` inside ``module.forward``.

    Upstream ``EmbeddingELModel.train`` evaluates ``self.module(gci_dataset[:], gci_name)``
    in a single full-batch call per GCI normal form rather than iterating over
    ``training_dataloaders``. Wrapping ``module.forward`` to slice inputs along
    axis 0 when ``len(gci_tensor) > chunk_size`` bounds peak *non-saved*
    intermediate tensor allocations inside a single chunk's forward pass (though
    not autograd saved activations across chunks, because ``backward()`` is
    called once on the concatenated loss per GCI normal form without gradient
    checkpointing) while preserving exact full-batch mean losses and gradients.
    """
    if not enabled:
        yield
        return

    orig_forward = module.forward

    def _wrapped_forward(gci_tensor: torch.Tensor, gci_name: str, neg: bool = False):
        if (
            isinstance(gci_tensor, torch.Tensor)
            and gci_tensor.dim() == 2
            and gci_tensor.shape[0] > chunk_size
        ):
            parts = [
                orig_forward(gci_tensor[start : start + chunk_size], gci_name, neg=neg)
                for start in range(0, int(gci_tensor.shape[0]), chunk_size)
            ]
            return torch.cat(parts, dim=0)
        return orig_forward(gci_tensor, gci_name, neg=neg)

    module.forward = _wrapped_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = orig_forward  # type: ignore[method-assign]


def _train_model_with_chunking(
    model: Any,
    epochs: int,
    validate_every: int,
    effective_batch_size: int,
    epoch_times: List[float],
) -> None:
    """Run ``model.train(epochs, validate_every)`` recording per-epoch wall-clock times."""
    epoch_times.clear()
    _ = model.training_datasets
    t_last = [time.monotonic()]
    extra_cb = getattr(model, "_abox_epoch_callback", None)

    def _on_epoch(epoch_idx: int, _m: Any) -> None:
        now = time.monotonic()
        epoch_times.append(now - t_last[0])
        if extra_cb is not None:
            extra_cb(epoch_idx, _m)
        t_last[0] = time.monotonic()

    chunk_needed = effective_batch_size < int(DEFAULT_HYPERPARAMS["batch_size"])
    with _chunked_module_forward(model.module, effective_batch_size, enabled=chunk_needed):
        t_last[0] = time.monotonic()
        model.train(
            epochs=epochs,
            validate_every=validate_every,
            epoch_callback=_on_epoch,
        )


def execute_run(
    *,
    dataset: str,
    method: str,
    dim: int,
    seed: int,
    epochs: Optional[int] = None,
    batch_size: int = int(DEFAULT_HYPERPARAMS["batch_size"]),
    learning_rate: float = float(DEFAULT_HYPERPARAMS["learning_rate"]),
    lr: Optional[float] = None,
    objective: Optional[str] = None,
    validate_every: int = int(DEFAULT_HYPERPARAMS["validate_every"]),
    device: Optional[str] = None,
    out_dir: Path = RAW_RESULTS_DIR,
    runs_dir: Path = RUNS_DIR,
    bundle: Optional[DatasetBundle] = None,
    cm: Optional["CanonicalModel"] = None,
) -> Dict[str, Any]:
    """Execute one experiment cell, write its JSON atomically, and return the payload."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {list(METHODS)}")

    resolved_lr = float(lr if lr is not None else learning_rate)
    effective_objective = _validate_method_objective(method, METHODS[method], objective)

    resolved_device = (
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_epochs = int(epochs if epochs is not None else DATASET_EPOCHS[dataset])

    t_run_start = time.monotonic()
    if bundle is None:
        t0 = time.monotonic()
        bundle = load_dataset(dataset)
        dataset_load_time_s = time.monotonic() - t0
    else:
        dataset_load_time_s = 0.0

    t_cm0 = time.monotonic()
    need_cm = (
        bundle.task != "abox"
        or METHODS[method].get("init") is not None
    )
    if cm is None and need_cm:
        cm = get_canonical_model(bundle)
    cm_load_time_s = time.monotonic() - t_cm0

    runs_dir = Path(runs_dir).expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_fname = run_filename(
        dataset, method, dim, seed, lr=resolved_lr, objective=objective
    )
    ckpt_path = runs_dir / f"{Path(json_fname).stem}.pt"

    margin = float(DEFAULT_HYPERPARAMS["margin"])
    delta = float(DEFAULT_HYPERPARAMS["delta"])
    reg_factor = float(DEFAULT_HYPERPARAMS["reg_factor"])
    num_negs = int(DEFAULT_HYPERPARAMS["num_negs"])
    eval_gci_name = str(DEFAULT_HYPERPARAMS["eval_gci_name"])

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    effective_batch_size = int(batch_size)
    oom_retries = 0
    max_oom_retries = 3
    epoch_times: List[float] = []
    validation_curve: List[Dict[str, Any]] = []
    best_val_mrr: Optional[float] = None
    best_val_epoch: Optional[int] = None
    model: Any = None
    init_report: Optional[Dict[str, Any]] = None

    if bundle.task == "abox":
        from experiments.abox_eval import compute_validation_mrr, load_abox_split_data

        # Pre-load validation split structures once outside the training timer
        load_abox_split_data(bundle, split="valid")

    while True:
        t_init0 = time.monotonic()
        model, init_report = _build_model(
            bundle=bundle,
            cm=cm,
            method_id=method,
            dim=dim,
            seed=seed,
            batch_size=effective_batch_size,
            learning_rate=resolved_lr,
            margin=margin,
            delta=delta,
            reg_factor=reg_factor,
            num_negs=num_negs,
            device=resolved_device,
            checkpoint_path=ckpt_path,
            objective=objective,
        )
        # Ensure training/validation/testing ELDatasets are loaded before starting the training timer
        # so the first cell of a dataset does not absorb ontology/ELDataset loading time.
        _ = model.training_datasets
        model_init_time_s = time.monotonic() - t_init0

        validation_curve.clear()
        best_val_mrr = None
        best_val_epoch = None

        if bundle.task == "abox":
            torch.save(model.module.state_dict(), ckpt_path)
            # Disable mOWL's loss-based validation checkpointing; use 50-epoch validation MRR instead
            eff_validate_every = resolved_epochs + 1
            val_mrr_interval = 50
            best_mrr_box = [-1.0]
            best_ep_box = [0]

            def _abox_val_cb(epoch_idx: int, m_obj: Any) -> None:
                ep_num = int(epoch_idx + 1)
                if ep_num % val_mrr_interval == 0 or ep_num == resolved_epochs:
                    v_mrr = float(compute_validation_mrr(m_obj, bundle))
                    validation_curve.append({"epoch": ep_num, "val_mrr": v_mrr})
                    if v_mrr > best_mrr_box[0]:
                        best_mrr_box[0] = v_mrr
                        best_ep_box[0] = ep_num
                        torch.save(m_obj.module.state_dict(), ckpt_path)

            setattr(model, "_abox_epoch_callback", _abox_val_cb)
        elif bundle.task == "subsumption" and bundle.dataset.validation is not None:
            model.eval_gci_name = eval_gci_name
            # Ensure a valid initial checkpoint exists even if epochs < validate_every
            # (e.g. smoke runs with epochs=3 and validate_every=10).
            torch.save(model.module.state_dict(), ckpt_path)
            eff_validate_every = min(validate_every, max(1, resolved_epochs))
        else:
            # For satisfaction datasets (no validation), pass validate_every > epochs
            # unless validation is needed, avoiding per-epoch print overhead.
            eff_validate_every = max(validate_every, resolved_epochs + 1)

        try:
            _seed_everything(seed)
            t_train0 = time.monotonic()
            _train_model_with_chunking(
                model=model,
                epochs=resolved_epochs,
                validate_every=eff_validate_every,
                effective_batch_size=effective_batch_size,
                epoch_times=epoch_times,
            )
            total_train_time_s = time.monotonic() - t_train0
            if bundle.task == "abox" and validation_curve:
                best_val_mrr = float(best_mrr_box[0])
                best_val_epoch = int(best_ep_box[0])
            break
        except torch.cuda.OutOfMemoryError:
            if oom_retries >= max_oom_retries or effective_batch_size <= 1:
                raise
            oom_retries += 1
            effective_batch_size = max(1, effective_batch_size // 2)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load best validation checkpoint for subsumption and abox datasets; for satisfaction
    # datasets (no validation split), persist the final state dict to ckpt_path.
    if (
        bundle.task == "subsumption"
        and bundle.dataset.validation is not None
        and ckpt_path.exists()
    ):
        model.load_best_model()
    elif bundle.task == "abox" and ckpt_path.exists():
        model.module.load_state_dict(
            torch.load(ckpt_path, map_location=resolved_device, weights_only=True)
        )
    else:
        torch.save(model.module.state_dict(), ckpt_path)

    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated())
        if resolved_device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )

    # Compute all Spec §5.3 / §3.1-3.2 metrics BEFORE writing the JSON
    t_eval0 = time.monotonic()

    t_m0 = time.monotonic()
    m_modelness = modelness(model, batch_size=effective_batch_size)
    m_point_mod = point_modelness(model)
    m_train_margin_dict: Dict[str, float] = {}
    if method in OBJECTIVE_METHODS or method in PRISM_METHODS:
        raw_tm = modelness(model, batch_size=effective_batch_size, margin=margin)
        m_train_margin_dict["modelness_train_margin"] = float(raw_tm["modelness_all"])
        for k_tm, v_tm in raw_tm.items():
            if k_tm != "modelness_all" and k_tm.startswith("modelness_"):
                gci_tm = k_tm[len("modelness_") :]
                m_train_margin_dict[f"modelness_train_margin_{gci_tm}"] = float(v_tm)
    modelness_time_s = time.monotonic() - t_m0

    t_h0 = time.monotonic()
    if cm is not None and METHODS[method]["model"] not in ("CanonicalMixture", "ELEm"):
        certs = _resolve_certificates(bundle, cm)
        m_helly = helly_metrics(model, bundle=bundle, cm=cm, certificates=certs)
    else:
        m_helly = {
            "helly_violation_rate": None,
            "degeneracy_rate": None,
            "n_certificates": 0,
            "n_certificates_unmapped": 0,
        }
    helly_time_s = time.monotonic() - t_h0

    if bundle.task == "abox":
        t_ab0 = time.monotonic()
        m_abox = abox_metrics(model, bundle=bundle, split="test", seed=0)
        abox_eval_time_s = time.monotonic() - t_ab0
        faithfulness_time_s = abox_eval_time_s
        ranking_time_s = abox_eval_time_s
        m_faith = {}
        m_rank = {}
    else:
        m_abox = {}
        t_f0 = time.monotonic()
        if bundle.task == "subsumption":
            m_faith = faithfulness_metrics(model, bundle=bundle, cm=cm, seed=0)
        else:
            m_faith = {
                "auc_pos_neg": None,
                "pair_fvr": None,
                "fvr": None,
                "n_pos": 0,
                "n_und": 0,
                "n_neg": 0,
            }
        faithfulness_time_s = time.monotonic() - t_f0

        t_r0 = time.monotonic()
        if bundle.task == "subsumption":
            m_rank = ranking_metrics(model, bundle=bundle, device=resolved_device)
        else:
            m_rank = {}
        ranking_time_s = time.monotonic() - t_r0

    eval_time_s = time.monotonic() - t_eval0
    wall_clock_s = time.monotonic() - t_run_start

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        peak_gpu_bytes = max(peak_gpu_bytes, int(torch.cuda.max_memory_allocated()))

    timings: Dict[str, Any] = {
        "dataset_load_time_s": float(dataset_load_time_s),
        "cm_load_time_s": float(cm_load_time_s),
        "model_init_time_s": float(model_init_time_s),
        "total_train_time_s": float(total_train_time_s),
        "mean_epoch_time_s": (
            float(np.mean(epoch_times))
            if epoch_times
            else (
                float(total_train_time_s / resolved_epochs)
                if resolved_epochs > 0
                else 0.0
            )
        ),
        "eval_time_s": float(eval_time_s),
        "modelness_time_s": float(modelness_time_s),
        "helly_time_s": float(helly_time_s),
        "faithfulness_time_s": float(faithfulness_time_s),
        "ranking_time_s": float(ranking_time_s),
        "wall_clock_s": float(wall_clock_s),
        "epochs": int(resolved_epochs),
        "peak_gpu_memory_bytes": int(peak_gpu_bytes),
    }

    mod_obj = getattr(model, "module", model)
    effective_dim_val = int(
        getattr(
            model,
            "embed_dim",
            getattr(
                mod_obj,
                "embed_dim",
                max(int(dim), 3) if METHODS[method]["model"] == "PrismEL" else int(dim),
            ),
        )
    )
    has_tau = hasattr(model, "tau_start") and hasattr(model, "tau_end")
    tau_start_val: Optional[float] = (
        float(getattr(model, "tau_start")) if has_tau else None
    )
    tau_end_val: Optional[float] = (
        float(getattr(model, "tau_end")) if has_tau else None
    )
    tau_anneal_val: Optional[bool] = (
        bool(getattr(model, "tau_anneal", METHODS[method].get("tau_anneal", False)))
        if has_tau
        else None
    )
    tau_schedule_val: Optional[Dict[str, Any]] = (
        {
            "tau_start": tau_start_val,
            "tau_end": tau_end_val,
            "tau_anneal": tau_anneal_val,
        }
        if has_tau
        else None
    )
    diversity_reg_val = float(
        getattr(
            model,
            "diversity_reg",
            getattr(mod_obj, "diversity_reg", METHODS[method].get("diversity_reg", 0.0)),
        )
    )
    comp_agg_val = str(
        getattr(
            model,
            "component_aggregation",
            getattr(
                mod_obj,
                "component_aggregation",
                METHODS[method].get("component_aggregation", "hard"),
            ),
        )
    )

    m_cost = cost_metrics(model, timings=timings)
    del model
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_metrics: Dict[str, Any] = {}
    all_metrics.update(m_rank)
    all_metrics.update(m_faith)
    all_metrics.update(m_modelness)
    all_metrics.update(m_train_margin_dict)
    all_metrics["point_modelness"] = m_point_mod
    all_metrics.update(m_helly)
    all_metrics.update(m_cost)
    if bundle.task == "abox":
        all_metrics.update(m_abox)
        all_metrics["val_mrr_curve"] = validation_curve
        all_metrics["best_val_mrr"] = best_val_mrr
        all_metrics["best_val_epoch"] = best_val_epoch

    method_spec = METHODS[method]
    config_dict: Dict[str, Any] = {
        "dataset": dataset,
        "task": bundle.task,
        "method": method,
        "model": method_spec["model"],
        "n_components": int(method_spec["n_components"]),
        "init": method_spec["init"],
        "anchor_weight": float(method_spec["anchor_weight"]),
        "objective": str(effective_objective),
        "objective_variant": str(
            effective_objective
            if objective is not None
            else method_spec.get("objective_variant", effective_objective)
        ),
        "component_aggregation": comp_agg_val,
        "tau_start": tau_start_val,
        "tau_end": tau_end_val,
        "tau_anneal": tau_anneal_val,
        "tau_schedule": tau_schedule_val,
        "diversity_reg": diversity_reg_val,
        "dim": int(dim),
        "effective_dim": int(effective_dim_val),
        "seed": int(seed),
        "epochs": int(resolved_epochs),
        "validate_every": 50 if bundle.task == "abox" else int(validate_every),
        "eval_gci_name": (
            "class_assertion"
            if bundle.task == "abox"
            else (eval_gci_name if bundle.task == "subsumption" else None)
        ),
        "margin": margin,
        "delta": delta,
        "reg_factor": reg_factor,
        "num_negs": num_negs,
        "learning_rate": resolved_lr,
        "lr": resolved_lr,
        "batch_size": int(batch_size),
        "effective_batch_size": int(effective_batch_size),
        "oom_retries": int(oom_retries),
        "device": resolved_device,
        "checkpoint_path": str(ckpt_path),
        "init_report": init_report,
    }
    if method in OBJECTIVE_METHODS or method in PRISM_METHODS:
        config_dict["margin_semantics"] = (
            "relaxing"
            if method_spec["model"] in ("BoxSquaredEL", "MultiBoxEL")
            else "tightening"
        )

    payload: Dict[str, Any] = {
        "config": config_dict,
        "metrics": all_metrics,
        "timings": timings,
        "env": collect_env_info(resolved_device),
    }

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / json_fname
    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, out_path)
    except BaseException:  # pragma: no cover
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``experiments/run.py``."""
    parser = argparse.ArgumentParser(
        description="Run a single learnable geometric embedding experiment cell."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS,
        help="Dataset identifier.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(METHODS.keys()),
        help="Method identifier from config.METHODS.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension d (defaults to first entry in config.DATASET_DIMS[dataset]).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (defaults to config.DATASET_EPOCHS[dataset], or 3 with --smoke).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (3 epochs unless --epochs is set; default --out is results/smoke/).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(DEFAULT_HYPERPARAMS["batch_size"]),
        help="Initial training batch size (default: 32768).",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=float(DEFAULT_HYPERPARAMS["learning_rate"]),
        help="Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        help="Optional objective override ('axiom', 'point', or 'mixed').",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=int(DEFAULT_HYPERPARAMS["validate_every"]),
        help="Validation interval in epochs (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cuda' or 'cpu'; defaults to 'cuda' if available else 'cpu').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for JSON results (default: results/raw/, or results/smoke/ with --smoke).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory for model checkpoints (default: ~/tmp/gfe-runs/).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``experiments/run.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    dim = int(args.dim if args.dim is not None else DATASET_DIMS[args.dataset][0])
    epochs = args.epochs
    if epochs is None and args.smoke:
        epochs = 3
    out_dir = args.out
    if out_dir is None:
        out_dir = SMOKE_RESULTS_DIR if args.smoke else RAW_RESULTS_DIR

    payload = execute_run(
        dataset=args.dataset,
        method=args.method,
        dim=dim,
        seed=args.seed,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        objective=args.objective,
        validate_every=args.validate_every,
        device=args.device,
        out_dir=out_dir,
        runs_dir=args.runs_dir,
    )
    cfg = payload["config"]
    t = payload["timings"]
    m = payload["metrics"]
    print(
        f"[run] {cfg['dataset']} | {cfg['method']} | d={cfg['dim']} | s={cfg['seed']} | "
        f"epochs={cfg['epochs']} | modelness_all={m.get('modelness_all', 0.0):.4f} | "
        f"wall={t['wall_clock_s']:.2f}s | gpu_mb={m.get('peak_gpu_memory_mb', 0.0):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
