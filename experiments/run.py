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
    RAW_RESULTS_DIR,
    REPO_ROOT,
    RUNS_DIR,
    SMOKE_RESULTS_DIR,
)
from experiments.datasets import DatasetBundle, load_dataset
from experiments.metrics import (
    cost_metrics,
    faithfulness_metrics,
    helly_metrics,
    modelness,
    ranking_metrics,
)

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


def run_filename(dataset: str, method: str, dim: int, seed: int) -> str:
    """Return canonical JSON filename ``{dataset}__{method}__d{dim}__s{seed}.json``."""
    return f"{dataset}__{method}__d{dim}__s{seed}.json"


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
):
    """Instantiate the requested mOWL model and apply ``ExactBoxInit`` when configured."""
    from mowl.geometric import ExactBoxInit
    from mowl.models import BoxSquaredEL, ELBE, MultiBoxEL

    if method_id not in METHODS:
        raise ValueError(
            f"Unknown method {method_id!r}; expected one of {list(METHODS)}"
        )
    spec = METHODS[method_id]
    model_cls_name = spec["model"]
    k = int(spec["n_components"])
    init_method = spec["init"]
    anchor_weight = float(spec["anchor_weight"])

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
                embed_dim=dim,
                margin=margin,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_filepath=ckpt_str,
                device=device,
                init_boxes=None,
                anchor_weight=anchor_weight,
            )
        elif model_cls_name == "BoxSquaredEL":
            model = BoxSquaredEL(
                bundle.dataset,
                embed_dim=dim,
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
                embed_dim=dim,
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
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported model class {model_cls_name!r}")
    finally:
        ELNormalizer.normalize = orig_normalize  # type: ignore[method-assign]

    # Also wrap _load_datasets so validation/testing ELNormalizer calls reuse the cache
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
            int(dim),
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
                "embed_dim": dim,
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
                    embed_dim=dim,
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

    def _on_epoch(epoch_idx: int, _m: Any) -> None:
        now = time.monotonic()
        epoch_times.append(now - t_last[0])
        t_last[0] = now

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
    if cm is None:
        cm = get_canonical_model(bundle)
    cm_load_time_s = time.monotonic() - t_cm0

    runs_dir = Path(runs_dir).expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = runs_dir / f"{dataset}__{method}__d{dim}__s{seed}.pt"

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
    model: Any = None
    init_report: Optional[Dict[str, Any]] = None

    while True:
        t_init0 = time.monotonic()
        model, init_report = _build_model(
            bundle=bundle,
            cm=cm,
            method_id=method,
            dim=dim,
            seed=seed,
            batch_size=effective_batch_size,
            learning_rate=learning_rate,
            margin=margin,
            delta=delta,
            reg_factor=reg_factor,
            num_negs=num_negs,
            device=resolved_device,
            checkpoint_path=ckpt_path,
        )
        # Ensure training/validation/testing ELDatasets are loaded before starting the training timer
        # so the first cell of a dataset does not absorb ontology/ELDataset loading time.
        _ = model.training_datasets
        model_init_time_s = time.monotonic() - t_init0

        if bundle.task == "subsumption" and bundle.dataset.validation is not None:
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
            break
        except torch.cuda.OutOfMemoryError:
            if oom_retries >= max_oom_retries or effective_batch_size <= 1:
                raise
            oom_retries += 1
            effective_batch_size = max(1, effective_batch_size // 2)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load best validation checkpoint for subsumption datasets
    if (
        bundle.task == "subsumption"
        and bundle.dataset.validation is not None
        and ckpt_path.exists()
    ):
        model.load_best_model()

    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated())
        if resolved_device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )

    # Compute all Spec §5.3 metrics BEFORE writing the JSON
    t_eval0 = time.monotonic()

    t_m0 = time.monotonic()
    m_modelness = modelness(model, batch_size=effective_batch_size)
    modelness_time_s = time.monotonic() - t_m0

    t_h0 = time.monotonic()
    certs = _resolve_certificates(bundle, cm)
    m_helly = helly_metrics(model, bundle=bundle, cm=cm, certificates=certs)
    helly_time_s = time.monotonic() - t_h0

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

    m_cost = cost_metrics(model, timings=timings)
    del model
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_metrics: Dict[str, Any] = {}
    all_metrics.update(m_rank)
    all_metrics.update(m_faith)
    all_metrics.update(m_modelness)
    all_metrics.update(m_helly)
    all_metrics.update(m_cost)

    method_spec = METHODS[method]
    config_dict: Dict[str, Any] = {
        "dataset": dataset,
        "task": bundle.task,
        "method": method,
        "model": method_spec["model"],
        "n_components": int(method_spec["n_components"]),
        "init": method_spec["init"],
        "anchor_weight": float(method_spec["anchor_weight"]),
        "dim": int(dim),
        "seed": int(seed),
        "epochs": int(resolved_epochs),
        "validate_every": int(validate_every),
        "eval_gci_name": eval_gci_name if bundle.task == "subsumption" else None,
        "margin": margin,
        "delta": delta,
        "reg_factor": reg_factor,
        "num_negs": num_negs,
        "learning_rate": learning_rate,
        "batch_size": int(batch_size),
        "effective_batch_size": int(effective_batch_size),
        "oom_retries": int(oom_retries),
        "device": resolved_device,
        "checkpoint_path": str(ckpt_path),
        "init_report": init_report,
    }

    payload: Dict[str, Any] = {
        "config": config_dict,
        "metrics": all_metrics,
        "timings": timings,
        "env": collect_env_info(resolved_device),
    }

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / run_filename(dataset, method, dim, seed)
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
        "--learning-rate",
        type=float,
        default=float(DEFAULT_HYPERPARAMS["learning_rate"]),
        help="Adam learning rate (default: 0.001).",
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
