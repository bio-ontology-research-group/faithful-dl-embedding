"""B3: single-box baselines (ELEm, ELBE, Box²EL, TransBox) at equal parameter count.

Purpose
-------
Train the existing single-region methods through the building blocks of the
experiment harness (:func:`experiments.run._build_model` and
:func:`experiments.run._train_model_with_chunking`, same hyper-parameters; the model
after the last epoch is evaluated, ``--checkpoint valid`` restores best-validation
checkpointing) at
dimensions ``d in {1, 2, 4, 8, 16, 32, 64, 128}`` and record, per trained model,

- ``modelness_all`` and per-GCI modelness from :func:`experiments.metrics.modelness`
  (margin 0);
- crisp named subsumption: ``A ⊑ B`` holds iff the box of ``A`` is contained in the
  box of ``B`` (ELEm: ball containment ``|c_A - c_B| + r_A <= r_B``); boxes/balls of
  width (diameter) ``<= --deg-eps`` in some coordinate count as unsatisfiable, i.e.
  subsumed by every class and containing only unsatisfiable classes; exact
  precision/recall over named classes satisfiable in the reference
  (``CanonicalModel``, as in B1), computed chunk-wise in torch without materialising
  the predicted pair set; ``precision_live``/``recall_live`` restrict both sides to
  classes with a non-degenerate region (the convention of ``budget.py``);
- the same under two further readings: ``crisp_closed`` (every region is a non-empty
  closed set, exact containment) and a tolerance sweep ``tol_sweep`` (closed sets,
  containment up to a slack; ``crisp_tol`` is the slack equal to the training margin,
  ``*_best_tol`` the slack with the best F1 against the reference, i.e. the most
  favourable crisp reading of the trained model);
- parameter counts: all ``requires_grad`` parameters of the module and, separately,
  those of class-indexed tables (parameter names containing ``class``).

``transbox`` is registered here (in-process, not in ``experiments/config.py``) with
mOWL's :class:`~mowl.models.TransBox` and its default hyper-parameters (margin 0.1,
learning rate 0.001, batch size 4096, ``reg_factor`` 0.1, ``use_enhancement=True``).

Inputs
------
Either one cell (``--dataset --method --dim --seed``) or a cells file
(``--cells FILE`` with lines ``<dataset> <method> <dim> <seed>``; ``--start``/``--count``
select a slice). ``--epochs`` overrides the default epochs (harness defaults:
``DATASET_EPOCHS`` for pizza and tk3..tk6 = 2000; the other synthetic theories use the
synthetic default 2000; GO/FoodOn default to the harness value 300, which under-fits
-- the sweep uses ``--epochs 3000`` for them, see ``baselines.sbatch``).

Outputs
-------
``<out>/<dataset>__<method>__d<dim>__s<seed>.json`` (default
``experiments/results/raw-capacity/baselines/``); checkpoints under
``$GFE_RUNS_DIR/capacity`` (default ``~/tmp/gfe-runs/capacity``).

Usage
-----
``python -m experiments.capacity.baselines --dataset pizza --method box2el --dim 16 --seed 0``
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from experiments.capacity.common import (
    CAPACITY_RAW_DIR,
    J24_DATASETS,
    Reference,
    load_capacity_dataset,
    peak_rss_gb,
    reference_entailments,
    write_json,
)
from experiments.config import DATASET_EPOCHS, DEFAULT_HYPERPARAMS, METHODS, RUNS_DIR
from experiments.grounded.theories import THEORIES

DEFAULT_OUT = CAPACITY_RAW_DIR / "baselines"
BASELINE_METHODS = ("elem", "elbe", "box2el", "transbox")
BASELINE_DIMS = (1, 2, 4, 8, 16, 32, 64, 128)
BASELINE_SEEDS = (0, 1, 2)
# containment slacks of the tolerance sweep (closed-set reading); the best F1 over
# these is the most favourable crisp reading of a trained model ("oracle tolerance")
TOL_SWEEP = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
# No Box²EL benchmark ontologies: corrupt upstream (mOWL issue #178).
BASELINE_DATASETS = ("pizza", "go", "foodon") + tuple(THEORIES)

TRANSBOX_SPEC: Dict[str, Any] = {
    "model": "TransBox",
    "n_components": 1,
    "init": None,
    "anchor_weight": 0.0,
}


def register_transbox() -> None:
    """Register method id ``transbox`` in the in-process ``METHODS`` table (idempotent)."""
    METHODS.setdefault("transbox", dict(TRANSBOX_SPEC))


register_transbox()


def default_epochs(dataset: str) -> int:
    """Harness default number of epochs for ``dataset``.

    Parameters
    ----------
    dataset : str

    Returns
    -------
    int
        ``DATASET_EPOCHS[dataset]`` when defined, 300 for the Box²EL benchmark
        ontologies (as for GO/FoodOn), else 2000 (synthetic default).
    """
    if dataset in DATASET_EPOCHS:
        return int(DATASET_EPOCHS[dataset])
    if dataset in J24_DATASETS:
        return int(DATASET_EPOCHS["go"])
    return int(DATASET_EPOCHS["tk3"])


def parse_cells(lines: Iterable[str]) -> List[Tuple[str, str, int, int]]:
    """Parse a cells file: one ``<dataset> <method> <dim> <seed>`` per line.

    Blank lines and lines starting with ``#`` are ignored.

    Parameters
    ----------
    lines : iterable of str

    Returns
    -------
    list of (dataset, method, dim, seed)
    """
    cells: List[Tuple[str, str, int, int]] = []
    for n, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"line {n}: expected '<dataset> <method> <dim> <seed>', got {raw!r}")
        ds, method, dim, seed = parts
        cells.append((ds, method, int(dim), int(seed)))
    return cells


def write_cells_file(path: Path, datasets: Sequence[str] = BASELINE_DATASETS,
                     methods: Sequence[str] = BASELINE_METHODS,
                     dims: Sequence[int] = BASELINE_DIMS,
                     seeds: Sequence[int] = BASELINE_SEEDS) -> int:
    """Write the full grid of cells (dataset-major) and return the number of lines."""
    lines = [f"{ds} {m} {d} {s}" for ds in datasets for m in methods for d in dims for s in seeds]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def cell_filename(dataset: str, method: str, dim: int, seed: int) -> str:
    """Result file name of one cell."""
    return f"{dataset}__{method}__d{int(dim)}__s{int(seed)}.json"


# --------------------------------------------------------------------------- regions


def class_region_arrays(model: Any):
    """Class regions of a trained single-region model.

    Parameters
    ----------
    model : mOWL model
        ELEm, ELBE, Box²EL or TransBox.

    Returns
    -------
    (kind, a, b)
        ``("box", lower, upper)`` with ``(n, d)`` float tensors, or
        ``("ball", center, radius)`` with ``(n, d)`` and ``(n,)`` tensors.
    """
    import torch

    from experiments.metrics import class_regions

    module = getattr(model, "module", model)
    with torch.no_grad():
        if hasattr(module, "class_center_embedding") and hasattr(module, "class_offset_embedding"):
            c = module.class_center_embedding.weight.detach().float()
            o = module.class_offset_embedding.weight.detach().float().abs()
            return "box", c - o, c + o
        if hasattr(module, "class_rad") and hasattr(module, "class_embed"):
            c = module.class_embed.weight.detach().float()
            r = module.class_rad.weight.detach().float().abs().reshape(-1)
            return "ball", c, r
        cen, off = class_regions(model)
        cen_t = torch.as_tensor(cen[:, 0, :])
        off_t = torch.as_tensor(off[:, 0, :])
        return "box", cen_t - off_t, cen_t + off_t


def degenerate_mask(kind: str, a, b, deg_eps: float):
    """``True`` for regions counted as empty (unsatisfiable).

    Boxes: some side ``upper - lower <= deg_eps``. Balls: diameter ``2 r <= deg_eps``.
    """
    if kind == "box":
        return ((b - a) <= deg_eps).any(dim=1)
    return (2.0 * b) <= deg_eps


def crisp_subsumption_metrics(
    kind: str,
    a,
    b,
    ids: np.ndarray,
    ref_matrix,
    deg_eps: float = 1e-3,
    chunk: int = 1024,
    device: str = "cpu",
    dense_until: float = 0.02,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Exact precision/recall of crisp region containment over a class subset.

    Parameters
    ----------
    kind : {"box", "ball"}
    a, b : torch.Tensor
        Region parameters of all classes (see :func:`class_region_arrays`).
    ids : numpy.ndarray
        Class indices of the compared classes (pairs ``(A, B)``, ``A != B``).
    ref_matrix : scipy.sparse matrix
        Reference ``(len(ids), len(ids))`` entailments (no diagonal).
    deg_eps : float
        Degeneracy threshold.
    chunk : int
        Source classes per chunk.
    device : str
        Torch device for the chunked comparison.
    dense_until : float
        Box mode: switch from dense masks to candidate lists once the candidate
        density falls below this.
    tol : float
        Containment slack: box ``i`` is in box ``j`` iff ``lo_j - tol <= lo_i`` and
        ``hi_i <= hi_j + tol`` in every coordinate (balls: ``|c_i - c_j| + r_i <= r_j
        + tol``). ``0`` is exact containment; the training margin gives the reading
        under which a zero-loss Box²EL model (loss ``relu(. - gamma)``) is correct.

    Returns
    -------
    dict
        ``tp``, ``fp``, ``fn``, ``n_pred``, ``n_ref``, ``precision``, ``recall``,
        ``n_degenerate`` (among ``ids``).

    Notes
    -----
    The predicted pair set is never stored: its size is counted chunk-wise and
    true positives are obtained by testing the reference pairs directly. A
    degenerate class is subsumed by every other class and contains only
    degenerate classes.
    """
    import torch

    from scipy import sparse

    ids_t = torch.as_tensor(np.asarray(ids, dtype=np.int64))
    A = a[ids_t].to(device)
    B = b[ids_t].to(device)
    n = int(len(ids))
    deg = degenerate_mask(kind, A, B, deg_eps)
    good = torch.nonzero(~deg).reshape(-1)
    Ag, Bg = A[good], B[good]
    ng = int(len(good))

    def contains(i, j, Ai, Bi, Aj, Bj):
        """Region i contained in region j for index tensors (vectorised)."""
        if kind == "box":
            return ((Aj - tol <= Ai) & (Bi <= Bj + tol)).all(dim=-1)
        dist = torch.linalg.norm(Ai - Aj, dim=-1)
        return dist + Bi <= Bj + tol

    # count predicted pairs among non-degenerate classes (self pairs excluded below)
    n_pred_good = 0
    n_self = 0
    for s in range(0, ng, chunk):
        e = min(s + chunk, ng)
        if kind == "ball":
            # exact differences (the default matmul shortcut of cdist is inexact near 0
            # and would disagree with ``contains`` on boundary pairs)
            dist = torch.cdist(Ag[s:e], Ag, compute_mode="donot_use_mm_for_euclid_dist")
            held = dist + Bg[s:e, None] <= Bg[None, :] + tol
            n_pred_good += int(held.sum().item())
            n_self += int(held[torch.arange(e - s, device=held.device),
                               torch.arange(s, e, device=held.device)].sum().item())
            continue
        d = Ag.shape[1]
        mask = ((Ag[None, :, 0] - tol <= Ag[s:e, None, 0])
                & (Bg[s:e, None, 0] <= Bg[None, :, 0] + tol))
        k = 1
        while k < d and mask.float().mean().item() > dense_until:
            mask &= ((Ag[None, :, k] - tol <= Ag[s:e, None, k])
                     & (Bg[s:e, None, k] <= Bg[None, :, k] + tol))
            k += 1
        if k >= d:  # all coordinates tested densely: count without listing the pairs
            n_pred_good += int(mask.sum().item())
            n_self += e - s
            del mask
            continue
        ii, jj = torch.nonzero(mask, as_tuple=True)
        del mask
        n_self += e - s
        if len(ii):
            ok = ((Ag[jj, k:] - tol <= Ag[s + ii, k:])
                  & (Bg[s + ii, k:] <= Bg[jj, k:] + tol)).all(dim=1)
            n_pred_good += int(ok.sum().item())
        else:
            n_pred_good += int(len(ii))
    n_pred_good -= n_self
    n_deg = n - ng
    n_pred = n_pred_good + n_deg * (n - 1)

    ref = sparse.coo_matrix(ref_matrix)
    keep = ref.row != ref.col
    ri = torch.as_tensor(ref.row[keep].astype(np.int64), device=device)
    rj = torch.as_tensor(ref.col[keep].astype(np.int64), device=device)
    n_ref = int(len(ri))
    tp = 0
    tp_live = 0
    n_ref_live = 0
    for s in range(0, n_ref, 1 << 20):
        i, j = ri[s:s + (1 << 20)], rj[s:s + (1 << 20)]
        di, dj = deg[i], deg[j]
        geo = contains(i, j, A[i], B[i], A[j], B[j])
        held = di | (~dj & geo)
        tp += int(held.sum().item())
        both = ~di & ~dj
        n_ref_live += int(both.sum().item())
        tp_live += int((both & geo).sum().item())
    return {
        # same convention as budget.py: pairs among classes with a non-empty region
        "n_live": ng,
        "n_pred_live": n_pred_good,
        "n_ref_live": n_ref_live,
        "tp_live": tp_live,
        "precision_live": float(tp_live / n_pred_good) if n_pred_good else 1.0,
        "recall_live": float(tp_live / n_ref_live) if n_ref_live else 1.0,
        "n_false_pairs_live": n_pred_good - tp_live,
        "false_pairs_per_class_live": float((n_pred_good - tp_live) / ng) if ng else 0.0,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_ref - tp,
        "n_pred": n_pred,
        "n_ref": n_ref,
        "precision": float(tp / n_pred) if n_pred else 1.0,
        "recall": float(tp / n_ref) if n_ref else 1.0,
        "n_degenerate": int(n_deg),
    }


def parameter_counts(model: Any) -> Dict[str, int]:
    """Count trainable parameters: all, and those of class-indexed tables.

    Parameters
    ----------
    model : mOWL model

    Returns
    -------
    dict
        ``n_params_total``, ``n_params_class``, ``n_classes``.
    """
    module = getattr(model, "module", model)
    total = 0
    cls = 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        total += int(p.numel())
        if "class" in name:
            cls += int(p.numel())
    return {"n_params_total": total, "n_params_class": cls,
            "n_classes": int(len(model.class_index_dict))}


# --------------------------------------------------------------------------- training


def build_model(bundle, method: str, dim: int, seed: int, device: str, ckpt_path: Path,
                batch_size: Optional[int] = None):
    """Construct the model of one cell with the harness hyper-parameters.

    Parameters
    ----------
    bundle : DatasetBundle
    method : str
        ``elem``, ``elbe``, ``box2el`` (via :func:`experiments.run._build_model`) or
        ``transbox`` (mOWL defaults).
    dim : int
    seed : int
    device : str
    ckpt_path : Path
        Best-model checkpoint path.
    batch_size : int, optional
        Override of the batch size.

    Returns
    -------
    model
    """
    from experiments import run as harness

    if method == "transbox":
        from mowl.models import TransBox

        harness._seed_everything(seed)
        kw: Dict[str, Any] = {}
        if batch_size is not None:
            kw["batch_size"] = int(batch_size)
        return TransBox(bundle.dataset, embed_dim=int(dim), model_filepath=str(ckpt_path),
                        device=device, **kw)
    model, _ = harness._build_model(
        bundle=bundle,
        cm=None,
        method_id=method,
        dim=int(dim),
        seed=int(seed),
        batch_size=int(batch_size or DEFAULT_HYPERPARAMS["batch_size"]),
        learning_rate=float(DEFAULT_HYPERPARAMS["learning_rate"]),
        margin=float(DEFAULT_HYPERPARAMS["margin"]),
        delta=float(DEFAULT_HYPERPARAMS["delta"]),
        reg_factor=float(DEFAULT_HYPERPARAMS["reg_factor"]),
        num_negs=int(DEFAULT_HYPERPARAMS["num_negs"]),
        device=device,
        checkpoint_path=ckpt_path,
    )
    return model


def train_model(model, bundle, epochs: int, seed: int, ckpt_path: Path,
                validate_every: int = int(DEFAULT_HYPERPARAMS["validate_every"]),
                checkpoint: str = "last") -> Dict[str, Any]:
    """Train with the harness loop and return the model to evaluate.

    ``checkpoint="last"`` (default) evaluates the model after the last epoch: the
    capacity experiment asks how well a model represents the theory it was trained
    on, and the validation split of GO/FoodOn consists of held-out axioms, so the
    best-validation model is an early-stopped, under-fitted one (GO, Box²EL, d=32:
    validation loss is minimal near epoch 600 while the training loss still falls
    at epoch 3000). ``checkpoint="valid"`` restores the harness behaviour
    (best-validation checkpoint on subsumption datasets).

    Parameters
    ----------
    model : mOWL model
    bundle : DatasetBundle
    epochs : int
    seed : int
    ckpt_path : Path
    validate_every : int
    checkpoint : {"last", "valid"}

    Returns
    -------
    dict
        ``train_time_s``, ``mean_epoch_time_s``, ``validated`` (bool).
    """
    import torch

    from experiments import run as harness

    _ = model.training_datasets
    if checkpoint not in ("last", "valid"):
        raise ValueError(f"checkpoint must be 'last' or 'valid', got {checkpoint!r}")
    validated = (checkpoint == "valid" and bundle.task == "subsumption"
                 and bundle.dataset.validation is not None)
    if validated:
        model.eval_gci_name = str(DEFAULT_HYPERPARAMS["eval_gci_name"])
        torch.save(model.module.state_dict(), ckpt_path)
        eff = min(int(validate_every), max(1, int(epochs)))
    else:
        eff = max(int(validate_every), int(epochs) + 1)
    epoch_times: List[float] = []
    harness._seed_everything(seed)
    t0 = time.monotonic()
    harness._train_model_with_chunking(
        model=model,
        epochs=int(epochs),
        validate_every=eff,
        effective_batch_size=int(getattr(model, "batch_size", DEFAULT_HYPERPARAMS["batch_size"])),
        epoch_times=epoch_times,
    )
    dt = time.monotonic() - t0
    if validated and ckpt_path.exists():
        model.load_best_model()
    return {
        "train_time_s": dt,
        "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else None,
        "validated": bool(validated),
        "validate_every": int(eff),
    }


def run_cell(
    dataset: str,
    method: str,
    dim: int,
    seed: int,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
    out_dir: Path = DEFAULT_OUT,
    runs_dir: Path = RUNS_DIR / "capacity",
    deg_eps: float = 1e-3,
    bundle: Any = None,
    ref: Optional[Reference] = None,
    checkpoint: str = "last",
) -> Dict[str, Any]:
    """Train and evaluate one cell; write its JSON.

    Parameters
    ----------
    dataset, method : str
    dim, seed : int
    epochs : int, optional
        Default: :func:`default_epochs`.
    device : str, optional
        Default ``cuda`` when available, else ``cpu``.
    out_dir : Path
    runs_dir : Path
        Checkpoint directory.
    deg_eps : float
        Degeneracy threshold of the crisp evaluation.
    bundle, ref : optional
        Pre-loaded dataset / reference (reused across cells of one dataset).

    Returns
    -------
    dict
        Payload written to ``<out_dir>/<cell file>``.
    """
    import torch

    from experiments.metrics import _named_class_indices, modelness

    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {BASELINE_METHODS}")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = int(epochs if epochs is not None else default_epochs(dataset))
    t_all = time.monotonic()
    if bundle is None:
        bundle = load_capacity_dataset(dataset)
    runs_dir = Path(runs_dir).expanduser().resolve()
    if str(runs_dir).startswith("/tmp"):
        raise ValueError(f"refusing to write checkpoints under /tmp: {runs_dir}")
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt = runs_dir / f"{Path(cell_filename(dataset, method, dim, seed)).stem}.pt"

    t0 = time.monotonic()
    model = build_model(bundle, method, dim, seed, dev, ckpt)
    _ = model.training_datasets
    init_time = time.monotonic() - t0
    tr = train_model(model, bundle, n_epochs, seed, ckpt, checkpoint=checkpoint)

    t0 = time.monotonic()
    mod = modelness(model, margin=0.0)
    nc = int(len(model.class_index_dict))
    iris = [""] * nc
    for iri, idx in model.class_index_dict.items():
        iris[int(idx)] = str(iri)
    named = np.asarray(_named_class_indices(model, nc), dtype=np.int64)
    if ref is None:
        ref = reference_entailments(bundle, [iris[i] for i in named])
    elif list(ref.iris) != [iris[i] for i in named]:
        raise ValueError("reference does not match the model's named classes")
    kind, a, b = class_region_arrays(model)
    sat_idx = np.flatnonzero(ref.satisfiable)
    from experiments.capacity.common import restrict

    crisp = crisp_subsumption_metrics(kind, a, b, named[sat_idx], restrict(ref.matrix, sat_idx),
                                      deg_eps=deg_eps, device=dev)
    # second reading: every box/ball is a non-empty closed set (a zero-width box is a
    # lower-dimensional box, not the empty set), so no class is counted as unsatisfiable
    closed = crisp_subsumption_metrics(kind, a, b, named[sat_idx], restrict(ref.matrix, sat_idx),
                                       deg_eps=float("-inf"), device=dev)
    # third reading: closed sets with the training margin as containment slack
    train_margin = 0.1 if method == "transbox" else float(DEFAULT_HYPERPARAMS["margin"])
    tolerant = crisp_subsumption_metrics(kind, a, b, named[sat_idx],
                                         restrict(ref.matrix, sat_idx),
                                         deg_eps=float("-inf"), device=dev, tol=train_margin)
    sweep: Dict[str, Dict[str, Any]] = {}
    for t in TOL_SWEEP:
        if t == 0.0:
            r = closed
        elif t == train_margin:
            r = tolerant
        else:
            r = crisp_subsumption_metrics(kind, a, b, named[sat_idx],
                                          restrict(ref.matrix, sat_idx),
                                          deg_eps=float("-inf"), device=dev, tol=float(t))
        pr, rc = float(r["precision"]), float(r["recall"])
        sweep[f"{t:g}"] = {"precision": pr, "recall": rc, "n_pred": int(r["n_pred"]),
                           "f1": float(2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0}
    best_tol = max(sweep, key=lambda k: sweep[k]["f1"])
    try:
        mod_tm = float(modelness(model, margin=train_margin)["modelness_all"])
    except Exception:  # noqa: BLE001 -- optional diagnostic
        mod_tm = None
    deg_all = degenerate_mask(kind, a[torch.as_tensor(named)], b[torch.as_tensor(named)], deg_eps)
    eval_time = time.monotonic() - t0
    pc = parameter_counts(model)
    payload: Dict[str, Any] = {
        "config": {
            "dataset": dataset,
            "method": method,
            "model": type(model).__name__,
            "dim": int(dim),
            "seed": int(seed),
            "epochs": n_epochs,
            "device": dev,
            "deg_eps": float(deg_eps),
            "checkpoint": checkpoint,
            "region_kind": kind,
            "hyperparameters": (
                "mowl TransBox defaults" if method == "transbox" else dict(DEFAULT_HYPERPARAMS)
            ),
            "checkpoint_path": str(ckpt),
        },
        "modelness": mod,
        "modelness_all": float(mod["modelness_all"]),
        "crisp": crisp,
        "crisp_closed": closed,
        "precision_closed": closed["precision"],
        "recall_closed": closed["recall"],
        "crisp_tol": tolerant,
        "train_margin": train_margin,
        "precision_tol": tolerant["precision"],
        "recall_tol": tolerant["recall"],
        "modelness_train_margin": mod_tm,
        "tol_sweep": sweep,
        "best_tol": float(best_tol),
        "precision_best_tol": sweep[best_tol]["precision"],
        "recall_best_tol": sweep[best_tol]["recall"],
        "f1_best_tol": sweep[best_tol]["f1"],
        "precision": crisp["precision"],
        "recall": crisp["recall"],
        "precision_live": crisp["precision_live"],
        "recall_live": crisp["recall_live"],
        "false_pairs_per_class_live": crisp["false_pairs_per_class_live"],
        "n_named": int(len(named)),
        "n_named_satisfiable_ref": int(len(sat_idx)),
        "n_unsat_named": int(deg_all.sum().item()),
        "n_unsat_named_ref": int((~ref.satisfiable).sum()),
        "params": pc,
        "params_per_class": float(pc["n_params_class"] / max(pc["n_classes"], 1)),
        "timings": {
            "model_init_s": init_time,
            **tr,
            "eval_s": eval_time,
            "reference_s": ref.time_s,
            "wall_s": time.monotonic() - t_all,
        },
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(payload, Path(out_dir) / cell_filename(dataset, method, dim, seed))
    print(
        f"[baselines] {dataset} {method} d={dim} s={seed}: modelness={payload['modelness_all']:.4f} "
        f"P={crisp['precision']:.4f} R={crisp['recall']:.4f} "
        f"P_live={crisp['precision_live']:.4f} R_live={crisp['recall_live']:.4f} "
        f"P_closed={closed['precision']:.4f} R_closed={closed['recall']:.4f} "
        f"P_tol={tolerant['precision']:.4f} R_tol={tolerant['recall']:.4f} "
        f"best tol={best_tol}: P={sweep[best_tol]['precision']:.4f} "
        f"R={sweep[best_tol]['recall']:.4f} unsat={payload['n_unsat_named']} "
        f"params={pc['n_params_total']} t={payload['timings']['wall_s']:.1f}s",
        flush=True,
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser of B3."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cells", type=Path, default=None, help="Cells file (one cell per line).")
    p.add_argument("--start", type=int, default=0, help="First cell (0-based) of --cells.")
    p.add_argument("--count", type=int, default=None, help="Number of cells of --cells.")
    p.add_argument("--dataset", default=None)
    p.add_argument("--method", default=None, choices=BASELINE_METHODS)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--deg-eps", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR / "capacity")
    p.add_argument("--checkpoint", choices=("last", "valid"), default="last",
                   help="Model to evaluate: after the last epoch (default) or best validation.")
    p.add_argument("--skip-existing", action="store_true", help="Skip cells whose JSON exists.")
    p.add_argument("--write-cells", type=Path, default=None,
                   help="Write the full default grid to this file and exit.")
    p.add_argument("--prepare", nargs="+", default=None, metavar="DATASET",
                   help="Download the datasets and build their CanonicalModel caches, then exit.")
    return p


def prepare(datasets: Sequence[str]) -> None:
    """Download ``datasets`` and build their CanonicalModel pickle caches.

    Parameters
    ----------
    datasets : sequence of str
        Dataset identifiers.
    """
    from experiments.cm_cache import get_canonical_model

    for ds in dict.fromkeys(datasets):
        t0 = time.monotonic()
        get_canonical_model(load_capacity_dataset(ds))
        print(f"[baselines] prepared {ds} in {time.monotonic() - t0:.1f}s", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.write_cells is not None:
        n = write_cells_file(args.write_cells)
        print(f"[baselines] wrote {n} cells to {args.write_cells}")
        return 0
    if args.prepare is not None:
        prepare(args.prepare)
        return 0
    if args.cells is not None:
        cells = parse_cells(args.cells.read_text(encoding="utf-8").splitlines())
        end = len(cells) if args.count is None else args.start + args.count
        cells = cells[args.start:end]
    else:
        if args.dataset is None or args.method is None or args.dim is None:
            raise SystemExit("give --cells or --dataset/--method/--dim")
        cells = [(args.dataset, args.method, args.dim, args.seed)]
    cache: Dict[str, Any] = {}
    failed = []
    for ds, method, dim, seed in cells:
        if args.skip_existing and (Path(args.out) / cell_filename(ds, method, dim, seed)).exists():
            continue
        try:
            if ds not in cache:
                cache.clear()
                cache[ds] = load_capacity_dataset(ds)
            run_cell(ds, method, dim, seed, epochs=args.epochs, device=args.device,
                     out_dir=args.out, runs_dir=args.runs_dir, deg_eps=args.deg_eps,
                     bundle=cache[ds], checkpoint=args.checkpoint)
        except Exception:  # noqa: BLE001 -- report and continue with the other cells
            traceback.print_exc()
            failed.append((ds, method, dim, seed))
    if failed:
        print(f"[baselines] FAILED cells: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
