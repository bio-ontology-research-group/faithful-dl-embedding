#!/usr/bin/env python3
"""Single-cell synthetic runner for GroundedEL Experiment E1 (design §9 E1).

CLI usage
---------
::

    python -m experiments.grounded.run_synthetic \\
        --theory tk3 --K 4 --Kr 4 --dim 2 --seed 0 --epochs 3000 --lr 0.01 \\
        --out /mnt/data1/gfe/runs/grounded-e1/

Guarantees
----------
- Trains :class:`mowl.models.GroundedEL` from ``init="random"`` (strictly
  reasoner-free on the training path, R1).
- Resolves ``K = "full"`` (and ``Kr = "full"``) to ``model.nb_elements``
  (:math:`|\\Delta|`). When ``--Kr`` is omitted, defaults to ``--K``.
- Evaluates both :meth:`GroundedEL.crisp_report` and
  :meth:`GroundedEL.evaluate_model` (with ``use_reasoner=True`` at evaluation
  time only) and writes one JSON file atomically via ``os.replace``.
- Records ``config``, ``metrics`` (:meth:`evaluate_model`), ``crisp_report``,
  ``wall_time_s`` (plus ``train_time_s`` and ``eval_time_s``), ``final_losses``,
  and ``loss_curve`` sampled every 100 epochs.

Non-guarantees
--------------
- Gradient-based optimisation from random initialisation is non-convex and is
  not guaranteed to reach ``exact=True`` at insufficient capacity :math:`K`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import mowl
import numpy as np
import torch as th

from experiments.grounded.theories import THEORIES, load_theory_dataset

DEFAULT_E1_OUT_DIR = Path("/mnt/data1/gfe/runs/grounded-e1")


def parse_capacity_arg(raw: Union[str, int], name: str = "K") -> Union[str, int]:
    """Validate and parse a component capacity argument (positive int or ``'full'``).

    Parameters
    ----------
    raw : str or int
        Raw capacity value (e.g. ``"1"``, ``"4"``, ``"full"``, or ``4``).
    name : str, default "K"
        Parameter name for error messages.

    Returns
    -------
    Union[str, int]
        ``"full"`` or a positive integer ``>= 1``.

    Guarantees
    ----------
    - Rejects booleans, zero/negative integers, and unknown strings with
      :class:`ValueError`.

    Non-guarantees
    --------------
    - Does not resolve ``"full"`` to an integer until the dataset element count
      is known.
    """
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a positive integer or 'full', got {raw!r}")
    if isinstance(raw, int):
        if raw < 1:
            raise ValueError(f"{name} must be >= 1, got {raw}")
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "full":
            return "full"
        try:
            val = int(s)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a positive integer or 'full', got {raw!r}"
            ) from exc
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}")
        return val
    raise ValueError(f"{name} must be a positive integer or 'full', got {raw!r}")


def cell_filename(
    theory: str,
    k_spec: Union[str, int],
    dim: int,
    seed: int,
    kr_spec: Optional[Union[str, int]] = None,
    reseed_every: int = 100,
) -> str:
    """Return canonical E1 JSON filename ``{theory}__K{K}[__Kr{Kr}]__d{dim}__r{reseed}__s{seed}.json``."""
    k_str = str(k_spec)
    kr_str = str(k_spec if kr_spec is None else kr_spec)
    if kr_str != k_str:
        return (
            f"{theory}__K{k_str}__Kr{kr_str}__d{int(dim)}"
            f"__r{int(reseed_every)}__s{int(seed)}.json"
        )
    return f"{theory}__K{k_str}__d{int(dim)}__r{int(reseed_every)}__s{int(seed)}.json"


def resolve_out_path(
    out: Path | str,
    theory: str,
    k_spec: Union[str, int],
    dim: int,
    seed: int,
    kr_spec: Optional[Union[str, int]] = None,
    reseed_every: int = 100,
) -> Path:
    """Resolve ``out`` as either a direct ``.json`` file path or a directory containing :func:`cell_filename`."""
    p = Path(out).expanduser().resolve()
    if p.suffix.lower() == ".json":
        return p
    return p / cell_filename(
        theory, k_spec, dim, seed, kr_spec=kr_spec, reseed_every=reseed_every
    )


def resolve_save_model_path(
    save_model: Path | str,
    theory: str,
    k_spec: Union[str, int],
    dim: int,
    seed: int,
    kr_spec: Optional[Union[str, int]] = None,
    reseed_every: int = 100,
) -> Path:
    """Resolve ``save_model`` as either a direct checkpoint file path or a directory containing the ``.pt`` file for :func:`cell_filename`."""
    p = Path(save_model).expanduser().resolve()
    if p.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin"):
        return p
    json_name = cell_filename(
        theory, k_spec, dim, seed, kr_spec=kr_spec, reseed_every=reseed_every
    )
    pt_name = json_name[:-5] + ".pt" if json_name.endswith(".json") else json_name + ".pt"
    return p / pt_name


def subsample_loss_curve(
    history: Sequence[Dict[str, float]],
    every: int = 100,
) -> List[Dict[str, Union[int, float]]]:
    """Extract per-term loss records at epoch 1, every ``every`` epochs, and the final epoch."""
    if not history:
        return []
    step = max(1, int(every))
    n = len(history)
    curve: List[Dict[str, Union[int, float]]] = []
    for idx, rec in enumerate(history):
        ep = idx + 1
        if ep == 1 or ep % step == 0 or ep == n:
            entry: Dict[str, Union[int, float]] = {"epoch": int(ep)}
            for k, v in rec.items():
                entry[k] = float(v)
            curve.append(entry)
    return curve


def run_synthetic_cell(
    theory: str,
    K: Union[str, int],
    Kr: Optional[Union[str, int]] = None,
    dim: int = 2,
    seed: int = 0,
    epochs: int = 3000,
    lr: float = 0.01,
    reseed_every: int = 100,
    n_filler_samples: int = 64,
    out: Path | str = DEFAULT_E1_OUT_DIR,
    device: str = "cpu",
    n_eval: int = 10000,
    theory_dir: Optional[Path | str] = None,
    lr_final: Optional[float] = None,
    lr_schedule: str = "cosine",
    save_model: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Train and evaluate ``GroundedEL`` on synthetic ``theory`` and write the output JSON.

    Parameters
    ----------
    theory : str
        Synthetic theory identifier in ``THEORIES``.
    K : str or int
        Class component count (:math:`\\ge 1` or ``"full"``).
    Kr : str or int or None, optional
        Role component count (:math:`\\ge 1` or ``"full"``); defaults to ``K``.
    dim : int, default 2
        Embedding dimension :math:`d \\ge 1`.
    seed : int, default 0
        Random seed.
    epochs : int, default 3000
        Training epochs :math:`\\ge 0`.
    lr : float, default 0.01
        Adam learning rate :math:`> 0`.
    reseed_every : int, default 100
        Reseed interval in epochs (``0`` disables).
    n_filler_samples : int, default 64
        Maximum number of candidate filler elements per batch row in ``gci3`` / ``gci3_bot``.
    out : Path or str, default ``/mnt/data1/gfe/runs/grounded-e1``
        Output directory or explicit ``.json`` path.
    device : str, default "cpu"
        PyTorch device string.
    n_eval : int, default 10000
        Sample budget for ``crisp_report`` and ``evaluate_model``.
    theory_dir : Path or str or None, optional
        Directory for caching generated ``<theory>.owl`` files (default:
        ``$TMPDIR/gfe-grounded/``).
    lr_final : float or None, default None
        Optional final learning rate :math:`> 0` for decay; ``None`` keeps constant ``lr``.
    lr_schedule : {"cosine", "linear"}, default "cosine"
        Learning-rate decay schedule used when ``lr_final`` is set.
    save_model : Path or str or None, default None
        Optional output checkpoint path (or directory) where ``torch.save`` writes
        ``{"state_dict": model.module.state_dict(), "config": payload["config"]}``.

    Returns
    -------
    Dict[str, Any]
        Complete cell result dictionary written to disk.

    Guarantees
    ----------
    - Writes the JSON output atomically via ``os.replace``.
    - Uses ``init="random"`` and never invokes a reasoner during training.

    Non-guarantees
    --------------
    - Mutates global Python, NumPy, and PyTorch RNG states with ``seed``.
    """
    if theory not in THEORIES:
        raise ValueError(f"Unknown theory {theory!r}; expected one of {THEORIES}")
    k_spec = parse_capacity_arg(K, "K")
    kr_spec = k_spec if Kr is None else parse_capacity_arg(Kr, "Kr")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be an int >= 1, got {dim!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise ValueError(f"epochs must be an int >= 0, got {epochs!r}")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    if lr_final is not None:
        if (
            isinstance(lr_final, bool)
            or not isinstance(lr_final, (int, float))
            or float(lr_final) <= 0.0
        ):
            raise ValueError(f"lr_final must be > 0 or None, got {lr_final!r}")
    if not isinstance(lr_schedule, str) or lr_schedule not in ("cosine", "linear"):
        raise ValueError(
            f"lr_schedule must be 'cosine' or 'linear', got {lr_schedule!r}"
        )
    if isinstance(reseed_every, bool) or not isinstance(reseed_every, int) or reseed_every < 0:
        raise ValueError(f"reseed_every must be an int >= 0, got {reseed_every!r}")
    if (
        isinstance(n_filler_samples, bool)
        or not isinstance(n_filler_samples, int)
        or n_filler_samples < 1
    ):
        raise ValueError(
            f"n_filler_samples must be an int >= 1, got {n_filler_samples!r}"
        )

    out_path = resolve_out_path(
        out,
        theory,
        k_spec,
        dim,
        seed,
        kr_spec=kr_spec,
        reseed_every=reseed_every,
    )
    if str(out_path).startswith("/tmp/"):
        raise ValueError(f"Refusing to write under /tmp: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_path: Optional[Path] = None
    if save_model is not None:
        model_path = resolve_save_model_path(
            save_model,
            theory,
            k_spec,
            dim,
            seed,
            kr_spec=kr_spec,
            reseed_every=reseed_every,
        )
        if str(model_path).startswith("/tmp/"):
            raise ValueError(f"Refusing to write model under /tmp: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)

    mowl.init_jvm("4g")
    from mowl.models import GroundedEL

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    t0 = time.perf_counter()
    ds = load_theory_dataset(theory, out_dir=theory_dir)

    init_k = 1 if k_spec == "full" else int(k_spec)
    init_kr = 1 if kr_spec == "full" else int(kr_spec)

    model = GroundedEL(
        ds,
        embed_dim=int(dim),
        n_components=init_k,
        n_role_components=init_kr,
        epochs=int(epochs),
        learning_rate=float(lr),
        reseed_every=int(reseed_every),
        n_filler_samples=int(n_filler_samples),
        device=device,
        seed=int(seed),
        init="random",
        log_every=0,
    )
    if k_spec == "full" or kr_spec == "full":
        model.n_components = (
            int(model.nb_elements) if k_spec == "full" else int(k_spec)
        )
        model.n_role_components = (
            int(model.nb_elements) if kr_spec == "full" else int(kr_spec)
        )
        model.init_module()

    t_train_start = time.perf_counter()
    history = model.train(
        epochs=int(epochs),
        lr_final=None if lr_final is None else float(lr_final),
        lr_schedule=str(lr_schedule),
    )
    train_time_s = float(time.perf_counter() - t_train_start)

    t_eval_start = time.perf_counter()
    crisp = model.crisp_report(n_eval=int(n_eval))
    eval_metrics = model.evaluate_model(use_reasoner=True, n_eval=int(n_eval))
    eval_time_s = float(time.perf_counter() - t_eval_start)
    wall_time_s = float(time.perf_counter() - t0)

    if history:
        final_losses: Dict[str, float] = {
            k: float(v) for k, v in history[-1].items()
        }
    else:
        with th.no_grad():
            raw_losses = model.compute_batch_losses()
        final_losses = {
            k: float(v.detach().item()) for k, v in raw_losses.items()
        }
        final_losses["tau"] = float(model.module.tau.item())

    loss_curve = subsample_loss_curve(history, every=100)
    param_count = int(sum(p.numel() for p in model.module.parameters()))

    # Convert tuple in eval_metrics["faithful_named"] to list for JSON serialisation
    serialisable_metrics: Dict[str, Any] = {}
    for mk, mv in eval_metrics.items():
        if isinstance(mv, tuple):
            serialisable_metrics[mk] = list(mv)
        else:
            serialisable_metrics[mk] = mv

    payload: Dict[str, Any] = {
        "config": {
            "theory": theory,
            "method": f"grounded-k{k_spec}",
            "K": k_spec,
            "Kr": kr_spec,
            "n_components": int(model.n_components),
            "n_role_components": int(model.n_role_components),
            "nb_elements": int(model.nb_elements),
            "nb_classes": int(model.nb_classes),
            "nb_rels": int(model.nb_rels),
            "nb_inds": int(model.nb_inds),
            "dim": int(dim),
            "seed": int(seed),
            "epochs": int(epochs),
            "lr": float(lr),
            "lr_final": None if lr_final is None else float(lr_final),
            "lr_schedule": str(lr_schedule),
            "reseed_every": int(reseed_every),
            "n_filler_samples": int(n_filler_samples),
            "init": "random",
            "device": str(device),
            "param_count": param_count,
        },
        "metrics": serialisable_metrics,
        "crisp_report": {
            "elem_viol": float(crisp["elem_viol"]),
            "region_viol": float(crisp["region_viol"]),
            "worst_depth": crisp["worst_depth"],
            "n_axioms_violated_by_depth": dict(crisp["n_axioms_violated_by_depth"]),
        },
        "worst_depth": crisp["worst_depth"],
        "n_axioms_violated_by_depth": dict(crisp["n_axioms_violated_by_depth"]),
        "wall_time_s": wall_time_s,
        "timings": {
            "wall_time_s": wall_time_s,
            "train_time_s": train_time_s,
            "eval_time_s": eval_time_s,
        },
        "final_losses": final_losses,
        "loss_curve": loss_curve,
    }

    if model_path is not None:
        tmp_model_path = (
            model_path.parent / f".{model_path.name}.{os.getpid()}.tmp"
        )
        th.save(
            {
                "state_dict": model.module.state_dict(),
                "config": dict(payload["config"]),
            },
            tmp_model_path,
        )
        os.replace(tmp_model_path, model_path)

    tmp_path = out_path.parent / f".{out_path.name}.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``run_synthetic.py``."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate GroundedEL on a synthetic E1 cell."
    )
    parser.add_argument(
        "--theory",
        type=str,
        required=True,
        choices=list(THEORIES),
        help="Synthetic theory identifier.",
    )
    parser.add_argument(
        "--K",
        type=str,
        required=True,
        help="Class box component count (positive int or 'full').",
    )
    parser.add_argument(
        "--Kr",
        type=str,
        default=None,
        help="Role product-box component count (positive int or 'full'; default: same as --K).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Embedding dimension d (default: 2).",
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
        default=3000,
        help="Number of training epochs (default: 3000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Final Adam learning rate for decay; None keeps constant lr (default: None).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("cosine", "linear"),
        help="Learning rate decay schedule when --lr-final is set (default: cosine).",
    )
    parser.add_argument(
        "--reseed-every",
        type=int,
        default=100,
        help="Reseed interval in epochs; 0 disables reseeding (default: 100).",
    )
    parser.add_argument(
        "--n-filler-samples",
        type=int,
        default=64,
        help="Max filler samples per batch in gci3/gci3_bot (default: 64).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_E1_OUT_DIR,
        help="Output directory or explicit .json path (default: /mnt/data1/gfe/runs/grounded-e1/).",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional output checkpoint path or directory for saving module state_dict + config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (default: cpu).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point for ``run_synthetic.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    res = run_synthetic_cell(
        theory=args.theory,
        K=args.K,
        Kr=args.Kr,
        dim=args.dim,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        lr_final=args.lr_final,
        lr_schedule=args.lr_schedule,
        reseed_every=args.reseed_every,
        n_filler_samples=args.n_filler_samples,
        out=args.out,
        save_model=args.save_model,
        device=args.device,
    )
    cfg = res["config"]
    m = res["metrics"]
    print(
        f"[{cfg['theory']} K={cfg['K']} (eff={cfg['n_components']}) d={cfg['dim']} "
        f"r={cfg['reseed_every']} s={cfg['seed']}] "
        f"exact={m['exact']} exact_tol1e-4={m['exact_tol1e-4']} "
        f"elem_viol={m['elem_viol']:.4f} region_viol={m['region_viol']:.4f} "
        f"region_viol_tol1e-4={m['region_viol_tol1e-4']:.4f} "
        f"exist={m['exist']:.4f} dead_ok={m['dead_ok']} "
        f"faith=({m['faithful_named_precision']:.4f}, {m['faithful_named_recall']:.4f}) "
        f"wall={res['wall_time_s']:.2f}s"
    )
    return res


if __name__ == "__main__":
    main()
