"""Training CLI for DeepGO-SE-style inductive protein function prediction with GroundedEL (E3).

Each optimisation step samples one protein mini-batch (binary cross-entropy via
:meth:`GroundedEL.compute_data_loss`) and, when ``lambda_tbox > 0``, one TBox
mini-batch (:meth:`GroundedEL.compute_batch_losses`, weighted by ``lambda_tbox``).
Optimises two Adam parameter groups (encoder + ``raw_tau_s`` at ``lr_encoder``;
boxes/positions at ``lr_boxes`` with cosine decay to ``lr_final``) while annealing
:math:`\\tau` geometrically from ``tau0`` to ``tau1`` as :meth:`GroundedEL.train`
does. Evaluates validation BCE and validation Fmax every epoch, checkpoints by
validation BCE (DeepGO-SE's rule), records the best-Fmax epoch, and saves test
predictions in DeepGO-SE's DataFrame pickle format alongside ``metrics.json``.

Guarantees
----------
- Reasoner-free (R1): never imports or calls any OWL reasoner or
  :class:`~mowl.geometric.completion.CanonicalModel`.
- Never writes temporary files under ``/tmp``.
- Checkpoints the model parameters achieving the lowest validation BCE and
  restores them before generating ``test_predictions.pkl``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F

from experiments.deepgose.data import DeepGOSESplitData, load_deepgose_data
from experiments.deepgose.model import (
    DeepGOSEGroundedModel,
    build_deepgose_grounded_model,
    build_optimizer_groups,
)


def compute_fast_fmax(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 101,
) -> Tuple[float, float]:
    """Compute protein-centric Fmax and best threshold over ``0.00 .. 1.00`` on multi-hot matrices.

    Matches DeepGO-SE's ``evaluate_annotations`` formula restricted to the
    evaluated term columns: at threshold :math:`t`, precision is averaged over
    proteins with at least one predicted positive (:math:`\\text{score} \\ge t`),
    recall is averaged over proteins with at least one ground-truth positive,
    and :math:`F_1(t) = 2 P(t) R(t) / (P(t) + R(t))`.

    Guarantees
    ----------
    - Returns ``(fmax, best_threshold)`` with ``0.0 <= fmax <= 1.0``.
    """
    if scores.size == 0 or labels.size == 0:
        return 0.0, 0.0
    label_bool = labels > 0.5
    true_counts = label_bool.sum(axis=1)  # (N,)
    has_true = true_counts > 0
    if not np.any(has_true):
        return 0.0, 0.0

    best_fmax = 0.0
    best_thresh = 0.0
    for k in range(n_thresholds):
        t = k / float(n_thresholds - 1)
        pred_bool = scores >= t
        pred_counts = pred_bool.sum(axis=1)
        tp = (pred_bool & label_bool).sum(axis=1)

        has_pred = pred_counts > 0
        if np.any(has_pred):
            avg_prec = float(np.mean(tp[has_pred] / pred_counts[has_pred]))
        else:
            avg_prec = 0.0

        avg_rec = float(np.mean(tp[has_true] / true_counts[has_true]))
        if avg_prec + avg_rec > 0.0:
            f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)
            if f1 > best_fmax:
                best_fmax = float(f1)
                best_thresh = float(t)
    return best_fmax, best_thresh


@th.no_grad()
def predict_in_batches(
    model: DeepGOSEGroundedModel,
    features: th.Tensor,
    batch_size: int = 256,
    max_elements: int = 8_388_608,
) -> th.Tensor:
    """Compute raw score matrix ``(N, n_terms)`` on CPU in protein batches."""
    model.module.eval()
    n = int(features.shape[0])
    n_terms = int(model.term_class_ids.numel())
    if n == 0:
        return th.empty((0, n_terms), dtype=th.float32)
    parts: List[th.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s_batch = model.score(features[start:end], max_elements=max_elements)
        parts.append(s_batch.detach().cpu())
    return th.cat(parts, dim=0)


def run_training(
    ont: str = "cc",
    seed: int = 0,
    model_id: int = 0,
    model_name: str = "grounded_esm",
    dim: int = 128,
    K: int = 1,
    lambda_tbox: float = 1.0,
    epochs: int = 64,
    protein_batch: int = 128,
    tbox_batch: int = 4096,
    lr_encoder: float = 5e-4,
    lr_boxes: float = 1e-2,
    lr_final: float = 1e-4,
    device: str = "cpu",
    data_root: Union[str, Path] = "/mnt/data1/gfe/data/deepgo2/data",
    out_dir: Union[str, Path] = "/mnt/data/u/hohndor/gfe/runs/deepgose",
    max_train_proteins: Optional[int] = None,
    min_term_annotations: Optional[int] = None,
    predict_only: bool = False,
    tbox_every: int = 1,
) -> Dict[str, Any]:
    """Run DeepGO-SE GroundedEL training and write checkpoint, predictions, and metrics JSON."""
    if ont not in ("mf", "bp", "cc"):
        raise ValueError(f"ont must be one of {{'mf', 'bp', 'cc'}}, got {ont!r}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if protein_batch < 1:
        raise ValueError(f"protein_batch must be >= 1, got {protein_batch}")
    if tbox_batch < 1:
        raise ValueError(f"tbox_batch must be >= 1, got {tbox_batch}")
    if tbox_every < 1:
        raise ValueError(f"tbox_every must be >= 1, got {tbox_every}")

    eff_device = device
    if eff_device.startswith("cuda") and not th.cuda.is_available():
        eff_device = "cpu"

    out_ont_dir = Path(out_dir) / ont
    out_ont_dir.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    split_data: DeepGOSESplitData = load_deepgose_data(
        data_root=data_root,
        ont=ont,
        cache_dir=out_ont_dir,
        max_train_proteins=max_train_proteins,
        min_term_annotations=min_term_annotations,
    )
    load_wall_sec = float(time.perf_counter() - t_load0)

    eff_seed = int(seed)
    th.manual_seed(eff_seed)
    np.random.seed(eff_seed)

    in_dim = (
        int(split_data.train_features.shape[1])
        if split_data.train_features.ndim == 2 and split_data.train_features.shape[0] > 0
        else 2560
    )
    model = build_deepgose_grounded_model(
        tables=split_data.tables,
        embed_dim=dim,
        n_components=K,
        n_role_components=K,
        lambda_tbox=lambda_tbox,
        tbox_batch_size=tbox_batch,
        lr_boxes=lr_boxes,
        in_dim=in_dim,
        seed=eff_seed,
        device=eff_device,
    )
    optimizer, _, _ = build_optimizer_groups(
        model, lr_encoder=lr_encoder, lr_boxes=lr_boxes
    )

    n_train = int(split_data.train_features.shape[0])
    steps_per_epoch = max(1, math.ceil(n_train / protein_batch))
    total_steps = max(1, epochs * steps_per_epoch)

    prot_gen = th.Generator(device="cpu")
    prot_gen.manual_seed(eff_seed + 4242)
    model.grounded.reset_generator(eff_seed + 1337)

    ckpt_path = out_ont_dir / f"{model_name}_{model_id}.pt"
    best_val_bce = float("inf")
    best_val_bce_epoch = 0
    best_val_fmax = -1.0
    best_val_fmax_epoch = 0
    best_state_dict: Optional[Dict[str, th.Tensor]] = None

    epoch_records: List[Dict[str, Any]] = []
    valid_labels_np = split_data.valid_labels.numpy()

    epochs_to_run = epochs
    if predict_only:
        # Recover a run from its best-validation-BCE checkpoint (e.g. after the job
        # was killed before it wrote predictions): no training, same outputs.
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--predict-only needs the checkpoint {ckpt_path}")
        ckpt = th.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_terms = [str(t) for t in ckpt.get("terms", [])]
        if ckpt_terms and ckpt_terms != [str(t) for t in split_data.terms]:
            raise ValueError(
                f"checkpoint terms ({len(ckpt_terms)}) differ from data terms ({len(split_data.terms)})"
            )
        best_state_dict = {k: v.detach().cpu().clone() for k, v in ckpt["state_dict"].items()}
        best_val_bce = float(ckpt.get("best_val_bce", float("nan")))
        best_val_bce_epoch = int(ckpt.get("best_val_bce_epoch", 0))
        model.module.load_state_dict({k: v.to(eff_device) for k, v in best_state_dict.items()})
        val_scores_cpu = predict_in_batches(
            model, split_data.valid_features, batch_size=max(protein_batch, 256)
        )
        best_val_fmax, _ = compute_fast_fmax(val_scores_cpu.numpy(), valid_labels_np)
        best_val_fmax_epoch = best_val_bce_epoch
        epochs_to_run = 0
        print(
            f"[predict-only] loaded {ckpt_path} (best_val_bce={best_val_bce:.5f} @ ep {best_val_bce_epoch}, "
            f"val_fmax of checkpoint={best_val_fmax:.4f})",
            flush=True,
        )

    train_t0 = time.perf_counter()
    global_step = 0

    for epoch in range(epochs_to_run):
        ep_t0 = time.perf_counter()
        model.module.train()

        # Cosine LR decay on the box group (group 1); encoder group (group 0) stays at lr_encoder
        cur_box_lr = model.grounded.scheduled_lr(
            epoch, epochs, lr=lr_boxes, lr_final=lr_final, lr_schedule="cosine"
        )
        optimizer.param_groups[0]["lr"] = float(lr_encoder)
        optimizer.param_groups[1]["lr"] = float(cur_box_lr)

        # Geometric tau annealing on the TBox memberships (as GroundedEL.train does)
        cur_tau = model.grounded.step_tau(epoch, epochs)

        perm = th.randperm(n_train, generator=prot_gen)
        bce_sum = 0.0
        tbox_sum = 0.0
        total_sum = 0.0
        n_batches = 0
        n_tbox_batches = 0

        for start in range(0, n_train, protein_batch):
            end = min(start + protein_batch, n_train)
            batch_idx = perm[start:end]
            feat_b = split_data.train_features[batch_idx]
            lbl_b = split_data.train_labels[batch_idx]

            bce_loss, _ = model.compute_data_loss(feat_b, lbl_b)
            # TBox step every ``tbox_every`` protein batches (a larger, rarer TBox batch
            # amortises the per-step kernel-launch overhead; same rows per epoch when
            # tbox_batch scales with tbox_every).
            if lambda_tbox > 0.0 and (global_step % tbox_every == 0):
                tbox_losses = model.grounded.compute_batch_losses()
                tbox_total = tbox_losses["total"]
                step_loss = bce_loss + lambda_tbox * tbox_total
                tbox_val = float(tbox_total.detach().item())
                tbox_sum += tbox_val
                n_tbox_batches += 1
            else:
                step_loss = bce_loss
                tbox_val = 0.0

            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()
            model.grounded.clamp_dead_logits()

            bce_sum += float(bce_loss.detach().item())
            total_sum += float(step_loss.detach().item())
            n_batches += 1
            global_step += 1

        # Validation pass (BCE and Fmax)
        val_scores_cpu = predict_in_batches(
            model, split_data.valid_features, batch_size=max(protein_batch, 256)
        )
        val_bce = float(
            F.binary_cross_entropy(
                th.clamp(val_scores_cpu, min=1e-7, max=1.0 - 1e-7),
                split_data.valid_labels,
                reduction="mean",
            ).item()
        )
        val_fmax, val_thresh = compute_fast_fmax(
            val_scores_cpu.numpy(), valid_labels_np
        )
        ep_sec = float(time.perf_counter() - ep_t0)

        if val_bce < best_val_bce:
            best_val_bce = val_bce
            best_val_bce_epoch = epoch + 1
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.module.state_dict().items()
            }
            tmp_ckpt = ckpt_path.with_suffix(f".pt.tmp.{os.getpid()}")
            th.save(
                {
                    "state_dict": best_state_dict,
                    "ont": ont,
                    "seed": eff_seed,
                    "model_id": model_id,
                    "model_name": model_name,
                    "dim": dim,
                    "K": K,
                    "lambda_tbox": lambda_tbox,
                    "best_val_bce": best_val_bce,
                    "best_val_bce_epoch": best_val_bce_epoch,
                    "terms": split_data.terms,
                },
                tmp_ckpt,
            )
            tmp_ckpt.replace(ckpt_path)

        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_val_fmax_epoch = epoch + 1

        rec = {
            "epoch": epoch + 1,
            "train_bce": bce_sum / max(n_batches, 1),
            "train_tbox": tbox_sum / max(n_tbox_batches, 1),
            "train_total": total_sum / max(n_batches, 1),
            "val_bce": val_bce,
            "val_fmax": val_fmax,
            "val_best_threshold": val_thresh,
            "tau": float(cur_tau),
            "tau_s": float(model.tau_s.detach().item()),
            "lr_encoder": float(lr_encoder),
            "lr_boxes": float(cur_box_lr),
            "epoch_sec": ep_sec,
        }
        epoch_records.append(rec)
        print(
            f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
            f"train_bce={rec['train_bce']:.5f} train_tbox={rec['train_tbox']:.5f} "
            f"val_bce={val_bce:.5f} (best={best_val_bce:.5f} @ ep {best_val_bce_epoch}) "
            f"val_fmax={val_fmax:.4f} (best={best_val_fmax:.4f} @ ep {best_val_fmax_epoch}) "
            f"tau={cur_tau:.4f} tau_s={rec['tau_s']:.4f} time={ep_sec:.2f}s",
            flush=True,
        )

    train_wall_sec = float(time.perf_counter() - train_t0)
    avg_epoch_sec = float(train_wall_sec / max(epochs, 1))

    # Restore best validation BCE checkpoint before generating test predictions
    if best_state_dict is not None:
        model.module.load_state_dict(
            {k: v.to(eff_device) for k, v in best_state_dict.items()}
        )

    test_scores_np = predict_in_batches(
        model, split_data.test_features, batch_size=max(protein_batch, 256)
    ).numpy()

    out_test_df = split_data.test_df.copy()
    out_test_df["preds"] = list(test_scores_np)

    pred_id_path = out_ont_dir / f"test_predictions_{model_name}_{model_id}.pkl"
    pred_default_path = out_ont_dir / "test_predictions.pkl"
    for p_target in (pred_id_path, pred_default_path):
        tmp_p = p_target.with_suffix(f".pkl.tmp.{os.getpid()}")
        out_test_df.to_pickle(tmp_p)
        tmp_p.replace(p_target)

    metrics_payload: Dict[str, Any] = {
        "ont": ont,
        "seed": eff_seed,
        "model_id": int(model_id),
        "model_name": model_name,
        "dim": int(dim),
        "K": int(K),
        "lambda_tbox": float(lambda_tbox),
        "epochs": int(epochs),
        "predict_only": bool(predict_only),
        "protein_batch": int(protein_batch),
        "tbox_batch": int(tbox_batch),
        "tbox_every": int(tbox_every),
        "lr_encoder": float(lr_encoder),
        "lr_boxes": float(lr_boxes),
        "lr_final": float(lr_final),
        "device": eff_device,
        "n_train_proteins": n_train,
        "n_valid_proteins": int(split_data.valid_features.shape[0]),
        "n_test_proteins": int(split_data.test_features.shape[0]),
        "n_terms": len(split_data.terms),
        "nb_classes": int(model.grounded.nb_classes),
        "nb_rels": int(model.grounded.nb_rels),
        "axiom_counts": {
            k: int(len(ds)) for k, ds in model.grounded.training_datasets.items()
        },
        "best_val_bce": float(best_val_bce),
        "best_val_bce_epoch": int(best_val_bce_epoch),
        "best_val_fmax": float(best_val_fmax),
        "best_val_fmax_epoch": int(best_val_fmax_epoch),
        "final_tau_s": float(model.tau_s.detach().item()),
        "load_wall_sec": load_wall_sec,
        "train_wall_sec": train_wall_sec,
        "seconds_per_epoch": avg_epoch_sec,
        "checkpoint_path": str(ckpt_path),
        "predictions_path": str(pred_id_path),
        "history": epoch_records,
    }

    metrics_id_path = out_ont_dir / f"metrics_{model_name}_{model_id}.json"
    metrics_default_path = out_ont_dir / "metrics.json"
    for m_target in (metrics_id_path, metrics_default_path):
        tmp_m = m_target.with_suffix(f".json.tmp.{os.getpid()}")
        with open(tmp_m, "w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp_m.replace(m_target)

    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``experiments.deepgose.train``."""
    parser = argparse.ArgumentParser(
        description="Train GroundedEL with ESM2 encoder on DeepGO-SE benchmark (E3)."
    )
    parser.add_argument(
        "--ont",
        "-ont",
        choices=["mf", "bp", "cc"],
        default="cc",
        help="GO sub-ontology (mf, bp, cc).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--model-id",
        "-mi",
        type=int,
        default=0,
        help="Model index for ensemble runs (default: 0).",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default=None,
        help="Model name prefix (default: grounded_esm if lambda_tbox > 0 else grounded_esm_l0).",
    )
    parser.add_argument(
        "--dim",
        "--d",
        dest="dim",
        type=int,
        default=128,
        help="Embedding dimension d (default: 128).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="Number of boxes K per class/role (default: 1).",
    )
    parser.add_argument(
        "--lambda-tbox",
        type=float,
        default=1.0,
        help="Weight on GroundedEL TBox loss (default: 1.0; 0.0 for box-head MLP ablation).",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=64,
        help="Number of training epochs (default: 64).",
    )
    parser.add_argument(
        "--protein-batch",
        "-bs",
        type=int,
        default=128,
        help="Protein mini-batch size (default: 128).",
    )
    parser.add_argument(
        "--tbox-batch",
        type=int,
        default=4096,
        help="TBox mini-batch size per axiom type (default: 4096).",
    )
    parser.add_argument(
        "--tbox-every",
        type=int,
        default=1,
        help="Run the GroundedEL TBox step every N protein batches (default: 1).",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=5e-4,
        help="Adam learning rate for encoder and tau_s (default: 5e-4).",
    )
    parser.add_argument(
        "--lr-boxes",
        type=float,
        default=1e-2,
        help="Initial Adam learning rate for boxes and element positions (default: 1e-2).",
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=1e-4,
        help="Final cosine-decayed learning rate for boxes and element positions (default: 1e-4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda; falls back to cpu if unavailable).",
    )
    parser.add_argument(
        "--data-root",
        "-dr",
        type=Path,
        default=Path("/mnt/data1/gfe/data/deepgo2/data"),
        help="Root data directory containing {mf,bp,cc}/, go.norm, go.obo.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/data/u/hohndor/gfe/runs/deepgose"),
        help="Output directory for checkpoints, predictions, and metrics.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training; load {out}/{ont}/{model_name}_{model_id}.pt and write predictions and metrics.",
    )
    parser.add_argument(
        "--max-train-proteins",
        type=int,
        default=None,
        help="Optional cap on training proteins (used only for smoke/tests).",
    )
    parser.add_argument(
        "--min-term-annotations",
        type=int,
        default=None,
        help="Optional minimum positive training annotations per term (used only for smoke/tests).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``experiments.deepgose.train``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    eff_model_name = args.model_name
    if eff_model_name is None:
        eff_model_name = (
            "grounded_esm" if args.lambda_tbox > 0.0 else "grounded_esm_l0"
        )

    res = run_training(
        ont=args.ont,
        seed=args.seed,
        model_id=args.model_id,
        model_name=eff_model_name,
        dim=args.dim,
        K=args.K,
        lambda_tbox=args.lambda_tbox,
        epochs=args.epochs,
        protein_batch=args.protein_batch,
        tbox_batch=args.tbox_batch,
        tbox_every=args.tbox_every,
        lr_encoder=args.lr_encoder,
        lr_boxes=args.lr_boxes,
        lr_final=args.lr_final,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out,
        max_train_proteins=args.max_train_proteins,
        min_term_annotations=args.min_term_annotations,
        predict_only=args.predict_only,
    )
    print(
        f"[TRAIN DONE] ont={res['ont']} model={res['model_name']}_{res['model_id']} "
        f"best_val_bce={res['best_val_bce']:.5f}@ep{res['best_val_bce_epoch']} "
        f"best_val_fmax={res['best_val_fmax']:.4f}@ep{res['best_val_fmax_epoch']} "
        f"sec/epoch={res['seconds_per_epoch']:.2f}s total={res['train_wall_sec']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
