"""Run GroundedEL on the Gene Ontology (GO) subsumption benchmark (design §9 E2).

Trains :class:`~mowl.models.GroundedEL` on :class:`~mowl.datasets.builtin.GOSubsumptionDataset`
(loaded via :func:`experiments.datasets.load_dataset` for ``"go"``) and writes a
single JSON report containing:

* ``config`` (CLI parameters, element/class/role counts, axiom counts),
* ``crisp_report`` (``elem_viol`` and ``region_viol`` on :math:`10^4` samples),
* ``evaluate_model`` (``use_reasoner=False`` metrics: ``exist``, ``dead_ok``, ``exact``),
* ``test_ranking`` (subsumption ranking on the held-out GO test split using
  score :math:`m(p_{x_A}, B)`: MR, MRR, Hits@1, Hits@3, Hits@10, Hits@50,
  Hits@100, and ROC-AUC, both unfiltered and filtered),
* ``final_losses``, ``loss_curve`` (every 100 epochs), and ``wall_time_sec``.

Guarantees
----------
- Reasoner-free (R1): never imports or calls any OWL reasoner or
  :class:`~mowl.geometric.completion.CanonicalModel`.
- Never writes temporary files under ``/tmp``.
- Writes the output JSON atomically via a temporary sibling file.

Non-guarantees
--------------
- Does not compute reasoner-based faithfulness on GO (``use_reasoner=False``).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th


def _validate_args(args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments for ``run_go``.

    Guarantees
    ----------
    - Raises :class:`ValueError` if any numeric hyperparameter is out of domain.

    Non-guarantees
    --------------
    - Does not check GPU availability before PyTorch initialisation.
    """
    if args.K < 1:
        raise ValueError(f"--K must be >= 1, got {args.K}")
    if args.Kr is not None and args.Kr < 1:
        raise ValueError(f"--Kr must be >= 1 when specified, got {args.Kr}")
    if args.d < 1:
        raise ValueError(f"--d / --dim must be >= 1, got {args.d}")
    if args.eps_min < 0.0:
        raise ValueError(f"--eps-min must be >= 0.0, got {args.eps_min}")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if args.n_eval < 1:
        raise ValueError(f"--n-eval must be >= 1, got {args.n_eval}")
    if args.eval_batch_size < 1:
        raise ValueError(f"--eval-batch-size must be >= 1, got {args.eval_batch_size}")
    if args.max_test_tuples is not None and args.max_test_tuples < 1:
        raise ValueError(
            f"--max-test-tuples must be >= 1 when specified, got {args.max_test_tuples}"
        )
    if args.reseed_every < 0:
        raise ValueError(f"--reseed-every must be >= 0, got {args.reseed_every}")
    if args.n_filler_samples < 1:
        raise ValueError(
            f"--n-filler-samples must be >= 1, got {args.n_filler_samples}"
        )


def _extract_subclass_pairs(
    ontology: Any,
    class_to_eval_idx: Dict[str, int],
) -> np.ndarray:
    """Extract ``(head_eval_idx, tail_eval_idx)`` pairs of named ``SubClassOf`` axioms.

    Matches :meth:`mowl.evaluation.SubsumptionEvaluator.create_tuples`.

    Guarantees
    ----------
    - Returns a 2-D ``np.ndarray`` of shape ``(N, 2)`` and dtype ``int64``.

    Non-guarantees
    --------------
    - Complex class expressions (non-named classes) are ignored, matching
      :class:`~mowl.evaluation.SubsumptionEvaluator`.
    """
    from org.semanticweb.owlapi.model import AxiomType, ClassExpressionType

    pairs: List[Tuple[int, int]] = []
    if ontology is None:
        return np.zeros((0, 2), dtype=np.int64)

    for axiom in ontology.getAxioms():
        if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
            continue
        sub = axiom.getSubClass()
        sup = axiom.getSuperClass()
        if (
            sub.getClassExpressionType() == ClassExpressionType.OWL_CLASS
            and sup.getClassExpressionType() == ClassExpressionType.OWL_CLASS
        ):
            sub_iri = str(sub.toStringID())
            sup_iri = str(sup.toStringID())
            if sub_iri in class_to_eval_idx and sup_iri in class_to_eval_idx:
                pairs.append((class_to_eval_idx[sub_iri], class_to_eval_idx[sup_iri]))

    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def evaluate_subsumption_ranking(
    model: Any,
    dataset: Any,
    eval_batch_size: int = 64,
    max_test_tuples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate test-split subsumption ranking using score :math:`m(p_{x_A}, B)`.

    Since :class:`~mowl.evaluation.SubsumptionEvaluator` hardcodes
    ``model(batch, "gci0")`` with ascending sort and multiplicative ``10000``
    filtering (designed for non-negative losses rather than membership scores
    :math:`m(p_{x_A}, B) \\in [0, 1]`), this function extracts the exact same
    evaluation classes and test/filter tuples and computes both unfiltered and
    filtered MR, MRR, Hits@{1, 3, 10, 50, 100}, and ROC-AUC (via
    :func:`~mowl.evaluation.base.compute_rank_roc`).

    Guarantees
    ----------
    - Scores every candidate class :math:`B` for query head :math:`A` via
      :meth:`~mowl.nn.el.grounded.GroundedELModule.membership` at
      :math:`p_{x_A} = \\text{positions}(A)`, resolving floating-point sigmoid
      saturation ties via :meth:`~mowl.nn.el.grounded.GroundedELModule.depth`.
    - Returns a dictionary of float metrics including ``mrr``, ``hits@1``,
      ``hits@10``, ``hits@100``, ``auc``, and their ``f_`` filtered counterparts.

    Non-guarantees
    --------------
    - If ``dataset.testing`` has zero named ``SubClassOf`` axioms, returns zeros.
    """
    from mowl.evaluation.base import compute_rank_roc

    mod = model.module
    device = model.device

    # Named evaluation classes in model.named_class_index_dict order (0 .. n_eval_classes - 1)
    named_dict = getattr(model, "named_class_index_dict", model.class_index_dict)
    eval_iris = list(named_dict.keys())
    n_eval_classes = len(eval_iris)
    if n_eval_classes == 0:
        raise RuntimeError("No named classes found in model.named_class_index_dict.")

    class_to_eval_idx = {iri: idx for idx, iri in enumerate(eval_iris)}
    eval_to_model_idx = th.tensor(
        [model.class_index_dict[iri] for iri in eval_iris],
        dtype=th.long,
        device=device,
    )
    model_to_eval_idx: Dict[int, int] = {
        int(model.class_index_dict[iri]): idx for idx, iri in enumerate(eval_iris)
    }

    def _pairs_from_gci0_tensor(gci0_ds: Any) -> np.ndarray:
        if gci0_ds is None or len(gci0_ds) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        arr = gci0_ds[:].detach().cpu().numpy()
        valid_pairs: List[Tuple[int, int]] = []
        for row in arr:
            h_m, t_m = int(row[0]), int(row[1])
            if h_m in model_to_eval_idx and t_m in model_to_eval_idx:
                valid_pairs.append((model_to_eval_idx[h_m], model_to_eval_idx[t_m]))
        if not valid_pairs:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(valid_pairs, dtype=np.int64)

    test_ds_dict = getattr(model, "testing_datasets", None)
    if test_ds_dict and "gci0" in test_ds_dict and len(test_ds_dict["gci0"]) > 0:
        test_tuples = _pairs_from_gci0_tensor(test_ds_dict["gci0"])
    else:
        test_tuples = _extract_subclass_pairs(dataset.testing, class_to_eval_idx)

    if max_test_tuples is not None and test_tuples.shape[0] > max_test_tuples:
        test_tuples = test_tuples[:max_test_tuples]

    n_test = int(test_tuples.shape[0])
    if n_test == 0:
        return {
            "n_test": 0,
            "n_candidates": n_eval_classes,
            "mr": 0.0,
            "mrr": 0.0,
            "hits@1": 0.0,
            "hits@3": 0.0,
            "hits@10": 0.0,
            "hits@50": 0.0,
            "hits@100": 0.0,
            "auc": 0.0,
            "f_mr": 0.0,
            "f_mrr": 0.0,
            "f_hits@1": 0.0,
            "f_hits@3": 0.0,
            "f_hits@10": 0.0,
            "f_hits@50": 0.0,
            "f_hits@100": 0.0,
            "f_auc": 0.0,
        }

    # Build filter sets per head (train + validation + self) from pre-extracted tensors
    filter_tails_by_head: Dict[int, set[int]] = defaultdict(set)
    test_heads_set = set(int(h) for h in test_tuples[:, 0])
    for h_eval in test_heads_set:
        filter_tails_by_head[h_eval].add(h_eval)

    for ds_dict in (
        getattr(model, "training_datasets", None),
        getattr(model, "validation_datasets", None),
    ):
        if ds_dict and "gci0" in ds_dict and len(ds_dict["gci0"]) > 0:
            ont_pairs = _pairs_from_gci0_tensor(ds_dict["gci0"])
            for h_eval, t_eval in ont_pairs:
                if int(h_eval) in test_heads_set:
                    filter_tails_by_head[int(h_eval)].add(int(t_eval))

    # Group test queries by unique head class A so each unique p_{x_A} is scored once
    queries_by_head: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for q_idx, (h_eval, t_eval) in enumerate(test_tuples):
        queries_by_head[int(h_eval)].append((q_idx, int(t_eval)))

    unique_heads = sorted(queries_by_head.keys())
    raw_ranks = np.empty(n_test, dtype=np.float64)
    filtered_ranks = np.empty(n_test, dtype=np.float64)

    with th.no_grad():
        pos_all = mod.positions()  # (nb_elements, d)
        cand_regs = mod.class_regions(eval_to_model_idx)  # (N_eval, K, d)
        tau_t = mod.tau.to(dtype=pos_all.dtype)
        is_top_cand = (
            (eval_to_model_idx == mod.top_id) if mod.top_id is not None else None
        )
        is_bot_cand = (
            (eval_to_model_idx == mod.bot_id) if mod.bot_id is not None else None
        )
        cand_chunk_size = 4096

        for h_start in range(0, len(unique_heads), eval_batch_size):
            batch_heads = unique_heads[h_start : h_start + eval_batch_size]
            b_sz = len(batch_heads)
            head_model_ids = eval_to_model_idx[
                th.tensor(batch_heads, dtype=th.long, device=device)
            ]
            pts_a = pos_all[head_model_ids]  # (B, d)
            pts_exp = pts_a.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d)

            # Score m(p_{x_A}, B) and depth(p_{x_A}, B) across all candidate classes B
            m_mat = th.empty((b_sz, n_eval_classes), dtype=th.float32, device=device)
            dep_mat = th.empty((b_sz, n_eval_classes), dtype=th.float32, device=device)

            for c_start in range(0, n_eval_classes, cand_chunk_size):
                c_end = min(c_start + cand_chunk_size, n_eval_classes)
                c_sz = c_end - c_start
                cen_c = cand_regs.cen[c_start:c_end].unsqueeze(0)  # (1, C, K, d)
                w_c = cand_regs.w[c_start:c_end].unsqueeze(0)  # (1, C, K, d)
                act_c = cand_regs.active[c_start:c_end]  # (C, K)

                coord_depth = w_c - th.abs(pts_exp - cen_c)  # (B, C, K, d)
                slot_depth = th.amin(coord_depth, dim=-1)  # (B, C, K)
                if mod.agg == "max":
                    has_act_c = act_c.any(dim=-1, keepdim=True).unsqueeze(0)  # (1, C, 1)
                    fill_c = th.where(
                        has_act_c,
                        th.full_like(slot_depth, float("-inf")),
                        th.full_like(slot_depth, -1e6),
                    )
                    dep_chunk = th.amax(
                        th.where(act_c.unsqueeze(0), slot_depth, fill_c), dim=-1
                    )
                else:
                    dep_chunk = mod._aggregate_slot_depth(
                        slot_depth.reshape(b_sz * c_sz, mod.n_components),
                        act_c.unsqueeze(0)
                        .expand(b_sz, c_sz, mod.n_components)
                        .reshape(b_sz * c_sz, mod.n_components),
                        crisp=False,
                    ).reshape(b_sz, c_sz)

                if is_top_cand is not None:
                    top_mask = is_top_cand[c_start:c_end].unsqueeze(0)
                    dep_chunk = th.where(
                        top_mask, th.full_like(dep_chunk, 1e6), dep_chunk
                    )
                if is_bot_cand is not None:
                    bot_mask = is_bot_cand[c_start:c_end].unsqueeze(0)
                    dep_chunk = th.where(
                        bot_mask, th.full_like(dep_chunk, -1e6), dep_chunk
                    )

                m_chunk = th.sigmoid(dep_chunk / tau_t)
                if is_top_cand is not None:
                    m_chunk = th.where(
                        is_top_cand[c_start:c_end].unsqueeze(0),
                        th.ones_like(m_chunk),
                        m_chunk,
                    )
                if is_bot_cand is not None:
                    m_chunk = th.where(
                        is_bot_cand[c_start:c_end].unsqueeze(0),
                        th.zeros_like(m_chunk),
                        m_chunk,
                    )

                m_mat[:, c_start:c_end] = m_chunk
                dep_mat[:, c_start:c_end] = dep_chunk

            for row_i, h_eval in enumerate(batch_heads):
                m_row = m_mat[row_i]  # (N_eval,)
                dep_row = dep_mat[row_i]  # (N_eval,)
                filt_list = sorted(filter_tails_by_head.get(h_eval, ()))
                filt_idx = (
                    th.tensor(filt_list, dtype=th.long, device=device)
                    if filt_list
                    else None
                )

                for q_idx, t_eval in queries_by_head[h_eval]:
                    m_target = m_row[t_eval]
                    dep_target = dep_row[t_eval]
                    strictly_better = (m_row > m_target) | (
                        (m_row == m_target) & (dep_row > dep_target)
                    )
                    raw_rank = 1 + int(strictly_better.sum().item())
                    raw_ranks[q_idx] = float(raw_rank)

                    if filt_idx is not None and filt_idx.numel() > 0:
                        better_filt = strictly_better.clone()
                        better_filt[filt_idx] = False
                        better_filt[t_eval] = False
                        f_rank = 1 + int(better_filt.sum().item())
                    else:
                        f_rank = raw_rank
                    filtered_ranks[q_idx] = float(f_rank)

    ranks_dict: Dict[int, int] = {}
    f_ranks_dict: Dict[int, int] = {}
    for r in raw_ranks.astype(np.int64):
        r_i = int(r)
        ranks_dict[r_i] = ranks_dict.get(r_i, 0) + 1
    for r in filtered_ranks.astype(np.int64):
        r_i = int(r)
        f_ranks_dict[r_i] = f_ranks_dict.get(r_i, 0) + 1

    auc = float(compute_rank_roc(ranks_dict, n_eval_classes))
    f_auc = float(compute_rank_roc(f_ranks_dict, n_eval_classes))

    return {
        "n_test": n_test,
        "n_candidates": n_eval_classes,
        "mr": float(np.mean(raw_ranks)),
        "mrr": float(np.mean(1.0 / raw_ranks)),
        "hits@1": float(np.mean(raw_ranks <= 1.0)),
        "hits@3": float(np.mean(raw_ranks <= 3.0)),
        "hits@10": float(np.mean(raw_ranks <= 10.0)),
        "hits@50": float(np.mean(raw_ranks <= 50.0)),
        "hits@100": float(np.mean(raw_ranks <= 100.0)),
        "auc": auc,
        "f_mr": float(np.mean(filtered_ranks)),
        "f_mrr": float(np.mean(1.0 / filtered_ranks)),
        "f_hits@1": float(np.mean(filtered_ranks <= 1.0)),
        "f_hits@3": float(np.mean(filtered_ranks <= 3.0)),
        "f_hits@10": float(np.mean(filtered_ranks <= 10.0)),
        "f_hits@50": float(np.mean(filtered_ranks <= 50.0)),
        "f_hits@100": float(np.mean(filtered_ranks <= 100.0)),
        "f_auc": f_auc,
    }


def run_go_cell(
    K: int = 2,
    Kr: Optional[int] = None,
    d: int = 50,
    eps_min: float = 0.01,
    epochs: int = 300,
    batch_size: int = 4096,
    lr: float = 0.01,
    seed: int = 0,
    device: str = "cuda",
    data_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    n_eval: int = 10000,
    eval_batch_size: int = 64,
    max_test_tuples: Optional[int] = None,
    reseed_every: int = 0,
    n_filler_samples: int = 64,
    jvm_memory: str = "4g",
) -> Dict[str, Any]:
    """Train :class:`~mowl.models.GroundedEL` on GO and compute §9 E2 metrics.

    Guarantees
    ----------
    - Loads GO via :func:`experiments.datasets.load_dataset` for ``"go"``.
    - Computes ``crisp_report()``, ``evaluate_model(use_reasoner=False)``, and
      test-split subsumption ranking with score :math:`m(p_{x_A}, B)`.
    - Writes the JSON report atomically if ``out_dir`` is provided.

    Non-guarantees
    --------------
    - Does not run an OWL reasoner (``use_reasoner=False``).
    """
    os.environ.setdefault(
        "_JAVA_OPTIONS",
        "-Xms64m -XX:CompressedClassSpaceSize=128m -XX:ReservedCodeCacheSize=64m",
    )
    import mowl

    mowl.init_jvm(os.environ.get("MOWL_JVM_MEMORY", jvm_memory))

    from experiments.datasets import load_dataset
    from mowl.models import GroundedEL

    loaded = (
        load_dataset("go", data_root=data_root)
        if data_root is not None
        else load_dataset("go")
    )
    dataset = loaded.dataset

    eff_kr = K if Kr is None else Kr
    eff_device = device
    if eff_device == "cuda" and not th.cuda.is_available():
        eff_device = "cpu"

    model = GroundedEL(
        dataset,
        embed_dim=d,
        n_components=K,
        n_role_components=eff_kr,
        eps_min=eps_min,
        eps_role=eps_min,
        eps_dead=eps_min,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        reseed_every=reseed_every,
        n_filler_samples=n_filler_samples,
        device=eff_device,
        seed=seed,
    )

    t0 = time.perf_counter()
    history = model.train(epochs=epochs)
    train_wall_sec = float(time.perf_counter() - t0)

    loss_curve: List[Dict[str, Any]] = []
    for idx, rec in enumerate(history):
        ep = idx + 1
        if ep == 1 or ep % 100 == 0 or ep == len(history):
            entry: Dict[str, Any] = {"epoch": ep}
            entry.update({k: float(v) for k, v in rec.items()})
            loss_curve.append(entry)

    final_losses: Dict[str, float] = (
        {k: float(v) for k, v in history[-1].items()} if history else {}
    )

    t_eval0 = time.perf_counter()
    crisp = model.crisp_report(n_eval=n_eval)
    eval_metrics = model.evaluate_model(use_reasoner=False, n_eval=n_eval)
    if isinstance(eval_metrics.get("faithful_named"), tuple):
        eval_metrics = dict(eval_metrics)
        eval_metrics["faithful_named"] = list(eval_metrics["faithful_named"])
    ranking = evaluate_subsumption_ranking(
        model,
        dataset,
        eval_batch_size=eval_batch_size,
        max_test_tuples=max_test_tuples,
    )
    eval_wall_sec = float(time.perf_counter() - t_eval0)

    payload: Dict[str, Any] = {
        "config": {
            "dataset": "go",
            "K": K,
            "Kr": eff_kr,
            "dim": d,
            "eps_min": eps_min,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "reseed_every": reseed_every,
            "n_filler_samples": n_filler_samples,
            "seed": seed,
            "device": eff_device,
            "nb_classes": int(model.nb_classes),
            "nb_elements": int(model.nb_elements),
            "nb_rels": int(model.nb_rels),
            "n_eval": n_eval,
            "axiom_counts": {
                k: int(len(ds_t)) for k, ds_t in model.training_datasets.items()
            },
        },
        "crisp_report": crisp,
        "evaluate_model": eval_metrics,
        "test_ranking": ranking,
        "final_losses": final_losses,
        "loss_curve": loss_curve,
        "wall_time_sec": train_wall_sec,
        "eval_time_sec": eval_wall_sec,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        eps_tag = f"{eps_min:g}"
        out_path = out_dir / f"go__K{K}__d{d}__eps{eps_tag}__s{seed}.json"
        tmp_path = out_path.with_suffix(f".json.tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp_path.replace(out_path)

    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``run_go.py``.

    Guarantees
    ----------
    - Exposes ``--K``, ``--Kr``, ``--d`` / ``--dim``, ``--eps-min`` / ``--eps``,
      ``--epochs``, ``--batch-size``, ``--lr``, ``--reseed-every``,
      ``--n-filler-samples``, ``--seed``, ``--device``, ``--data-root``, and ``--out``.

    Non-guarantees
    --------------
    - Does not validate values until :func:`_validate_args` is called.
    """
    parser = argparse.ArgumentParser(
        description="Train GroundedEL on GO subsumption (design §9 E2)."
    )
    parser.add_argument("--K", type=int, default=2, help="Number of class boxes K.")
    parser.add_argument(
        "--Kr",
        type=int,
        default=None,
        help="Number of role boxes Kr (defaults to K).",
    )
    parser.add_argument(
        "--d",
        "--dim",
        dest="d",
        type=int,
        default=50,
        help="Embedding dimension d (default: 50).",
    )
    parser.add_argument(
        "--eps-min",
        "--eps",
        dest="eps_min",
        type=float,
        default=0.01,
        help="Minimum box width / margin eps_min (default: 0.01).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs (default: 300).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Mini-batch size per axiom type (default: 4096).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--reseed-every",
        type=int,
        default=0,
        help="Reseed interval in epochs (0 disables reseeding; default: 0).",
    )
    parser.add_argument(
        "--n-filler-samples",
        type=int,
        default=64,
        help="Maximum candidate fillers sampled per gci3/gci3_bot batch (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda; falls back to cpu if unavailable).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory for cached mOWL datasets (default: MOWL_DATA_ROOT).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/data/u/hohndor/gfe/runs/grounded-e2"),
        help="Output directory for JSON results.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=10000,
        help="Sample budget for crisp region_viol evaluation (default: 10000).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Number of unique test head classes scored per batch (default: 64).",
    )
    parser.add_argument(
        "--max-test-tuples",
        type=int,
        default=None,
        help="Optional cap on test SubClassOf pairs evaluated (default: full test split).",
    )
    parser.add_argument(
        "--jvm-memory",
        type=str,
        default="4g",
        help="JVM max heap size passed to mowl.init_jvm (default: 4g).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``experiments.grounded.run_go``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    res = run_go_cell(
        K=args.K,
        Kr=args.Kr,
        d=args.d,
        eps_min=args.eps_min,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out,
        n_eval=args.n_eval,
        eval_batch_size=args.eval_batch_size,
        max_test_tuples=args.max_test_tuples,
        reseed_every=args.reseed_every,
        n_filler_samples=args.n_filler_samples,
        jvm_memory=args.jvm_memory,
    )
    cr = res["crisp_report"]
    ev = res["evaluate_model"]
    rk = res["test_ranking"]
    print(
        f"[done] dataset=go K={res['config']['K']} d={res['config']['dim']} "
        f"eps_min={res['config']['eps_min']} epochs={res['config']['epochs']} "
        f"elem_viol={cr['elem_viol']:.4f} region_viol={cr['region_viol']:.4f} "
        f"exist={ev['exist']:.4f} mrr={rk['mrr']:.4f} f_mrr={rk['f_mrr']:.4f} "
        f"hits@1={rk['hits@1']:.4f} hits@10={rk['hits@10']:.4f} "
        f"hits@100={rk['hits@100']:.4f} auc={rk['auc']:.4f} "
        f"train_wall={res['wall_time_sec']:.2f}s eval_wall={res['eval_time_sec']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
