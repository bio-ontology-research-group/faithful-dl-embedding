"""Evaluation CLI and metrics for DeepGO-SE GroundedEL and baselines (E3, design §3).

For a list of model IDs, selects the top-``--n-models`` checkpoints by validation
BCE, combines their raw test score matrices via ``--combine avg|min|max``, and
computes:

1. **DeepGO-SE metrics** (via ``deepgo.metrics.compute_metrics`` from the
   reference ``deepgo2`` checkout) on both **raw** scores and **max-propagated**
   (``deepgo.utils.propagate_annots``) scores: ``fmax``, ``smin``, ``aupr``,
   ``avg_auc``, and threshold ``tmax``.
2. **Hierarchy consistency of raw scores** (design §3):
   - ``hier_viol_rate``: fraction of ``(protein, G ⊑ G')`` pairs among predicted
     terms with ``s(p, G) > s(p, G') + 1e-6``.
   - ``ancestor_closed_rate``: fraction of test proteins whose thresholded
     prediction set at ``t_max`` (:math:`\\{G : s(p, G) \\ge t_{\\max}\\}`) is
     ancestor-closed with respect to :math:`G \\sqsubseteq G'`.
3. **GroundedEL TBox faithfulness** of the GO boxes (``use_reasoner=False``):
   ``elem_viol``, ``region_viol``, ``region_viol_tol1e-4``, ``n_false_dead``,
   ``n_empty_classes``.

Writes ``{out}/{ont}/eval_{model_name}_{combine}_{n}.json`` (or ``..._{n}_ids{i-j-k}.json``
when ``--model-ids`` is given) and prints a summary
table.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch as th

# NumPy 2.x compatibility for read-only deepgo/metrics.py (which calls np.trapz)
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

DEEPGO2_DEFAULT_DIR = Path("/home/leechuck/Public/software/deepgo2")


def _ensure_deepgo_on_path(deepgo_dir: Optional[Union[str, Path]] = None) -> None:
    """Add the reference ``deepgo2`` repository to ``sys.path`` if not already importable."""
    target = Path(
        deepgo_dir
        or os.environ.get("DEEPGO2_DIR", str(DEEPGO2_DEFAULT_DIR))
    )
    target_str = str(target.resolve())
    if target.is_dir() and target_str not in sys.path:
        sys.path.insert(0, target_str)


def compute_hierarchy_consistency(
    raw_scores: np.ndarray,
    subclass_pairs: Union[Sequence[Tuple[int, int]], np.ndarray, th.Tensor],
    t_max: float = 0.5,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Compute design §3 hierarchy-consistency metrics on raw score matrix ``(N_proteins, n_terms)``.

    Parameters
    ----------
    raw_scores : np.ndarray
        2-D array of shape ``(N_proteins, n_terms)`` containing raw (unpropagated)
        prediction scores :math:`s(p, G) \\in [0, 1]`.
    subclass_pairs : Union[Sequence[Tuple[int, int]], np.ndarray, torch.Tensor]
        Collection of ``(sub_idx, sup_idx)`` term index pairs representing
        :math:`G_{\\text{sub}} \\sqsubseteq G_{\\text{sup}}` among the ``n_terms``
        predicted columns.
    t_max : float, default 0.5
        Threshold :math:`t_{\\max} \\in [0, 1]` defining each protein's binary
        prediction set :math:`\\{G : s(p, G) \\ge t_{\\max}\\}`.
    tol : float, default 1e-6
        Numerical tolerance for score order violations: a pair ``(p, G ⊑ G')`` is
        violated iff ``s(p, G) > s(p, G') + tol``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - ``hier_viol_rate``: fraction of ``(protein, G ⊑ G')`` pairs with
          ``s(p, G) > s(p, G') + tol``.
        - ``n_violated_pairs``: integer count of violated ``(protein, G ⊑ G')`` pairs.
        - ``n_total_pairs``: integer ``N_proteins * len(subclass_pairs)``.
        - ``ancestor_closed_rate``: fraction of proteins whose threshold-``t_max``
          prediction set is closed under all ancestors reachable via ``subclass_pairs``.
        - ``n_ancestor_closed_proteins``: integer count of ancestor-closed proteins.
        - ``n_proteins``: integer ``N_proteins``.
        - ``n_subclass_pairs``: integer number of evaluated ``(G, G')`` pairs.
        - ``t_max``: float threshold used.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"raw_scores must be 2-D (N, n_terms), got shape {scores.shape}")
    n_proteins, n_terms = scores.shape

    if isinstance(subclass_pairs, th.Tensor):
        pairs_arr = subclass_pairs.detach().cpu().numpy().astype(np.int64)
    else:
        pairs_arr = np.asarray(subclass_pairs, dtype=np.int64)

    if pairs_arr.size == 0:
        pairs_arr = np.zeros((0, 2), dtype=np.int64)
    elif pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
        raise ValueError(
            f"subclass_pairs must have shape (M, 2), got {pairs_arr.shape}"
        )

    # Filter self-loops if any
    if pairs_arr.shape[0] > 0:
        mask = pairs_arr[:, 0] != pairs_arr[:, 1]
        pairs_arr = pairs_arr[mask]

    n_pairs = int(pairs_arr.shape[0])
    if n_proteins == 0 or n_pairs == 0:
        return {
            "hier_viol_rate": 0.0,
            "n_violated_pairs": 0,
            "n_total_pairs": 0,
            "ancestor_closed_rate": 1.0,
            "n_ancestor_closed_proteins": int(n_proteins),
            "n_proteins": int(n_proteins),
            "n_subclass_pairs": n_pairs,
            "t_max": float(t_max),
        }

    sub_idx = pairs_arr[:, 0]
    sup_idx = pairs_arr[:, 1]

    # 1. Pairwise score violation: s(p, G) > s(p, G') + tol
    # Chunk over pairs to keep memory small when n_pairs is large
    viol_count = 0
    pair_chunk = 4096
    for s_i in range(0, n_pairs, pair_chunk):
        e_i = min(s_i + pair_chunk, n_pairs)
        sub_c = sub_idx[s_i:e_i]
        sup_c = sup_idx[s_i:e_i]
        viol_mat = scores[:, sub_c] > (scores[:, sup_c] + float(tol))
        viol_count += int(np.sum(viol_mat))

    total_pairs = int(n_proteins * n_pairs)
    hier_viol_rate = float(viol_count / total_pairs) if total_pairs > 0 else 0.0

    # 2. Ancestor closure of thresholded prediction set {G : s(p, G) >= t_max}.
    # A set S is closed under transitive ancestors of `pairs_arr` iff for every
    # edge (u, v) in `pairs_arr`, u in S implies v in S.
    pred_bin = scores >= float(t_max)  # (N_proteins, n_terms)
    protein_has_edge_viol = np.zeros((n_proteins,), dtype=bool)
    for s_i in range(0, n_pairs, pair_chunk):
        e_i = min(s_i + pair_chunk, n_pairs)
        sub_c = sub_idx[s_i:e_i]
        sup_c = sup_idx[s_i:e_i]
        edge_viol = pred_bin[:, sub_c] & (~pred_bin[:, sup_c])
        protein_has_edge_viol |= np.any(edge_viol, axis=1)

    n_closed = int(np.sum(~protein_has_edge_viol))
    closed_rate = float(n_closed / n_proteins)

    return {
        "hier_viol_rate": hier_viol_rate,
        "n_violated_pairs": int(viol_count),
        "n_total_pairs": total_pairs,
        "ancestor_closed_rate": closed_rate,
        "n_ancestor_closed_proteins": n_closed,
        "n_proteins": int(n_proteins),
        "n_subclass_pairs": n_pairs,
        "t_max": float(t_max),
    }


def extract_ontology_term_pairs(
    go: Any,
    terms_dict: Mapping[str, int],
) -> np.ndarray:
    """Extract all entailed ``(sub_col_idx, sup_col_idx)`` pairs among ``terms_dict`` from ``go``."""
    pairs: Set[Tuple[int, int]] = set()
    for term_id, sub_col in terms_dict.items():
        if not go.has_term(term_id):
            continue
        for anc_id in go.get_ancestors(term_id):
            if anc_id != term_id and anc_id in terms_dict:
                pairs.add((int(sub_col), int(terms_dict[anc_id])))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def combine_prediction_matrices(
    matrices: Sequence[np.ndarray],
    combine: str = "avg",
) -> np.ndarray:
    """Combine a sequence of ``(N, n_terms)`` score matrices using ``avg``, ``min``, or ``max``."""
    if not matrices:
        raise ValueError("matrices must be non-empty")
    if len(matrices) == 1:
        return np.asarray(matrices[0], dtype=np.float32).copy()
    stacked = np.stack([np.asarray(m, dtype=np.float32) for m in matrices], axis=0)
    if combine == "avg":
        return np.mean(stacked, axis=0)
    if combine == "min":
        return np.min(stacked, axis=0)
    if combine == "max":
        return np.max(stacked, axis=0)
    raise ValueError(f"combine must be one of {{'avg', 'min', 'max'}}, got {combine!r}")


def _discover_candidate_models(
    out_ont_dir: Path,
    model_name: str,
    model_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float, Path, Optional[Path]]]:
    """Return ``(model_id, val_bce, pred_pkl_path, ckpt_pt_path)`` sorted by ``val_bce`` ascending."""
    if model_ids is not None:
        cand_ids = [int(x) for x in model_ids]
    else:
        cand_ids = list(range(10))

    found: List[Tuple[int, float, Path, Optional[Path]]] = []
    for mi in cand_ids:
        pred_p = out_ont_dir / f"test_predictions_{model_name}_{mi}.pkl"
        if not pred_p.is_file():
            continue

        val_bce = float("inf")
        metrics_p = out_ont_dir / f"metrics_{model_name}_{mi}.json"
        if metrics_p.is_file():
            try:
                with open(metrics_p, "r", encoding="utf-8") as fh:
                    m_data = json.load(fh)
                if "best_val_bce" in m_data:
                    val_bce = float(m_data["best_val_bce"])
            except Exception:
                pass

        ckpt_p = out_ont_dir / f"{model_name}_{mi}.pt"
        if not ckpt_p.is_file():
            ckpt_p_Opt: Optional[Path] = None
        else:
            ckpt_p_Opt = ckpt_p
            if not np.isfinite(val_bce):
                try:
                    ckpt_obj = th.load(ckpt_p, map_location="cpu", weights_only=False)
                    if isinstance(ckpt_obj, dict) and "best_val_bce" in ckpt_obj:
                        val_bce = float(ckpt_obj["best_val_bce"])
                except Exception:
                    pass

        if not np.isfinite(val_bce):
            val_bce = float(mi)
        found.append((mi, val_bce, pred_p, ckpt_p_Opt))

    if not found and 0 in cand_ids:
        fallback_p = out_ont_dir / "test_predictions.pkl"
        if fallback_p.is_file():
            val_bce = 0.0
            fallback_m = out_ont_dir / "metrics.json"
            if fallback_m.is_file():
                try:
                    with open(fallback_m, "r", encoding="utf-8") as fh:
                        m_data = json.load(fh)
                    if "best_val_bce" in m_data:
                        val_bce = float(m_data["best_val_bce"])
                except Exception:
                    pass
            ckpt_0 = out_ont_dir / f"{model_name}_0.pt"
            found.append(
                (0, val_bce, fallback_p, ckpt_0 if ckpt_0.is_file() else None)
            )

    if not found:
        raise FileNotFoundError(
            f"No prediction files found for model_name={model_name!r} under {out_ont_dir}"
        )

    found.sort(key=lambda item: (item[1], item[0]))
    return found


def evaluate_tbox_crisp_report(
    ckpt_path: Path,
    data_root: Path,
    out_ont_dir: Path,
    ont: str,
    n_eval: int = 10000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a GroundedEL checkpoint and compute its ``crisp_report`` / ``evaluate_model(use_reasoner=False)`` metrics."""
    from experiments.deepgose.data import parse_go_norm
    from experiments.deepgose.model import build_deepgose_grounded_model

    ckpt = th.load(ckpt_path, map_location="cpu", weights_only=False)
    terms = list(ckpt.get("terms", []))
    dim = int(ckpt.get("dim", 128))
    K = int(ckpt.get("K", 1))
    lambda_tbox = float(ckpt.get("lambda_tbox", 1.0))
    seed = int(ckpt.get("seed", 0))

    norm_file = data_root / "go.norm"
    if not norm_file.is_file():
        norm_file = data_root / ont / "go.norm"

    tables = parse_go_norm(norm_file, terms=terms, cache_path=None)
    model = build_deepgose_grounded_model(
        tables=tables,
        embed_dim=dim,
        n_components=K,
        n_role_components=K,
        lambda_tbox=lambda_tbox,
        seed=seed,
        device=device,
    )
    model.module.load_state_dict(
        {k: v.to(device) for k, v in ckpt["state_dict"].items()}
    )
    model.module.eval()

    ev = model.grounded.evaluate_model(use_reasoner=False, n_eval=n_eval)
    td = model.grounded.training_datasets
    has_bot_axioms = any(
        k in td and len(td[k]) > 0 for k in ("gci0_bot", "gci1_bot", "gci3_bot")
    )
    # In positive EL ontologies (such as go.norm, which has zero *_bot axioms),
    # every non-bottom class is satisfiable, so any dead non-bottom element is
    # a false dead element without needing an external OWL reasoner.
    n_false_dead = (
        int(ev["n_dead_elements"])
        if (ev["n_false_dead"] is None and not has_bot_axioms)
        else ev["n_false_dead"]
    )
    return {
        "elem_viol": float(ev["elem_viol"]),
        "region_viol": float(ev["region_viol"]),
        "region_viol_tol1e-4": float(ev["region_viol_tol1e-4"]),
        "n_false_dead": n_false_dead,
        "n_empty_classes": int(ev["n_empty_classes"]),
        "n_dead_elements": int(ev["n_dead_elements"]),
        "exist": float(ev["exist"]),
    }


def run_evaluation(
    ont: str = "cc",
    model_name: str = "grounded_esm",
    model_ids: Optional[Sequence[int]] = None,
    n_models: int = 6,
    combine: str = "avg",
    data_root: Union[str, Path] = "/mnt/data1/gfe/data/deepgo2/data",
    out_dir: Union[str, Path] = "/mnt/data/u/hohndor/gfe/runs/deepgose",
    n_eval: int = 10000,
    device: str = "cpu",
    deepgo_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate top-``n_models`` checkpoints for ``model_name`` on ``ont`` and write JSON report."""
    if combine not in ("avg", "min", "max"):
        raise ValueError(f"combine must be one of {{'avg', 'min', 'max'}}, got {combine!r}")
    if n_models < 1:
        raise ValueError(f"n_models must be >= 1, got {n_models}")

    _ensure_deepgo_on_path(deepgo_dir)
    from deepgo.metrics import compute_metrics
    from deepgo.utils import Ontology, propagate_annots

    root = Path(data_root)
    ont_dir = root / ont
    out_ont_dir = Path(out_dir) / ont
    out_ont_dir.mkdir(parents=True, exist_ok=True)

    obo_path = root / "go.obo"
    if not obo_path.is_file():
        obo_path = ont_dir / "go.obo"
    go = Ontology(str(obo_path), with_rels=True)

    candidates = _discover_candidate_models(
        out_ont_dir=out_ont_dir,
        model_name=model_name,
        model_ids=model_ids,
    )
    selected = candidates[: int(n_models)]
    selected_ids = [item[0] for item in selected]
    selected_val_bces = [float(item[1]) for item in selected]

    # Determine the evaluated terms list from checkpoint, metrics JSON, or terms.pkl
    first_ckpt = selected[0][3]
    ckpt_meta: Any = None
    terms: Optional[List[str]] = None
    if first_ckpt is not None and first_ckpt.is_file():
        ckpt_meta = th.load(first_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ckpt_meta, dict) and "terms" in ckpt_meta:
            terms = [str(t) for t in ckpt_meta["terms"]]
    if terms is None:
        first_mi = selected[0][0]
        first_metrics_p = out_ont_dir / f"metrics_{model_name}_{first_mi}.json"
        if not first_metrics_p.is_file() and first_mi == 0:
            fallback_m = out_ont_dir / "metrics.json"
            if fallback_m.is_file():
                first_metrics_p = fallback_m
        if first_metrics_p.is_file():
            try:
                with open(first_metrics_p, "r", encoding="utf-8") as fh:
                    m_info = json.load(fh)
                if isinstance(m_info, dict) and "terms" in m_info and m_info["terms"]:
                    terms = [str(t) for t in m_info["terms"]]
            except Exception:
                pass
    if terms is None:
        terms_df = pd.read_pickle(ont_dir / "terms.pkl")
        terms = [str(t) for t in np.asarray(terms_df["gos"]).flatten()]

    terms_dict = {t: i for i, t in enumerate(terms)}

    raw_matrices: List[np.ndarray] = []
    prop_matrices: List[Optional[np.ndarray]] = []
    base_test_df: Optional[pd.DataFrame] = None
    for _, _, pred_p, _ in selected:
        df_i = pd.read_pickle(pred_p)
        if base_test_df is None:
            base_test_df = df_i
        if "raw_preds" in df_i.columns:
            raw_mat_i = np.stack(
                [np.asarray(r, dtype=np.float32) for r in df_i["raw_preds"].values],
                axis=0,
            )
            prop_mat_i: Optional[np.ndarray] = np.stack(
                [np.asarray(r, dtype=np.float32) for r in df_i["preds"].values],
                axis=0,
            )
        else:
            raw_mat_i = np.stack(
                [np.asarray(r, dtype=np.float32) for r in df_i["preds"].values],
                axis=0,
            )
            prop_mat_i = None

        if raw_mat_i.shape[1] != len(terms):
            raise ValueError(
                f"Prediction matrix in {pred_p} has {raw_mat_i.shape[1]} columns, expected {len(terms)}"
            )
        raw_matrices.append(raw_mat_i)
        prop_matrices.append(prop_mat_i)

    assert base_test_df is not None

    # Populate Information Content (IC) on `go` for Smin / wfmax in compute_metrics
    ic_cache_path = out_ont_dir / f"ic_cache_{ont}.pt"
    if ic_cache_path.is_file():
        ic_cached = th.load(ic_cache_path, map_location="cpu", weights_only=False)
        go.ic = dict(ic_cached["ic"])
        go.ic_norm = float(ic_cached["ic_norm"])
    else:
        train_pkl = ont_dir / "train_data.pkl"
        valid_pkl = ont_dir / "valid_data.pkl"
        annots_all: List[Set[str]] = []
        if train_pkl.is_file() and valid_pkl.is_file():
            tr_df = pd.read_pickle(train_pkl)
            va_df = pd.read_pickle(valid_pkl)
            for a_seq in tr_df["prop_annotations"].values:
                annots_all.append(set(a_seq))
            for a_seq in va_df["prop_annotations"].values:
                annots_all.append(set(a_seq))
        for a_seq in base_test_df["prop_annotations"].values:
            annots_all.append(set(a_seq))
        go.calculate_ic(annots_all)
        tmp_ic = ic_cache_path.with_suffix(f".pt.tmp.{os.getpid()}")
        th.save({"ic": dict(go.ic), "ic_norm": float(go.ic_norm)}, tmp_ic)
        tmp_ic.replace(ic_cache_path)

    raw_combined = combine_prediction_matrices(raw_matrices, combine=combine)
    raw_preds_arr = raw_combined.copy()
    if all(m is not None for m in prop_matrices):
        prop_preds_arr = combine_prediction_matrices(
            [m for m in prop_matrices if m is not None], combine=combine
        )
    else:
        prop_preds_arr = np.stack(
            [propagate_annots(row.copy(), go, terms_dict) for row in raw_combined],
            axis=0,
        )

    # 1. DeepGO-SE compute_metrics on raw and propagated scores
    raw_fmax, raw_smin, raw_tmax, raw_wfmax, raw_wtmax, raw_avg_auc, raw_aupr, raw_avg_ic, _ = (
        compute_metrics(base_test_df, go, terms_dict, terms, ont, raw_preds_arr)
    )
    prop_fmax, prop_smin, prop_tmax, prop_wfmax, prop_wtmax, prop_avg_auc, prop_aupr, prop_avg_ic, _ = (
        compute_metrics(base_test_df, go, terms_dict, terms, ont, prop_preds_arr)
    )

    # 2. Hierarchy consistency of raw scores (§3)
    term_pairs = extract_ontology_term_pairs(go, terms_dict)
    hier_metrics = compute_hierarchy_consistency(
        raw_scores=raw_combined,
        subclass_pairs=term_pairs,
        t_max=float(raw_tmax),
        tol=1e-6,
    )

    # 3. GroundedEL crisp_report of the GO boxes (only for GroundedEL checkpoints)
    eff_device = device
    if eff_device.startswith("cuda") and not th.cuda.is_available():
        eff_device = "cpu"

    is_grounded_ckpt = (
        first_ckpt is not None
        and first_ckpt.is_file()
        and isinstance(ckpt_meta, dict)
        and "state_dict" in ckpt_meta
        and isinstance(ckpt_meta["state_dict"], Mapping)
        and {"class_centers", "class_raw_widths"} <= {str(k) for k in ckpt_meta["state_dict"].keys()}
    )
    if is_grounded_ckpt and first_ckpt is not None:
        tbox_report = evaluate_tbox_crisp_report(
            ckpt_path=first_ckpt,
            data_root=root,
            out_ont_dir=out_ont_dir,
            ont=ont,
            n_eval=n_eval,
            device=eff_device,
        )
    else:
        tbox_report = {
            "elem_viol": None,
            "region_viol": None,
            "region_viol_tol1e-4": None,
            "n_false_dead": None,
            "n_empty_classes": None,
            "n_dead_elements": None,
            "exist": None,
        }

    n_used = len(selected_ids)
    payload: Dict[str, Any] = {
        "ont": ont,
        "model_name": model_name,
        "combine": combine,
        "n_models_requested": int(n_models),
        "n_models_used": int(n_used),
        "selected_model_ids": selected_ids,
        "selected_val_bces": selected_val_bces,
        "n_test_proteins": int(raw_combined.shape[0]),
        "n_terms": len(terms),
        "raw": {
            "fmax": float(raw_fmax),
            "smin": float(raw_smin),
            "tmax": float(raw_tmax),
            "wfmax": float(raw_wfmax),
            "wtmax": float(raw_wtmax),
            "aupr": float(raw_aupr),
            "avg_auc": float(raw_avg_auc),
            "avg_ic": float(raw_avg_ic),
        },
        "propagated": {
            "fmax": float(prop_fmax),
            "smin": float(prop_smin),
            "tmax": float(prop_tmax),
            "wfmax": float(prop_wfmax),
            "wtmax": float(prop_wtmax),
            "aupr": float(prop_aupr),
            "avg_auc": float(prop_avg_auc),
            "avg_ic": float(prop_avg_ic),
        },
        "hierarchy_consistency": hier_metrics,
        "crisp_report": tbox_report,
    }

    # Explicit model ids: name the file by the ids so per-seed evaluations never
    # overwrite each other. Otherwise (top-n selection) write both the requested
    # n_models filename and the actual n_used filename if they differ.
    if model_ids is not None:
        id_tag = "-".join(str(i) for i in sorted(selected_ids))
        out_paths = {out_ont_dir / f"eval_{model_name}_{combine}_{n_used}_ids{id_tag}.json"}
    else:
        out_paths = {
            out_ont_dir / f"eval_{model_name}_{combine}_{n_models}.json",
            out_ont_dir / f"eval_{model_name}_{combine}_{n_used}.json",
        }
    for out_p in out_paths:
        tmp_p = out_p.with_suffix(f".json.tmp.{os.getpid()}")
        with open(tmp_p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp_p.replace(out_p)

    _print_summary_table(payload)
    return payload


def _print_summary_table(payload: Mapping[str, Any]) -> None:
    """Print a formatted summary table of raw, propagated, hierarchy, and TBox metrics."""
    raw = payload["raw"]
    prop = payload["propagated"]
    hc = payload["hierarchy_consistency"]
    cr = payload["crisp_report"]

    header = (
        f"=== DeepGO-SE Evaluation: ont={payload['ont']} "
        f"model={payload['model_name']} combine={payload['combine']} "
        f"S={payload['n_models_used']} (ids={payload['selected_model_ids']}) ==="
    )
    print(header)
    print(
        f"{'Mode':<12} {'Fmax':>8} {'Smin':>10} {'AUPR':>8} {'AvgAUC':>8} {'t_max':>7}"
    )
    print("-" * 58)
    print(
        f"{'raw':<12} {raw['fmax']:>8.4f} {raw['smin']:>10.4f} "
        f"{raw['aupr']:>8.4f} {raw['avg_auc']:>8.4f} {raw['tmax']:>7.2f}"
    )
    print(
        f"{'propagated':<12} {prop['fmax']:>8.4f} {prop['smin']:>10.4f} "
        f"{prop['aupr']:>8.4f} {prop['avg_auc']:>8.4f} {prop['tmax']:>7.2f}"
    )
    print("-" * 58)
    print(
        f"Hierarchy (raw): hier_viol_rate={hc['hier_viol_rate']:.6f} "
        f"({hc['n_violated_pairs']}/{hc['n_total_pairs']}), "
        f"ancestor_closed_rate={hc['ancestor_closed_rate']:.4f} "
        f"({hc['n_ancestor_closed_proteins']}/{hc['n_proteins']})"
    )
    if cr.get("elem_viol") is not None:
        print(
            f"TBox crisp_report: elem_viol={cr['elem_viol']:.4f} "
            f"region_viol={cr['region_viol']:.4f} "
            f"region_viol_tol1e-4={cr['region_viol_tol1e-4']:.4f} "
            f"n_false_dead={cr['n_false_dead']} "
            f"n_empty_classes={cr['n_empty_classes']}"
        )
    print("=" * len(header), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``experiments.deepgose.evaluate``."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeepGO-SE GroundedEL or baseline predictions (E3, design §3)."
    )
    parser.add_argument(
        "--ont",
        "-ont",
        choices=["mf", "bp", "cc"],
        default="cc",
        help="GO sub-ontology (mf, bp, cc).",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="grounded_esm",
        help="Model name prefix (default: grounded_esm).",
    )
    parser.add_argument(
        "--model-ids",
        type=int,
        nargs="+",
        default=None,
        help="Candidate model IDs to consider (default: discover available 0..9).",
    )
    parser.add_argument(
        "--n-models",
        "-n",
        type=int,
        default=1,
        help="Top-N models by validation BCE to combine (default: 1).",
    )
    parser.add_argument(
        "--combine",
        choices=["avg", "min", "max"],
        default="avg",
        help="Combination rule across selected models (default: avg).",
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
        help="Output directory containing model checkpoints and predictions.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=10000,
        help="Sample budget for GroundedEL crisp_report (default: 10000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for GroundedEL crisp_report (default: cpu).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``experiments.deepgose.evaluate``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluation(
        ont=args.ont,
        model_name=args.model_name,
        model_ids=args.model_ids,
        n_models=args.n_models,
        combine=args.combine,
        data_root=args.data_root,
        out_dir=args.out,
        n_eval=args.n_eval,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
