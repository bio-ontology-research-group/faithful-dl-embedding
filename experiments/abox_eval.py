"""ABox evaluation primitives for B1 (``yeast-function``) and B3 (``helis``) (Task 3).

Guarantees
----------
- ``score_candidates(model, individuals, candidate_classes)`` returns an
  ``(n_ind, n_cand)`` ``np.ndarray`` of plausibility scores (lower = more plausible)
  computed in chunks of ``<= 2**16`` rows on the model's device with automatic CPU
  fallback on CUDA OOM/runtime error, and uses ``module.rank_classes(ind_idx)``
  when available (``CanonicalMixture``). Missing individuals or candidate classes
  in the model's index dictionaries receive ``+inf``.
- ``ranking(scores, positives, skip_mask=None)`` implements filtered per-positive
  ranking after masking (i) KB-entailed candidates, (ii) the individual's other
  held-out positives in the same split, and (iii) the benchmark skip list, while
  **never masking the target itself**. Ties are broken **pessimistically** (every
  unmasked candidate with score ``<= target_score`` counts as ranked at or above
  the target). Reports both ``rankable_only`` and ``all_positives`` metrics (where
  unrankable positives outside ``candidates`` receive ``rank = n_candidates + 1``)
  along with coverage counts.
- ``fmax(scores, true_sets, hierarchy, ..., entailed_mask=None, labels=None)``
  computes the CAFA-style protein-centric :math:`F_{\\max}` over ancestor-closed
  held-out annotations using 100 thresholds over the score quantiles, excluding
  KB-entailed cells (``label == entailed``) from both the prediction set and the
  truth set so a model that scores KB-entailed candidates as plausible is not
  penalised.
- ``hierarchy_violation_rate(scores, entailed_pairs)`` samples up to 10,000
  ``(a, G ⊑ G')`` instances with ``G != G'`` and the subsumption entailed,
  counting a violation iff ``score(a, G) < score(a, G') - 1e-6``, and returning
  ``None`` when fewer than 50 instances exist.
- ``three_way_metrics(scores, labels)`` computes ``auc_pos_neg`` (``AUC(+/-)``),
  ``auc_pos_und`` (``AUC(+/?)``, pooled across individuals), ``pair_fvr``, and
  ``fvr`` following the sampling and ``< 50`` null rules of
  ``experiments.metrics.faithfulness_metrics``.
- ``abox_modelness(model)`` computes the fraction of training class assertions
  (and object-property assertions when supported by the module) whose strict loss
  is ``<= 1e-6`` with ``margin = 0`` (restoring ``margin``/``gamma`` afterwards),
  returning ``None`` when no training assertion was scored.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.stats import rankdata
import torch

logger = logging.getLogger(__name__)

MAX_CHUNK_ROWS: int = 65536  # 2**16


def _resolve_model_device(model: Any) -> torch.device:
    """Resolve the active torch device of ``model.module`` (or ``model.device``)."""
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "parameters"):
        try:
            first_param = next(module.parameters(), None)
            if first_param is not None:
                return first_param.device
        except TypeError:
            pass
        if hasattr(module, "buffers"):
            try:
                first_buf = next(module.buffers(), None)
                if first_buf is not None:
                    return first_buf.device
            except TypeError:
                pass
    dev = getattr(model, "device", "cpu")
    return torch.device(dev)


def _lookup_index(mapping: Mapping[Any, int], key: Any) -> int:
    """Look up ``key`` in ``mapping``, returning ``-1`` if absent."""
    if key in mapping:
        return int(mapping[key])
    s_key = str(key)
    if s_key in mapping:
        return int(mapping[s_key])
    return -1


def count_missing_entities(
    model: Any,
    individuals: Sequence[Any],
    candidate_classes: Sequence[Any],
) -> Dict[str, int]:
    """Count individuals and candidate classes missing from ``model``'s index dicts."""
    ind_dict: Mapping[Any, int] = getattr(model, "individual_index_dict", {}) or {}
    cls_dict: Mapping[Any, int] = getattr(model, "class_index_dict", {}) or {}
    n_missing_ind = sum(1 for ind in individuals if _lookup_index(ind_dict, ind) < 0)
    n_missing_cand = sum(
        1 for cls in candidate_classes if _lookup_index(cls_dict, cls) < 0
    )
    return {
        "n_missing_individuals": int(n_missing_ind),
        "n_missing_candidates": int(n_missing_cand),
    }


def score_candidates(
    model: Any,
    individuals: Sequence[Any],
    candidate_classes: Sequence[Any],
    *,
    batch_size: int = MAX_CHUNK_ROWS,
) -> np.ndarray:
    """Score all ``(individual, candidate_class)`` pairs for ``model``.

    Parameters
    ----------
    model : EmbeddingELModel
        Trained mOWL EL model exposing ``individual_index_dict``,
        ``class_index_dict``, and ``module``.
    individuals : Sequence[Any]
        Ordered sequence of evaluation individual IRIs (or keys in
        ``model.individual_index_dict``).
    candidate_classes : Sequence[Any]
        Ordered sequence of candidate class IRIs (or keys in
        ``model.class_index_dict``).
    batch_size : int, optional
        Maximum number of ``(individual, class)`` rows per forward batch
        (clamped to ``<= 65536`` = ``2**16``).

    Returns
    -------
    np.ndarray
        Float64 matrix of shape ``(n_ind, n_cand)`` where lower values indicate
        higher plausibility. Any individual or class missing from the model's
        index dictionaries receives ``+inf``.
    """
    n_ind = len(individuals)
    n_cand = len(candidate_classes)
    scores = np.full((n_ind, n_cand), np.inf, dtype=np.float64)
    if n_ind == 0 or n_cand == 0:
        return scores

    ind_dict: Mapping[Any, int] = getattr(model, "individual_index_dict", {}) or {}
    cls_dict: Mapping[Any, int] = getattr(model, "class_index_dict", {}) or {}

    ind_indices = np.array(
        [_lookup_index(ind_dict, ind) for ind in individuals], dtype=np.int64
    )
    cand_indices = np.array(
        [_lookup_index(cls_dict, cls) for cls in candidate_classes], dtype=np.int64
    )

    valid_ind_pos = np.nonzero(ind_indices >= 0)[0]
    valid_cand_pos = np.nonzero(cand_indices >= 0)[0]
    if valid_ind_pos.size == 0 or valid_cand_pos.size == 0:
        return scores

    valid_ind_ids = ind_indices[valid_ind_pos]
    valid_cand_ids = cand_indices[valid_cand_pos]

    module = model.module
    was_training = getattr(module, "training", False)
    if hasattr(module, "eval"):
        module.eval()

    chunk_rows = max(1, min(int(batch_size), MAX_CHUNK_ROWS))
    orig_device = _resolve_model_device(model)

    def _run_on_device(dev: torch.device) -> None:
        has_rank_classes = hasattr(module, "rank_classes") and callable(
            getattr(module, "rank_classes")
        )
        with torch.no_grad():
            if has_rank_classes:
                # Batch over individuals so (B, n_classes) stays within chunk budget
                n_mod_classes = int(getattr(module, "n_classes", len(cls_dict)) or 1)
                ind_batch_size = max(1, min(256, chunk_rows // max(1, n_mod_classes)))
                for start in range(0, valid_ind_ids.size, ind_batch_size):
                    end = min(start + ind_batch_size, valid_ind_ids.size)
                    b_ind_ids = torch.as_tensor(
                        valid_ind_ids[start:end], dtype=torch.long, device=dev
                    )
                    full_scores = (
                        module.rank_classes(b_ind_ids)
                        .detach()
                        .to(device="cpu", dtype=torch.float64)
                        .numpy()
                    )
                    n_cols = full_scores.shape[1]
                    in_bounds = (valid_cand_ids >= 0) & (valid_cand_ids < n_cols)
                    sub_block = np.full(
                        (end - start, valid_cand_ids.size), np.inf, dtype=np.float64
                    )
                    if np.any(in_bounds):
                        sub_block[:, in_bounds] = full_scores[
                            :, valid_cand_ids[in_bounds]
                        ]
                    scores[
                        np.ix_(valid_ind_pos[start:end], valid_cand_pos)
                    ] = sub_block
            else:
                n_valid_ind = valid_ind_ids.size
                n_valid_cand = valid_cand_ids.size
                total_pairs = n_valid_ind * n_valid_cand
                flat_out = np.empty(total_pairs, dtype=np.float64)

                for start in range(0, total_pairs, chunk_rows):
                    end = min(start + chunk_rows, total_pairs)
                    flat_idx = np.arange(start, end, dtype=np.int64)
                    r_idx = flat_idx // n_valid_cand
                    c_idx = flat_idx % n_valid_cand
                    pair_np = np.stack(
                        [valid_ind_ids[r_idx], valid_cand_ids[c_idx]], axis=1
                    )
                    pair_t = torch.as_tensor(pair_np, dtype=torch.long, device=dev)
                    out_t = module(pair_t, "class_assertion")
                    flat_out[start:end] = (
                        out_t.detach()
                        .reshape(-1)
                        .to(device="cpu", dtype=torch.float64)
                        .numpy()
                    )

                scores[np.ix_(valid_ind_pos, valid_cand_pos)] = flat_out.reshape(
                    n_valid_ind, n_valid_cand
                )

    moved_to_cpu = False
    try:
        try:
            _run_on_device(orig_device)
        except RuntimeError as exc:
            if orig_device.type != "cpu" and hasattr(module, "to"):
                logger.warning(
                    "score_candidates encountered RuntimeError on %s (%s); falling back to CPU.",
                    orig_device,
                    exc,
                )
                module.to("cpu")
                moved_to_cpu = True
                _run_on_device(torch.device("cpu"))
            else:
                raise
    finally:
        if moved_to_cpu and hasattr(module, "to"):
            module.to(orig_device)
        if was_training and hasattr(module, "train"):
            module.train()

    return scores


def _compute_rank_auc(ranks_arr: np.ndarray, num_entities: int) -> float:
    """Compute Riemann rank-ROC AUC from 1-indexed integer ``ranks_arr`` (matching ``mowl.evaluation.base.compute_rank_roc``)."""
    if ranks_arr.size == 0:
        return 0.0
    num_entities = int(num_entities)
    if num_entities <= 1:
        return 1.0
    ranks_0: Dict[int, int] = {}
    unique_r, counts_r = np.unique(ranks_arr, return_counts=True)
    for r_val, cnt in zip(unique_r.tolist(), counts_r.tolist()):
        ranks_0[int(r_val) - 1] = int(cnt)
    min_rank = min(ranks_0.keys())
    all_ranks = {k: 0 for k in range(min_rank, num_entities)}
    all_ranks.update(ranks_0)
    auc_x = sorted(all_ranks.keys())
    if len(auc_x) <= 1:
        return 0.0
    auc_y: List[float] = []
    tpr = 0
    sum_rank = float(sum(all_ranks.values()))
    for x in auc_x:
        tpr += all_ranks[x]
        auc_y.append(tpr / sum_rank)
    dx = np.diff(auc_x)
    heights = np.asarray(auc_y[:-1], dtype=np.float64)
    integral = float(np.sum(heights * dx))
    return float(integral / float(num_entities - 1))


def ranking(
    scores: np.ndarray,
    positives: Sequence[Tuple[int, Optional[int]]],
    skip_mask: Optional[Union[np.ndarray, Mapping[int, Iterable[int]]]] = None,
    *,
    n_unrankable: int = 0,
) -> Dict[str, Any]:
    """Compute filtered candidate-ranking metrics per held-out positive ``(a, C)``.

    Masking rules per held-out positive ``(a, C)``:
    (i)   Candidates labelled ``entailed`` for ``a`` (KB label, passed via ``skip_mask``),
    (ii)  The individual's other held-out positives in the same split,
    (iii) For ``helis``, the individual's skip-list classes (passed via ``skip_mask``).
    **The target candidate ``C`` itself is never masked.**

    Tie-breaking rule (**pessimistic**):
    Among all unmasked candidates (including the target ``C``), any candidate
    whose score is equal to (or lower/more plausible than) the target's score
    counts as ranked at or above the target:
    ``rank = sum(unmasked_scores <= target_score)``.

    Coverage and unrankable positives:
    Held-out positives whose class is not in the candidate list (passed either via
    ``n_unrankable`` or as ``(ind_idx, cand_idx)`` pairs with ``cand_idx is None``
    or ``cand_idx < 0`` or ``cand_idx >= n_candidates``) are counted separately and
    reported in two ways:
    - ``rankable_only`` metrics over the rankable held-out positives;
    - ``all_positives`` metrics in which each unrankable positive is assigned
      ``rank = n_candidates + 1``.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(n_ind, n_cand)`` (lower = more plausible).
    positives : Sequence[Tuple[int, Optional[int]]]
        Held-out positive ``(ind_idx, cand_idx)`` pairs. Pairs with
        ``cand_idx is None`` or outside ``0 .. n_cand - 1`` are counted as
        unrankable positives.
    skip_mask : np.ndarray or Mapping[int, Iterable[int]], optional
        Boolean array of shape ``(n_ind, n_cand)`` (``True`` = masked) or mapping
        from ``ind_idx`` to masked candidate indices (encoding KB-entailed
        candidates and, for ``helis``, the skip list).
    n_unrankable : int, optional
        Additional count of held-out positives whose class is outside the
        candidate set (default: ``0``).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ``mrr``, ``hits@1``, ``hits@5``, ``hits@10``, ``auc``
        (rankable-only), ``all_mrr``, ``all_hits@1``, ``all_hits@5``, ``all_hits@10``,
        ``all_auc`` (all-positives), ``coverage``, ``n_positives_total``,
        ``n_positives_rankable``, ``n_positives_unrankable``, ``ranks``, and
        nested ``rankable_only`` / ``all_positives`` sub-dictionaries.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_ind, n_cand), got shape {scores_arr.shape}")
    n_ind, n_cand = scores_arr.shape

    rankable_positives: List[Tuple[int, int]] = []
    extra_unrankable = int(n_unrankable)
    pos_by_ind: Dict[int, Set[int]] = defaultdict(set)

    for ind_idx, cand_idx in positives:
        i = int(ind_idx)
        if cand_idx is None or int(cand_idx) < 0 or int(cand_idx) >= n_cand:
            extra_unrankable += 1
        elif 0 <= i < n_ind:
            j = int(cand_idx)
            rankable_positives.append((i, j))
            pos_by_ind[i].add(j)
        else:
            extra_unrankable += 1

    mask_2d: Optional[np.ndarray] = None
    mask_map: Optional[Mapping[int, Iterable[int]]] = None
    if skip_mask is not None:
        if isinstance(skip_mask, np.ndarray):
            if skip_mask.shape != (n_ind, n_cand):
                raise ValueError(
                    f"skip_mask shape {skip_mask.shape} does not match scores shape {(n_ind, n_cand)}"
                )
            mask_2d = skip_mask.astype(bool, copy=False)
        else:
            mask_map = skip_mask

    ranks: List[int] = []
    for i, target_j in rankable_positives:
        if mask_2d is not None:
            row_mask = mask_2d[i].copy()
        else:
            row_mask = np.zeros(n_cand, dtype=bool)
            if mask_map is not None and i in mask_map:
                for c_idx in mask_map[i]:
                    if 0 <= int(c_idx) < n_cand:
                        row_mask[int(c_idx)] = True

        # (ii) Mask other held-out positives of individual i in the same split
        for other_j in pos_by_ind[i]:
            if other_j != target_j:
                row_mask[other_j] = True

        # The target itself is NEVER masked
        row_mask[target_j] = False

        row_scores = scores_arr[i]
        target_score = row_scores[target_j]
        unmasked_scores = row_scores[~row_mask]

        if np.isnan(target_score):
            rank = int(unmasked_scores.size)
        else:
            # Pessimistic tie-breaking: every unmasked candidate with score <= target_score
            # (including the target itself) or NaN counts as ranked at or above the target.
            rank = int(
                np.sum((unmasked_scores <= target_score) | np.isnan(unmasked_scores))
            )
        ranks.append(max(1, rank))

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    n_rankable = int(ranks_arr.size)
    n_unrank = int(extra_unrankable)
    n_total = n_rankable + n_unrank
    cov = float(n_rankable / n_total) if n_total > 0 else 1.0

    if n_rankable > 0:
        mrr = float(np.mean(1.0 / ranks_arr))
        h1 = float(np.mean(ranks_arr <= 1))
        h5 = float(np.mean(ranks_arr <= 5))
        h10 = float(np.mean(ranks_arr <= 10))
        auc = _compute_rank_auc(ranks_arr, n_cand)
    else:
        mrr = 0.0
        h1 = 0.0
        h5 = 0.0
        h10 = 0.0
        auc = 0.0

    if n_total > 0:
        unrank_arr = np.full(n_unrank, n_cand + 1, dtype=np.int64)
        all_ranks_arr = (
            np.concatenate([ranks_arr, unrank_arr]) if n_rankable > 0 else unrank_arr
        )
        all_mrr = float(np.mean(1.0 / all_ranks_arr))
        all_h1 = float(np.mean(all_ranks_arr <= 1))
        all_h5 = float(np.mean(all_ranks_arr <= 5))
        all_h10 = float(np.mean(all_ranks_arr <= 10))
        all_auc = _compute_rank_auc(
            all_ranks_arr, max(n_cand, n_cand + (1 if n_unrank > 0 else 0))
        )
    else:
        all_mrr = 0.0
        all_h1 = 0.0
        all_h5 = 0.0
        all_h10 = 0.0
        all_auc = 0.0

    rankable_dict = {
        "mrr": mrr,
        "hits@1": h1,
        "hits@5": h5,
        "hits@10": h10,
        "auc": auc,
    }
    all_dict = {
        "mrr": all_mrr,
        "hits@1": all_h1,
        "hits@5": all_h5,
        "hits@10": all_h10,
        "auc": all_auc,
    }

    return {
        "mrr": mrr,
        "hits@1": h1,
        "hits@5": h5,
        "hits@10": h10,
        "auc": auc,
        "all_mrr": all_mrr,
        "all_hits@1": all_h1,
        "all_hits@5": all_h5,
        "all_hits@10": all_h10,
        "all_auc": all_auc,
        "coverage": cov,
        "n_positives_total": n_total,
        "n_positives_rankable": n_rankable,
        "n_positives_unrankable": n_unrank,
        "ranks": ranks,
        "rankable_only": rankable_dict,
        "all_positives": all_dict,
    }


def fmax(
    scores: np.ndarray,
    true_sets: Union[Sequence[Iterable[Any]], Mapping[int, Iterable[Any]]],
    hierarchy: Mapping[Any, Iterable[Any]],
    *,
    candidates: Optional[Sequence[Any]] = None,
    entailed_mask: Optional[Any] = None,
    labels: Optional[Any] = None,
    n_thresholds: int = 100,
) -> float:
    """Compute CAFA-style protein-centric :math:`F_{\\max}` over ancestor-closed annotations.

    Both ``true_sets`` and thresholded predictions are closed under ``hierarchy``
    (where ``hierarchy[c]`` gives the superclasses ``{a : c ⊑ a}`` of ``c``).
    KB-entailed cells (passed via ``entailed_mask`` or ``labels`` where
    ``label == "entailed"`` / code ``1``) are excluded from **both** the
    prediction set and the truth set (matching the filtered ranking protocol) so
    that a model that scores KB-entailed candidates as plausible is not penalised.
    Because lower ``scores`` indicate higher plausibility, a non-entailed
    candidate ``c`` is predicted at threshold ``tau`` iff its propagated score is
    ``<= tau`` (and finite). Thresholds are evaluated at ``n_thresholds``
    quantiles of the finite non-entailed scores.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(n_ind, n_cand)`` (lower = more plausible).
    true_sets : Sequence[Iterable[Any]] or Mapping[int, Iterable[Any]]
        Per-individual held-out true classes (either candidate indices
        ``0 .. n_cand - 1`` or class IRIs when ``candidates`` or ``hierarchy`` uses IRIs).
    hierarchy : Mapping[Any, Iterable[Any]]
        Mapping from each class ``c`` to its ancestor classes ``{a : c ⊑ a}``.
    candidates : Sequence[Any], optional
        Ordered candidate class identifiers matching the columns of ``scores`` when
        ``true_sets`` / ``hierarchy`` are keyed by class IRI instead of column index.
    entailed_mask : np.ndarray, Sequence[Iterable[Any]], or Mapping[int, Iterable[Any]], optional
        Boolean mask of shape ``(n_ind, n_cand)`` (``True`` = KB-entailed), label
        array, or per-individual iterable of KB-entailed candidate classes/indices
        to exclude from both prediction and truth sets.
    labels : array-like, optional
        Alternative to ``entailed_mask``: ``(n_ind, n_cand)`` label matrix where
        entries with ``label == "entailed"`` (or code ``1``) are excluded from
        both prediction and truth sets.
    n_thresholds : int, optional
        Number of quantile thresholds to evaluate (default: ``100``).

    Returns
    -------
    float
        Maximum protein-centric :math:`F_1` across the ``n_thresholds`` score
        quantiles in ``[0.0, 1.0]``.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_ind, n_cand), got shape {scores_arr.shape}")
    n_ind, n_cand = scores_arr.shape
    if n_ind == 0 or n_cand == 0:
        return 0.0

    cand_to_idx: Dict[Any, int] = {}
    if candidates is not None:
        for idx, c in enumerate(candidates):
            cand_to_idx[c] = idx
    for idx in range(n_cand):
        cand_to_idx.setdefault(idx, idx)

    def _to_cand_idx(c: Any) -> int:
        if c in cand_to_idx:
            return cand_to_idx[c]
        if isinstance(c, (int, np.integer)) and 0 <= int(c) < n_cand:
            return int(c)
        return -1

    # Build index-based ancestor mapping for candidate columns 0 .. n_cand - 1
    anc_by_cand: List[Set[int]] = [{j} for j in range(n_cand)]
    for raw_c, raw_ancs in hierarchy.items():
        c_idx = _to_cand_idx(raw_c)
        if 0 <= c_idx < n_cand:
            for raw_a in raw_ancs:
                a_idx = _to_cand_idx(raw_a)
                if 0 <= a_idx < n_cand:
                    anc_by_cand[c_idx].add(a_idx)

    # Transitive closure over candidate indices (in case hierarchy only gave direct parents)
    for j in range(n_cand):
        stack = list(anc_by_cand[j])
        closed = set(stack)
        while stack:
            curr = stack.pop()
            for nxt in anc_by_cand[curr]:
                if nxt not in closed:
                    closed.add(nxt)
                    stack.append(nxt)
        anc_by_cand[j] = closed

    # Resolve KB-entailed mask of shape (n_ind, n_cand)
    ent_mat = np.zeros((n_ind, n_cand), dtype=bool)
    ent_source = entailed_mask if entailed_mask is not None else labels
    if ent_source is not None:
        if isinstance(ent_source, Mapping):
            for i_key, items in ent_source.items():
                i_idx = int(i_key)
                if 0 <= i_idx < n_ind and items is not None:
                    for item in items:
                        c_idx = _to_cand_idx(item)
                        if 0 <= c_idx < n_cand:
                            ent_mat[i_idx, c_idx] = True
        else:
            arr_try: Optional[np.ndarray] = None
            try:
                arr_cand = np.asarray(ent_source)
                if arr_cand.size == n_ind * n_cand:
                    if arr_cand.dtype != object:
                        arr_try = arr_cand.reshape(n_ind, n_cand)
                    else:
                        first_el = arr_cand.flat[0] if arr_cand.size > 0 else None
                        if isinstance(
                            first_el, (str, bytes, int, np.integer, bool, np.bool_)
                        ):
                            arr_try = arr_cand.reshape(n_ind, n_cand)
            except ValueError:
                arr_try = None

            if arr_try is not None:
                if arr_try.dtype == bool or np.issubdtype(arr_try.dtype, np.bool_):
                    ent_mat = arr_try.astype(bool, copy=True)
                else:
                    ent_mat = (
                        _normalise_label_codes(arr_try, (n_ind, n_cand)).reshape(
                            n_ind, n_cand
                        )
                        == 1
                    )
            else:
                for i_idx, items in enumerate(ent_source):
                    if i_idx >= n_ind or items is None:
                        continue
                    for item in items:
                        c_idx = _to_cand_idx(item)
                        if 0 <= c_idx < n_cand:
                            ent_mat[i_idx, c_idx] = True

    # Build true indicator matrix T of shape (n_eval, n_cand), ancestor-closed
    # and excluding KB-entailed cells.
    true_rows: List[np.ndarray] = []
    eval_ind_indices: List[int] = []
    for i in range(n_ind):
        if isinstance(true_sets, Mapping):
            raw_t = true_sets.get(i, ())
        else:
            raw_t = true_sets[i] if i < len(true_sets) else ()

        closed_t: Set[int] = set()
        for item in raw_t:
            c_idx = _to_cand_idx(item)
            if 0 <= c_idx < n_cand:
                closed_t.update(anc_by_cand[c_idx])
            # Also check if item (e.g. a GO class outside candidates) has ancestors in candidates
            if item in hierarchy:
                for raw_a in hierarchy[item]:
                    a_idx = _to_cand_idx(raw_a)
                    if 0 <= a_idx < n_cand:
                        closed_t.update(anc_by_cand[a_idx])

        if closed_t:
            row_bool = np.zeros(n_cand, dtype=bool)
            row_bool[list(closed_t)] = True
            row_bool[ent_mat[i]] = False
            if np.any(row_bool):
                true_rows.append(row_bool)
                eval_ind_indices.append(i)

    if not true_rows:
        return 0.0

    true_mat = np.stack(true_rows, axis=0)  # (n_eval, n_cand)
    true_counts = true_mat.sum(axis=1).astype(np.float64)  # (n_eval,)
    eval_ent_mat = ent_mat[eval_ind_indices]  # (n_eval, n_cand)
    eval_scores = np.where(
        eval_ent_mat, np.inf, scores_arr[eval_ind_indices]
    )  # (n_eval, n_cand)

    # Propagate scores upward: prop_scores[p, a] = min_{c : a in anc_by_cand[c]} eval_scores[p, c]
    descendants_of_a: List[List[int]] = [[] for _ in range(n_cand)]
    for c_idx in range(n_cand):
        for a_idx in anc_by_cand[c_idx]:
            descendants_of_a[a_idx].append(c_idx)

    prop_scores = np.empty_like(eval_scores)
    for a_idx in range(n_cand):
        descs = descendants_of_a[a_idx]
        if len(descs) == 1:
            prop_scores[:, a_idx] = eval_scores[:, descs[0]]
        else:
            prop_scores[:, a_idx] = np.nanmin(eval_scores[:, descs], axis=1)

    # Exclude KB-entailed cells from predictions
    prop_scores[eval_ent_mat] = np.inf

    finite_vals = eval_scores[np.isfinite(eval_scores)]
    if finite_vals.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, max(1, int(n_thresholds)))
    thresholds = np.quantile(finite_vals, quantiles)

    best_f1 = 0.0
    for tau in np.unique(thresholds):
        pred_mat = np.isfinite(prop_scores) & (prop_scores <= tau)  # (n_eval, n_cand)
        pred_counts = pred_mat.sum(axis=1).astype(np.float64)
        tp_counts = (pred_mat & true_mat).sum(axis=1).astype(np.float64)

        recall = float(np.mean(tp_counts / true_counts))
        has_pred = pred_counts > 0
        if np.any(has_pred):
            precision = float(np.mean(tp_counts[has_pred] / pred_counts[has_pred]))
        else:
            precision = 0.0

        if precision + recall > 0.0:
            f1 = (2.0 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = float(f1)

    return float(best_f1)


def hierarchy_violation_rate(
    scores: np.ndarray,
    entailed_pairs: Sequence[Tuple[int, int]],
    *,
    n_samples: int = 10_000,
    seed: int = 0,
    tol: float = 1e-6,
) -> Optional[float]:
    """Compute the hierarchy violation rate over entailed candidate subsumptions.

    Samples up to ``n_samples`` (default 10,000) instances ``(a, G ⊑ G')`` with
    ``G != G'`` and the subsumption entailed. An instance is a violation iff
    ``score(a, G)`` is more plausible (strictly smaller) than ``score(a, G')`` by
    more than ``tol`` (``1e-6``):
    ``score(a, G) < score(a, G') - 1e-6``.
    Returns ``None`` (serialised as JSON ``null``) when fewer than 50 instances exist.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(n_ind, n_cand)`` (lower = more plausible).
    entailed_pairs : Sequence[Tuple[int, int]]
        Entailed candidate index pairs ``(g_sub_idx, g_sup_idx)`` with ``G ⊑ G'``.
        Any reflexive pairs ``G == G'`` or out-of-range pairs are filtered out.
    n_samples : int, optional
        Maximum number of ``(a, G, G')`` instances to sample (default: ``10_000``).
    seed : int, optional
        Random seed for deterministic sampling (default: ``0``).
    tol : float, optional
        Strict plausibility tolerance (default: ``1e-6``).

    Returns
    -------
    float or None
        Fraction of sampled ``(a, G ⊑ G')`` instances violating hierarchy
        monotonicity, or ``None`` if fewer than 50 instances exist.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_ind, n_cand), got shape {scores_arr.shape}")
    n_ind, n_cand = scores_arr.shape
    if n_ind == 0 or n_cand == 0:
        return None

    # Keep unique strict entailed pairs (g_sub != g_sup) within valid candidate bounds
    strict_pairs_set: Set[Tuple[int, int]] = set()
    for g_sub, g_sup in entailed_pairs:
        sub_i, sup_i = int(g_sub), int(g_sup)
        if sub_i != sup_i and 0 <= sub_i < n_cand and 0 <= sup_i < n_cand:
            strict_pairs_set.add((sub_i, sup_i))

    strict_pairs = sorted(strict_pairs_set)
    n_pairs = len(strict_pairs)
    total_instances = n_ind * n_pairs
    if n_pairs == 0 or total_instances < 50:
        return None

    pairs_arr = np.asarray(strict_pairs, dtype=np.int64)  # (n_pairs, 2)
    rng = np.random.default_rng(seed)

    if total_instances <= n_samples:
        ind_idx = np.repeat(np.arange(n_ind, dtype=np.int64), n_pairs)
        pair_idx = np.tile(np.arange(n_pairs, dtype=np.int64), n_ind)
    else:
        flat_choices = rng.choice(total_instances, size=int(n_samples), replace=False)
        ind_idx = flat_choices // n_pairs
        pair_idx = flat_choices % n_pairs

    g_sub_col = pairs_arr[pair_idx, 0]
    g_sup_col = pairs_arr[pair_idx, 1]

    s_sub = scores_arr[ind_idx, g_sub_col]
    s_sup = scores_arr[ind_idx, g_sup_col]

    # Violation iff score(a, G) is more plausible (smaller) than score(a, G') by > tol.
    # Handle +inf safely: if both are +inf, neither is more plausible.
    violations = np.isfinite(s_sub) & ((s_sup - s_sub) > tol)
    return float(np.mean(violations))


def _normalise_label_codes(labels: Any, expected_shape: Tuple[int, ...]) -> np.ndarray:
    """Convert string or integer labels into codes: ``1`` = entailed (+), ``0`` = unknown (?), ``2`` = contradicted (-)."""
    arr = np.asarray(labels)
    if arr.size != int(np.prod(expected_shape)):
        raise ValueError(
            f"labels size {arr.size} does not match scores shape {expected_shape}"
        )
    flat = arr.reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        codes = np.zeros(flat.size, dtype=np.int8)
        codes[flat == 1] = 1
        codes[(flat == 2) | (flat == -1)] = 2
        return codes

    s_flat = flat.astype(str)
    codes = np.zeros(s_flat.size, dtype=np.int8)
    codes[np.isin(s_flat, ["entailed", "entailed_gold", "+", "pos", "positive"])] = 1
    codes[
        np.isin(
            s_flat, ["contradicted", "contradicted_gold", "-", "neg", "negative"]
        )
    ] = 2
    return codes


def _mann_whitney_auc(s_better: np.ndarray, s_worse: np.ndarray) -> float:
    """Compute Mann-Whitney ROC AUC ``P[s_better < s_worse] + 0.5 * P[s_better == s_worse]``."""
    # Negate scores so higher = more plausible, handling +inf / -inf cleanly via rankdata
    combined = np.concatenate([-s_better, -s_worse])
    ranks = rankdata(combined, method="average")
    n1 = len(s_better)
    n2 = len(s_worse)
    u_stat = float(np.sum(ranks[:n1])) - n1 * (n1 + 1) / 2.0
    return float(u_stat / (n1 * n2))


def three_way_metrics(
    scores: np.ndarray,
    labels: Any,
    *,
    seed: int = 0,
    n_sample: int = 5000,
) -> Dict[str, Any]:
    """Compute three-way ABox faithfulness metrics over ``entailed`` (+), ``unknown`` (?), ``contradicted`` (-).

    Follows the exact sampling and ``< 50`` null conventions of
    :func:`experiments.metrics.faithfulness_metrics`:
    - Samples up to ``n_sample`` (default 5,000) ``(individual, candidate)`` cells
      **pooled across evaluation individuals** from each of ``+``, ``?``, ``-``.
    - ``auc_pos_und`` (``AUC(+/?)``) is computed over the pooled ``+`` vs. ``?``
      samples whenever ``n_pos >= 50`` and ``n_und >= 50`` (and is ``None``
      otherwise), even when ``n_neg < 50`` (as in B1 ``yeast-function`` where
      ``n_neg == 0`` by design).
    - ``auc_pos_neg`` (``AUC(+/-)``), ``pair_fvr``, and ``fvr`` require
      ``n_neg >= 50``, ``n_pos >= 50``, and ``n_und >= 50`` (returning ``None``
      when fewer than 50 negatives, positives, or unknowns exist).

    Parameters
    ----------
    scores : np.ndarray
        Plausibility scores of shape ``(n_ind, n_cand)`` or 1-D ``(N,)``
        (lower = more plausible).
    labels : array-like
        Matching labels with values in ``{"entailed", "unknown", "contradicted"}``
        or ``{"entailed_gold", "unknown", "contradicted_gold"}`` (or integer codes
        ``1`` = entailed, ``0`` = unknown, ``2`` / ``-1`` = contradicted).
    seed : int, optional
        RNG seed for sampling (default: ``0``).
    n_sample : int, optional
        Maximum sample size per partition (default: ``5000``).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``auc_pos_neg``, ``auc_pos_und``, ``pair_fvr``,
        ``fvr``, ``n_pos``, ``n_und``, ``n_neg``.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    flat_scores = scores_arr.reshape(-1)
    codes = _normalise_label_codes(labels, scores_arr.shape)

    pos_idx = np.nonzero(codes == 1)[0]
    und_idx = np.nonzero(codes == 0)[0]
    neg_idx = np.nonzero(codes == 2)[0]

    n_pos_total = int(pos_idx.size)
    n_und_total = int(und_idx.size)
    n_neg_total = int(neg_idx.size)

    rng = np.random.default_rng(seed)

    def _sample_scores(idx_arr: np.ndarray) -> np.ndarray:
        if idx_arr.size == 0:
            return np.empty(0, dtype=np.float64)
        if idx_arr.size > n_sample:
            chosen = rng.choice(idx_arr, size=int(n_sample), replace=False)
        else:
            chosen = idx_arr
        return flat_scores[chosen]

    s_pos = _sample_scores(pos_idx)
    s_und = _sample_scores(und_idx)
    s_neg = _sample_scores(neg_idx)

    # Replace NaN with +inf so rankdata / comparisons remain well-defined
    if s_pos.size > 0 and np.any(np.isnan(s_pos)):
        s_pos = np.where(np.isnan(s_pos), np.inf, s_pos)
    if s_und.size > 0 and np.any(np.isnan(s_und)):
        s_und = np.where(np.isnan(s_und), np.inf, s_und)
    if s_neg.size > 0 and np.any(np.isnan(s_neg)):
        s_neg = np.where(np.isnan(s_neg), np.inf, s_neg)

    if n_pos_total >= 50 and n_und_total >= 50:
        auc_pos_und: Optional[float] = _mann_whitney_auc(s_pos, s_und)
    else:
        auc_pos_und = None

    if n_neg_total < 50 or n_pos_total < 50 or n_und_total < 50:
        return {
            "auc_pos_neg": None,
            "auc_pos_und": auc_pos_und,
            "pair_fvr": None,
            "fvr": None,
            "n_pos": n_pos_total,
            "n_und": n_und_total,
            "n_neg": n_neg_total,
        }

    auc_pos_neg = _mann_whitney_auc(s_pos, s_neg)

    n_draws = 100_000
    u_idx = rng.integers(0, len(s_und), size=n_draws)
    n_idx = rng.integers(0, len(s_neg), size=n_draws)
    pair_fvr = float(np.mean(s_und[u_idx] >= s_neg[n_idx]))

    p_idx = rng.integers(0, len(s_pos), size=n_draws)
    u_idx2 = rng.integers(0, len(s_und), size=n_draws)
    n_idx2 = rng.integers(0, len(s_neg), size=n_draws)
    strict_chain = (s_neg[n_idx2] > s_und[u_idx2]) & (s_und[u_idx2] > s_pos[p_idx])
    fvr = float(np.mean(~strict_chain))

    return {
        "auc_pos_neg": auc_pos_neg,
        "auc_pos_und": auc_pos_und,
        "pair_fvr": pair_fvr,
        "fvr": fvr,
        "n_pos": n_pos_total,
        "n_und": n_und_total,
        "n_neg": n_neg_total,
    }


def abox_modelness(
    model: Any,
    *,
    tol: float = 1e-6,
    batch_size: int = 32768,
    margin: float = 0.0,
) -> Optional[float]:
    """Compute the fraction of training ABox assertions whose strict loss is ``<= 1e-6``.

    Evaluates all training ``class_assertion`` rows (and ``object_property_assertion``
    rows when the module implements ``object_property_assertion_loss``) with
    ``margin`` / ``gamma`` temporarily set to ``0.0``, restoring all margin
    attributes and the module's training mode in a ``finally`` block following
    :func:`experiments.metrics.modelness`.

    Parameters
    ----------
    model : EmbeddingELModel
        Trained mOWL EL model.
    tol : float, optional
        Strict tolerance for zero loss (default: ``1e-6``).
    batch_size : int, optional
        Evaluation chunk size (default: ``32768``).
    margin : float, optional
        Strict evaluation margin (default: ``0.0``).

    Returns
    -------
    float or None
        Fraction of training assertions satisfied within ``tol`` in ``[0.0, 1.0]``,
        or ``None`` when no training assertion was scored.
    """
    module = model.module
    saved_attrs: Dict[Tuple[str, str], float] = {}
    for target_name, target_obj in (("module", module), ("model", model)):
        for attr in ("gamma", "margin"):
            if hasattr(target_obj, attr) and isinstance(
                getattr(target_obj, attr), (int, float)
            ):
                saved_attrs[(target_name, attr)] = float(getattr(target_obj, attr))
                setattr(target_obj, attr, float(margin))

    was_training = getattr(module, "training", False)
    if hasattr(module, "eval"):
        module.eval()

    dev = _resolve_model_device(model)
    total_sat = 0
    total_cnt = 0

    try:
        training_datasets: Mapping[str, Any] = (
            getattr(model, "training_datasets", {}) or {}
        )
        with torch.no_grad():
            for assertion_name in ("class_assertion", "object_property_assertion"):
                ds = training_datasets.get(assertion_name)
                if ds is None or len(ds) == 0:
                    continue
                data = ds[:]
                n = int(data.shape[0])
                if n == 0:
                    continue

                for start in range(0, n, batch_size):
                    chunk = data[start : start + batch_size].to(dev)
                    try:
                        losses = module(chunk, assertion_name)
                    except NotImplementedError:
                        break
                    if isinstance(losses, Exception) or losses is NotImplemented:
                        break
                    losses_np = (
                        losses.detach().reshape(-1).to(device="cpu").numpy()
                    )
                    total_sat += int(np.sum(losses_np <= tol))
                    total_cnt += int(losses_np.size)
    finally:
        for (target_name, attr), val in saved_attrs.items():
            target_obj = module if target_name == "module" else model
            setattr(target_obj, attr, val)
        if was_training and hasattr(module, "train"):
            module.train()

    return float(total_sat / total_cnt) if total_cnt > 0 else None


def _file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_candidate_hierarchy(
    ontology_path: Union[str, Path],
    candidates: Sequence[str],
    *,
    cache_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """Compute or load the cached ELK subsumption closure over ``candidates``.

    For each class ``c`` in ``candidates`` (and, for ``yeast-function``, every
    ``Aux_G`` / ``GO_*`` class in the ontology), computes the set of candidate
    superclasses ``{d in candidates : c ⊑ d}`` using ELK through OWLAPI/mOWL,
    caching the result as JSON next to ``ontology.owl``.
    """
    ont_p = Path(ontology_path).resolve()
    c_path = (
        Path(cache_path).resolve()
        if cache_path is not None
        else ont_p.parent / "candidate_closure.json"
    )
    ont_sha = _file_sha256(ont_p)
    cand_sha = hashlib.sha256("\n".join(candidates).encode("utf-8")).hexdigest()

    if c_path.exists():
        try:
            cached = json.loads(c_path.read_text(encoding="utf-8"))
            if (
                cached.get("ontology_sha256") == ont_sha
                and cached.get("candidates_sha256") == cand_sha
                and isinstance(cached.get("hierarchy"), dict)
            ):
                return cached["hierarchy"]
        except Exception as exc:
            logger.warning("Failed to read cached closure %s (%s); recomputing.", c_path, exc)

    import mowl

    mowl.init_jvm("8g")
    import java
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.reasoner import InferenceType

    mgr = OWLManager.createOWLOntologyManager()
    ont = mgr.loadOntologyFromOntologyDocument(java.io.File(str(ont_p)))
    df = mgr.getOWLDataFactory()
    from org.semanticweb.owlapi.model import IRI

    cand_set = set(candidates)
    aux_prefix = "http://mowl.borg/yeast-function/has_function/"
    go_prefix = "http://purl.obolibrary.org/obo/GO_"

    # Also map GO IRIs <-> Aux_G IRIs when candidates use Aux_G
    go_to_aux: Dict[str, str] = {}
    for c in candidates:
        if c.startswith(aux_prefix):
            local_id = c[len(aux_prefix) :]
            go_to_aux[f"http://purl.obolibrary.org/obo/{local_id}"] = c

    rf = ElkReasonerFactory()
    reasoner = rf.createReasoner(ont)
    hierarchy: Dict[str, List[str]] = {}
    try:
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
        # Include all ontology classes that are either in cand_set or GO_* (for unrankable held-out leaves)
        query_classes: Set[str] = set(candidates)
        if go_to_aux:
            for cls_obj in ont.getClassesInSignature():
                cls_iri = str(cls_obj.toStringID())
                if cls_iri.startswith(go_prefix) or cls_iri.startswith(aux_prefix):
                    query_classes.add(cls_iri)

        for c_iri in sorted(query_classes):
            owl_cls = df.getOWLClass(IRI.create(c_iri))
            sups_raw = {
                str(s.toStringID())
                for s in reasoner.getSuperClasses(owl_cls, False).getFlattened()
            }
            eqs_raw = {
                str(s.toStringID())
                for s in reasoner.getEquivalentClasses(owl_cls).getEntities()
            }
            all_raw = sups_raw | eqs_raw | {c_iri}
            mapped_cands: Set[str] = set()
            for s_iri in all_raw:
                if s_iri in cand_set:
                    mapped_cands.add(s_iri)
                elif s_iri in go_to_aux:
                    mapped_cands.add(go_to_aux[s_iri])
            if c_iri in cand_set:
                mapped_cands.add(c_iri)
            if mapped_cands:
                hierarchy[c_iri] = sorted(mapped_cands)
                if c_iri.startswith(go_prefix):
                    local_id = c_iri.rsplit("/", 1)[-1]
                    hierarchy.setdefault(f"{aux_prefix}{local_id}", sorted(mapped_cands))
    finally:
        reasoner.dispose()

    try:
        c_path.write_text(
            json.dumps(
                {
                    "ontology_sha256": ont_sha,
                    "candidates_sha256": cand_sha,
                    "hierarchy": hierarchy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write closure cache %s: %s", c_path, exc)

    return hierarchy


def _extract_owl_class_assertions(owl_ont: Any) -> List[Tuple[str, str]]:
    """Extract sorted ``(individual_iri, class_iri)`` pairs from an OWLAPI ontology."""
    from org.semanticweb.owlapi.model import AxiomType

    pairs: List[Tuple[str, str]] = []
    for ax in owl_ont.getAxioms(AxiomType.CLASS_ASSERTION):
        ind = ax.getIndividual()
        cls_expr = ax.getClassExpression()
        if not ind.isAnonymous() and not cls_expr.isAnonymous():
            ind_iri = str(ind.asOWLNamedIndividual().getIRI().toString())
            cls_iri = str(cls_expr.asOWLClass().getIRI().toString())
            pairs.append((ind_iri, cls_iri))
    pairs.sort()
    return pairs


def load_abox_split_data(
    bundle: Any,
    split: str = "test",
) -> Dict[str, Any]:
    """Load and cache in-memory evaluation structures for ``split`` (``"valid"`` or ``"test"``)."""
    if split not in ("valid", "test"):
        raise ValueError(f"split must be 'valid' or 'test', got {split!r}")

    cache_attr = f"_abox_split_cache_{split}"
    cached = getattr(bundle, cache_attr, None)
    if cached is not None:
        return cached

    import pyarrow.parquet as pq

    cand_path = Path(bundle.paths["candidates"])
    candidates = [
        line.strip()
        for line in cand_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_cand = len(candidates)
    cand_to_idx = {c: j for j, c in enumerate(candidates)}

    parquet_path = Path(bundle.paths[f"labels_{split}"])
    table = pq.read_table(str(parquet_path))
    n_rows = table.num_rows
    if n_cand == 0 or n_rows % n_cand != 0:
        raise ValueError(
            f"Parquet {parquet_path} has {n_rows} rows, not divisible by n_cand={n_cand}"
        )
    n_ind = n_rows // n_cand

    ind_col = table["individual"].to_numpy(zero_copy_only=False)
    eval_individuals: List[str] = [str(x) for x in ind_col[::n_cand]]
    ind_to_idx = {ind: i for i, ind in enumerate(eval_individuals)}

    label_col = table["label"].to_numpy(zero_copy_only=False)
    label_codes = _normalise_label_codes(label_col, (n_ind, n_cand)).reshape(
        n_ind, n_cand
    )

    if "label_gold" in table.column_names:
        gold_col = table["label_gold"].to_numpy(zero_copy_only=False)
        label_gold_codes: Optional[np.ndarray] = _normalise_label_codes(
            gold_col, (n_ind, n_cand)
        ).reshape(n_ind, n_cand)
    else:
        label_gold_codes = None

    # Load held-out positive assertions
    annot_key = f"{split}_annotations"
    aux_prefix = "http://mowl.borg/yeast-function/has_function/"
    raw_pos_pairs: List[Tuple[str, str]] = []
    if annot_key in bundle.paths and Path(bundle.paths[annot_key]).exists():
        for raw_line in (
            Path(bundle.paths[annot_key]).read_text(encoding="utf-8").splitlines()
        ):
            line = raw_line.strip()
            if not line:
                continue
            p_iri, g_iri = line.split("\t")
            local_id = g_iri.rsplit("/", 1)[-1]
            aux_iri = (
                f"{aux_prefix}{local_id}"
                if local_id.startswith("GO_")
                else g_iri
            )
            raw_pos_pairs.append((p_iri, aux_iri))
    else:
        owl_ont = (
            bundle.dataset.validation
            if split == "valid"
            else bundle.dataset.testing
        )
        raw_pos_pairs = _extract_owl_class_assertions(owl_ont)

    positives: List[Tuple[int, Optional[int]]] = []
    true_sets: List[List[str]] = [[] for _ in range(n_ind)]
    for ind_iri, cls_iri in raw_pos_pairs:
        i = ind_to_idx.get(ind_iri, -1)
        if i < 0:
            positives.append((-1, None))
            continue
        true_sets[i].append(cls_iri)
        j = cand_to_idx.get(cls_iri, None)
        positives.append((i, j))

    # Build skip_mask: (i) KB-entailed candidates + (iii) benchmark skip list
    skip_mask = (label_codes == 1).copy()
    skip_key = f"skip_{split}"
    if skip_key in bundle.paths and Path(bundle.paths[skip_key]).exists():
        for raw_line in (
            Path(bundle.paths[skip_key]).read_text(encoding="utf-8").splitlines()
        ):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            ind_iri = parts[0].strip()
            i = ind_to_idx.get(ind_iri, -1)
            if i < 0 or len(parts) < 2 or not parts[1].strip():
                continue
            for c_str in parts[1].split(","):
                c_clean = c_str.strip()
                j = cand_to_idx.get(c_clean, -1)
                if j >= 0:
                    skip_mask[i, j] = True

    result = {
        "individuals": eval_individuals,
        "candidates": candidates,
        "cand_to_idx": cand_to_idx,
        "positives": positives,
        "true_sets": true_sets,
        "skip_mask": skip_mask,
        "label_codes": label_codes,
        "label_gold_codes": label_gold_codes,
    }
    try:
        setattr(bundle, cache_attr, result)
    except Exception:
        pass
    return result


def compute_validation_mrr(model: Any, bundle: Any) -> float:
    """Compute validation rankable-only MRR on ``bundle`` for checkpoint selection."""
    split_data = load_abox_split_data(bundle, split="valid")
    scores = score_candidates(
        model, split_data["individuals"], split_data["candidates"]
    )
    rank_res = ranking(
        scores,
        split_data["positives"],
        skip_mask=split_data["skip_mask"],
    )
    return float(rank_res["mrr"])

