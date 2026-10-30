"""Evaluation metrics for learnable geometric embedding experiments (Task 7, Spec §5.3).

Implements:
- ``ranking_metrics(model, bundle, device)``: filtered and unfiltered subsumption
  ranking metrics via :class:`~mowl.evaluation.SubsumptionEvaluator`, prefixed
  with ``rank_``.
- ``faithfulness_metrics(model, bundle, cm, seed=0, n_sample=5000)``: ordering of
  entailed (``T+``), undetermined (``T?``), and contradicted (``T-``) named-class
  subsumption pairs, reporting ``auc_pos_neg``, ``pair_fvr``, ``fvr``, ``n_pos``,
  ``n_und``, ``n_neg`` (with ``None`` ordering metrics when ``n_neg < 50``).
- ``modelness(model)``: fraction of training normalised axioms per GCI normal form
  (``modelness_<gci>``) and overall (``modelness_all``) whose loss is ``<= 1e-6``
  when the module's margin (``margin`` for ELBE, ``gamma`` for Box²EL/MultiBoxEL)
  is temporarily set to ``0`` and restored afterwards.
- ``helly_metrics(model, bundle, cm, certificates)``: ``helly_violation_rate``,
  ``degeneracy_rate``, and ``n_certificates`` over :class:`~mowl.geometric.HellyCertificate`
  instances (supporting arbitrary tuple lengths in ``c1``, ``c2``, ``c3``).
- ``cost_metrics(model, timings)``: parameter count, epoch and total training
  times, ``n_components``, mean effective components, and peak GPU memory.
- ``class_regions(model)``: ``(centers, offsets)`` NumPy arrays of shape
  ``(n_classes, k, d)`` with non-negative ``offsets`` for ELBE, Box²EL (``k = 1``),
  and MultiBoxEL (``k = n_components``).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import rankdata
import torch

from mowl.geometric.helly import HellyCertificate, certificate_triangles
from mowl.geometric.normal_forms import BOTTOM, NF2, TOP

from experiments.cm_cache import get_canonical_model
from experiments.datasets import DatasetBundle

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric import CanonicalModel

AUXILIARY_IRI_PREFIXES = (
    "http://mowl.borg/aux/",
    "http://mowl.borg/aux#",
)


@dataclass(frozen=True)
class PrismRegion:
    """Equilateral-triangle × box prism region(s) with support-function and corner arithmetic.

    Parameters
    ----------
    supports : np.ndarray
        Triangle support values of shape ``(..., 3)`` along the 3 outward unit normals.
    lower : np.ndarray
        Box lower corners of shape ``(..., d_box)``.
    upper : np.ndarray
        Box upper corners of shape ``(..., d_box)``.
    """

    supports: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.supports
        yield self.lower
        yield self.upper

    def __getitem__(self, idx: Any) -> "PrismRegion":
        return PrismRegion(
            supports=self.supports[idx],
            lower=self.lower[idx],
            upper=self.upper[idx],
        )

    def intersect(self, other: "PrismRegion") -> "PrismRegion":
        """Return the intersection of ``self`` and ``other``."""
        return PrismRegion(
            supports=np.minimum(self.supports, other.supports),
            lower=np.maximum(self.lower, other.lower),
            upper=np.minimum(self.upper, other.upper),
        )

    @classmethod
    def intersect_all(cls, regions: Sequence["PrismRegion"]) -> "PrismRegion":
        """Return the common intersection of a non-empty sequence of prism regions."""
        if not regions:
            raise ValueError("PrismRegion.intersect_all requires at least one region.")
        acc = regions[0]
        for reg in regions[1:]:
            acc = acc.intersect(reg)
        return acc

    def is_empty(self, tol: float = 1e-6) -> bool:
        """Return ``True`` iff the prism region is empty up to tolerance ``tol``."""
        tri_nonempty = bool(np.all(np.sum(self.supports, axis=-1) >= -tol))
        box_nonempty = bool(np.all(self.lower <= self.upper + tol))
        return not (tri_nonempty and box_nonempty)

    def is_nonempty(self, tol: float = 1e-6) -> bool:
        """Return ``True`` iff the prism region is non-empty up to tolerance ``tol``."""
        return not self.is_empty(tol=tol)

    def intersects(self, other: "PrismRegion", tol: float = 1e-6) -> bool:
        """Return ``True`` iff ``self`` and ``other`` have non-empty intersection."""
        return not self.intersect(other).is_empty(tol=tol)

    def is_degenerate(self, deg_eps: float = 1e-3) -> bool:
        """Return ``True`` iff ``sum(supports) <= deg_eps`` or any box side ``<= deg_eps``."""
        tri_deg = bool(np.any(np.sum(self.supports, axis=-1) <= deg_eps))
        box_deg = bool(np.any((self.upper - self.lower) <= deg_eps))
        return tri_deg or box_deg


Prism = PrismRegion


def _is_named_class(iri: str) -> bool:
    """Return ``True`` iff ``iri`` is a named ontology class (not ``TOP``, ``BOTTOM``, or auxiliary)."""
    if iri in (TOP, BOTTOM):
        return False
    if iri.startswith(AUXILIARY_IRI_PREFIXES):
        return False
    return True


def _module_device(module: torch.nn.Module) -> torch.device:
    """Return the device of ``module``'s first parameter (defaulting to CPU)."""
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def class_regions(model: Any) -> Union[Tuple[np.ndarray, np.ndarray], PrismRegion]:
    """Extract class regions from ``model``.

    Parameters
    ----------
    model : mOWL model or torch.nn.Module
        An ``ELBE``, ``BoxSquaredEL``, ``MultiBoxEL``, ``MultiELBE``, or ``PrismEL``
        model (or its underlying ``.module``).

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray], PrismRegion]
        - For box/multi-box models: ``(centers, offsets)`` float32 arrays of shape
          ``(n_classes, k, d)`` with ``offsets >= 0``.
        - For ``PrismEL`` / ``PrismModule``: a :class:`PrismRegion` with ``supports``
          of shape ``(n_classes, 3)`` and ``lower``, ``upper`` of shape
          ``(n_classes, d - 2)``.
    """
    module = getattr(model, "module", model)

    if (
        hasattr(module, "tri_center")
        and hasattr(module, "tri_inradius")
        and hasattr(module, "box_center")
        and hasattr(module, "box_offset")
    ):
        from mowl.nn.el.prism.losses import supports as _prism_supports

        with torch.no_grad():
            sup_t = _prism_supports(
                module.tri_center.weight.detach(),
                torch.abs(module.tri_inradius.weight.detach()),
            )
        sup_np = sup_t.cpu().numpy().astype(np.float32, copy=False)
        box_c = module.box_center.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        box_o = np.abs(module.box_offset.weight.detach().cpu().numpy()).astype(np.float32, copy=False)
        return PrismRegion(
            supports=sup_np,
            lower=(box_c - box_o).astype(np.float32, copy=False),
            upper=(box_c + box_o).astype(np.float32, copy=False),
        )

    if hasattr(module, "n_components"):
        if not (hasattr(module, "class_center") and hasattr(module, "class_offset")):
            raise TypeError(
                f"Module {type(module).__name__!r} has n_components but lacks "
                f"class_center/class_offset."
            )
        k = int(module.n_components)
        d = int(module.embed_dim)
        c_raw = module.class_center.weight.detach().cpu().numpy()
        o_raw = np.abs(module.class_offset.weight.detach().cpu().numpy())
        n = int(c_raw.shape[0])
        return (
            c_raw.reshape(n, k, d).astype(np.float32, copy=False),
            o_raw.reshape(n, k, d).astype(np.float32, copy=False),
        )

    if hasattr(module, "class_center") and hasattr(module, "class_offset"):
        c_raw = module.class_center.weight.detach().cpu().numpy()
        o_raw = np.abs(module.class_offset.weight.detach().cpu().numpy())
        n, d = c_raw.shape
        return (
            c_raw.reshape(n, 1, d).astype(np.float32, copy=False),
            o_raw.reshape(n, 1, d).astype(np.float32, copy=False),
        )

    if hasattr(module, "class_embed") and hasattr(module, "class_offset"):
        c_raw = module.class_embed.weight.detach().cpu().numpy()
        o_raw = np.abs(module.class_offset.weight.detach().cpu().numpy())
        n, d = c_raw.shape
        return (
            c_raw.reshape(n, 1, d).astype(np.float32, copy=False),
            o_raw.reshape(n, 1, d).astype(np.float32, copy=False),
        )

    raise TypeError(
        f"Unsupported model/module {type(module).__name__!r} for class_regions."
    )


class _CPUFilteredRowLookup:
    """Thin wrapper around a CPU ``filtering_labels`` tensor that moves sliced rows to ``target_device``."""

    def __init__(self, cpu_labels: torch.Tensor, target_device: str) -> None:
        self._cpu_labels = cpu_labels
        self._target_device = target_device

    def __getitem__(self, item: Any) -> torch.Tensor:
        return self._cpu_labels[item].to(self._target_device, non_blocking=True)


def _compute_fast_head_centric_ranking_metrics(
    evaluator: Any,
    module: torch.nn.Module,
    test_data: torch.Tensor,
    eval_chunk_size: int = 65536,
) -> Dict[str, float]:
    """Compute head-centric ranking metrics while batching host-device transfers and chunking large ``gci0`` score expansions.

    Deliberate deviation from upstream ``BaseRankingEvaluator.compute_ranking_metrics``
    (Task 8 Fix 1b): before applying ``filtering_labels`` to a test tuple ``(head, tail)``,
    the filter multiplier of ``(head, tail)`` itself is reset to ``1.0`` on the cloned
    batch filter row so that standard filtered ranking removes only *other* known
    positives and never masks the target test tuple being ranked. Unfiltered metrics
    (``mr``, ``mrr``, ``auc``, ``hits@k``) are computed on the unmasked scores as usual.
    """
    from mowl.evaluation.base import compute_rank_roc
    from mowl.utils.data import FastTensorDataLoader

    module.to(evaluator.device)
    module.eval()

    num_tails = len(evaluator.tails)
    head_to_idx: Optional[Dict[int, int]] = getattr(evaluator, "_head_to_idx", None)
    tail_to_idx: Optional[Dict[int, int]] = getattr(evaluator, "_tail_to_idx", None)
    if head_to_idx is None or tail_to_idx is None:
        head_to_idx = {
            int(h): idx for idx, h in enumerate(evaluator.heads.detach().cpu().tolist())
        }
        tail_to_idx = {
            int(t): idx for idx, t in enumerate(evaluator.tails.detach().cpu().tolist())
        }
        setattr(evaluator, "_head_to_idx", head_to_idx)
        setattr(evaluator, "_tail_to_idx", tail_to_idx)

    filt = evaluator.filtering_labels
    cpu_labels: Optional[torch.Tensor] = (
        filt._cpu_labels if isinstance(filt, _CPUFilteredRowLookup) else None
    )

    dataloader = FastTensorDataLoader(
        test_data, batch_size=evaluator.batch_size, shuffle=False
    )

    mrr, fmrr = 0.0, 0.0
    mr, fmr = 0.0, 0.0
    ranks: Dict[int, int] = {}
    franks: Dict[int, int] = {}
    hits_k: Dict[str, float] = {"1": 0, "3": 0, "10": 0, "50": 0, "100": 0}
    f_hits_k: Dict[str, float] = {"1": 0, "3": 0, "10": 0, "50": 0, "100": 0}

    orig_forward = module.forward

    def _chunked_eval_forward(gci_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(gci_tensor, torch.Tensor)
            and gci_tensor.ndim == 2
            and gci_tensor.shape[0] > eval_chunk_size
        ):
            parts = [
                orig_forward(gci_tensor[start : start + eval_chunk_size], *args, **kwargs)
                for start in range(0, gci_tensor.shape[0], eval_chunk_size)
            ]
            return torch.cat(parts, dim=0)
        return orig_forward(gci_tensor, *args, **kwargs)

    module.forward = _chunked_eval_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            for (batch,) in dataloader:
                if batch.shape[1] == 2:
                    heads, tails = batch[:, 0], batch[:, 1]
                elif batch.shape[1] == 3:
                    heads, tails = batch[:, 0], batch[:, 2]
                else:
                    raise ValueError("Batch shape must be either (n, 2) or (n, 3)")

                head_ids = heads.detach().cpu().tolist()
                tail_ids = tails.detach().cpu().tolist()
                head_indices = [head_to_idx[int(h)] for h in head_ids]
                tail_indices = [tail_to_idx[int(t)] for t in tail_ids]

                batch_dev = batch.to(evaluator.device)
                head_scores, _ = evaluator.get_expanded_scores(
                    module, batch_dev, "head_centric"
                )
                if head_scores is None:
                    continue

                if cpu_labels is not None:
                    batch_filter = cpu_labels[head_indices].to(
                        head_scores.device, non_blocking=True
                    )
                else:
                    batch_filter = filt[head_indices].to(head_scores.device)
                batch_filter = batch_filter.clone()

                tail_idx_tensor = torch.tensor(
                    tail_indices, device=head_scores.device, dtype=torch.long
                )
                row_idx_tensor = torch.arange(
                    len(tail_indices), device=head_scores.device, dtype=torch.long
                )
                # Deliberate deviation from upstream BaseRankingEvaluator (Task 8 Fix 1b):
                # reset the filter label of the test tuple (head_i, tail_i) itself to 1.0
                # before ranking so standard filtered ranking removes OTHER known positives
                # and never multiplies the target test tuple's score by 10000.
                batch_filter[row_idx_tensor, tail_idx_tensor] = 1.0
                batch_f_scores = head_scores * batch_filter

                rank_tensors = []
                frank_tensors = []
                for i in range(len(head_indices)):
                    t_idx = tail_idx_tensor[i]
                    order = torch.argsort(head_scores[i], descending=False)
                    rank_tensors.append(torch.where(order == t_idx)[0][0])
                    f_order = torch.argsort(batch_f_scores[i], descending=False)
                    frank_tensors.append(torch.where(f_order == t_idx)[0][0])

                batch_ranks = (torch.stack(rank_tensors) + 1).cpu().tolist()
                batch_franks = (torch.stack(frank_tensors) + 1).cpu().tolist()

                for rank, f_rank in zip(batch_ranks, batch_franks):
                    rank = int(rank)
                    f_rank = int(f_rank)
                    mr += rank
                    mrr += 1.0 / rank
                    fmr += f_rank
                    fmrr += 1.0 / f_rank

                    for k in hits_k:
                        if rank <= int(k):
                            hits_k[k] += 1
                    for k in f_hits_k:
                        if f_rank <= int(k):
                            f_hits_k[k] += 1

                    ranks[rank] = ranks.get(rank, 0) + 1
                    franks[f_rank] = franks.get(f_rank, 0) + 1
    finally:
        module.forward = orig_forward  # type: ignore[method-assign]

    n_test = float(len(test_data))
    metrics: Dict[str, float] = {
        "mr": mr / n_test,
        "mrr": mrr / n_test,
        "f_mr": fmr / n_test,
        "f_mrr": fmrr / n_test,
        "auc": float(compute_rank_roc(ranks, num_tails)),
        "f_auc": float(compute_rank_roc(franks, num_tails)),
    }
    for k in hits_k:
        metrics[f"hits@{k}"] = hits_k[k] / n_test
    for k in f_hits_k:
        metrics[f"f_hits@{k}"] = f_hits_k[k] / n_test
    return metrics


def ranking_metrics(
    model: Any,
    bundle: DatasetBundle,
    device: Optional[str] = None,
    batch_size: int = 16,
    evaluator: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute mOWL subsumption ranking metrics on ``bundle.dataset.testing``, prefixed with ``rank_``.

    Parameters
    ----------
    model : EmbeddingELModel
        Trained mOWL model exposing ``model.module``.
    bundle : DatasetBundle
        Experiment dataset bundle. Returns ``{}`` when ``bundle.task != "subsumption"``
        or ``bundle.dataset.testing is None``.
    device : str, optional
        Evaluation device string. Defaults to ``module``'s own parameter device,
        and restores ``module``'s original device and ``train``/``eval`` mode
        after evaluation.
    batch_size : int, optional
        Test-tuple batch size for :class:`~mowl.evaluation.SubsumptionEvaluator`
        (default: ``16``).
    evaluator : SubsumptionEvaluator, optional
        Pre-constructed evaluator for ``bundle.dataset`` (avoids re-projecting
        large filter ontologies across runs when cached on ``bundle``).

    Returns
    -------
    Dict[str, float]
        Dictionary mapping ``f"rank_{k}"`` to ``float(v)`` for every metric ``k``
        returned by :meth:`SubsumptionEvaluator.evaluate`.
    """
    if bundle.task != "subsumption" or bundle.dataset.testing is None:
        return {}

    from mowl.evaluation import SubsumptionEvaluator

    module = getattr(model, "module", model)
    orig_device = _module_device(module)
    was_training = module.training
    resolved_device = str(orig_device) if device is None else str(device)

    filter_ontologies = [
        ont
        for ont in (
            bundle.dataset.ontology,
            bundle.dataset.validation,
            bundle.deductive_closure_ontology,
        )
        if ont is not None
    ]

    cached_eval = evaluator
    if cached_eval is None:
        cached_eval = getattr(bundle, "_subsumption_evaluator", None)
        if cached_eval is not None and str(getattr(cached_eval, "device", None)) != resolved_device:
            cached_eval = None

    if cached_eval is None:
        # Build the evaluator on CPU first so the dense (n_heads, n_tails) filtering_labels
        # matrix (7.47 GB on GO with 44,246 classes) is allocated in host RAM rather than
        # exhausting the 8 GB GPU VRAM.
        cached_eval = SubsumptionEvaluator(
            bundle.dataset, batch_size=batch_size, device="cpu"
        )
        testing_data = cached_eval.create_tuples(bundle.dataset.testing)
        if filter_ontologies:
            import torch as th

            filter_data = th.cat(
                [cached_eval.create_tuples(ont) for ont in filter_ontologies], dim=0
            )
            cached_eval.update_filtering_labels(filter_data)

        if resolved_device != "cpu":
            cached_eval.device = resolved_device
            cached_eval.heads = cached_eval.heads.to(resolved_device)
            cached_eval.tails = cached_eval.tails.to(resolved_device)
            cached_eval.mapped_heads = cached_eval.mapped_heads.to(resolved_device)
            cached_eval.mapped_tails = cached_eval.mapped_tails.to(resolved_device)
            cpu_labels = cached_eval.filtering_labels.cpu()
            cached_eval.filtering_labels = _CPUFilteredRowLookup(  # type: ignore[assignment]
                cpu_labels, resolved_device
            )

        setattr(cached_eval, "_cached_testing_data", testing_data)
        setattr(bundle, "_subsumption_evaluator", cached_eval)

    cached_eval.batch_size = int(batch_size)
    eval_oom_retries = 0
    try:
        while True:
            try:
                testing_data = getattr(cached_eval, "_cached_testing_data", None)
                if testing_data is None:
                    testing_data = cached_eval.create_tuples(bundle.dataset.testing)
                    if filter_ontologies:
                        import torch as th

                        filter_data = th.cat(
                            [cached_eval.create_tuples(ont) for ont in filter_ontologies],
                            dim=0,
                        )
                        cached_eval.update_filtering_labels(filter_data)
                    setattr(cached_eval, "_cached_testing_data", testing_data)
                raw_metrics = _compute_fast_head_centric_ranking_metrics(
                    cached_eval,
                    module,
                    testing_data,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if eval_oom_retries >= 4 or cached_eval.batch_size <= 1:
                    raise
                eval_oom_retries += 1
                cached_eval.batch_size = max(1, cached_eval.batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        module.to(orig_device)
        if was_training:
            module.train()
        else:
            module.eval()

    return {f"rank_{k}": float(v) for k, v in raw_metrics.items()}


def _build_nf2_index(cm: "CanonicalModel") -> Dict[str, List[NF2]]:
    """Build or retrieve a cached ``sub -> List[NF2]`` lookup table on ``cm``."""
    cached = getattr(cm, "_fast_nf2_by_sub", None)
    if cached is not None:
        return cached
    nf2_by_sub: Dict[str, List[NF2]] = defaultdict(list)
    for ax in cm.axioms:
        if isinstance(ax, NF2):
            nf2_by_sub[ax.sub1].append(ax)
            if ax.sub2 != ax.sub1:
                nf2_by_sub[ax.sub2].append(ax)
    setattr(cm, "_fast_nf2_by_sub", nf2_by_sub)
    return nf2_by_sub


# NOTE: _fast_joint_satisfiable intentionally remains in experiments/metrics.py
# rather than moving upstream because it directly inspects and populates
# CanonicalModel's private _conjunction_cache and _joint_sat_cache dicts.
def _fast_joint_satisfiable(
    cm: "CanonicalModel",
    nf2_by_sub: Dict[str, List[NF2]],
    names: Sequence[str],
) -> bool:
    """Check whether ``names`` are jointly satisfiable in ``cm``, populating ``cm._joint_sat_cache``.

    Because a fresh conjunction concept ``Z_joint <= A_i`` has no incoming role
    edges in the ELK completion calculus, its saturated subsumer set ``S(Z_joint)``
    is identical to the closure of ``{TOP} | U_{A in names} cm.completion.sub_super[A]``
    under ``NF2`` rules ``D1 and D2 <= E ==> S |= cm.completion.sub_super[E]``.
    """
    names_fset: FrozenSet[str] = frozenset(names)
    if names_fset in cm._joint_sat_cache:
        return cm._joint_sat_cache[names_fset]

    if BOTTOM in names_fset:
        cm._conjunction_cache[names_fset] = frozenset({BOTTOM, TOP})
        cm._joint_sat_cache[names_fset] = False
        return False

    # Fast positive check: non-empty intersection of canonical extensions
    it = iter(names_fset)
    first = next(it, None)
    if first is None:
        cm._joint_sat_cache[names_fset] = True
        return True
    common = cm.extension(first)
    for other_name in it:
        common = common & cm.extension(other_name)
        if not common:
            break
    if common:
        cm._joint_sat_cache[names_fset] = True
        return True

    sub_super = cm.completion.sub_super
    s: set[str] = {TOP}
    queue: List[str] = []
    for a in names_fset:
        for sup in sub_super.get(a, ()):
            if sup not in s:
                s.add(sup)
                queue.append(sup)

    idx = 0
    while idx < len(queue):
        d = queue[idx]
        idx += 1
        for nf2 in nf2_by_sub.get(d, ()):
            other = nf2.sub2 if nf2.sub1 == d else nf2.sub1
            if other in s and nf2.sup not in s:
                for sup in sub_super.get(nf2.sup, ()):
                    if sup not in s:
                        s.add(sup)
                        queue.append(sup)

    sups_fset = frozenset(s)
    is_sat = BOTTOM not in sups_fset
    cm._conjunction_cache[names_fset] = sups_fset
    cm._joint_sat_cache[names_fset] = is_sat
    return is_sat


def build_faithfulness_sets(
    bundle: DatasetBundle,
    cm: Optional["CanonicalModel"] = None,
    class_index_dict: Optional[Mapping[str, int]] = None,
    seed: int = 0,
    n_sample: int = 5000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Construct pairwise-disjoint ``(T+, T?, T-)`` named-class pair lists for ``bundle``.

    Parameters
    ----------
    bundle : DatasetBundle
        Experiment dataset bundle.
    cm : CanonicalModel, optional
        Canonical model of the training ontology. If ``None``, obtained via
        :func:`~experiments.cm_cache.get_canonical_model`.
    class_index_dict : Mapping[str, int], optional
        Model class index dictionary to filter candidate named classes.
    seed : int, optional
        Seed for ``numpy.random.default_rng(seed)`` (default: ``0``).
    n_sample : int, optional
        Maximum sample size for ``T-`` and ``T?`` (default: ``5000``).

    Returns
    -------
    t_pos : List[Tuple[str, str]]
        Held-out entailed test subsumption pairs ``(A, B)``.
    t_und : List[Tuple[str, str]]
        Sampled undetermined pairs ``(A, B)`` (neither entailed nor contradicted).
    t_neg : List[Tuple[str, str]]
        Sampled contradicted pairs ``(A, B)`` (``not cm.joint_satisfiable({A, B})``).
    """
    if cm is None:
        cm = get_canonical_model(bundle)

    cached_sets = getattr(bundle, "_faithfulness_sets_cache", None)
    cache_key = (int(seed), int(n_sample))
    if (
        cached_sets is not None
        and cached_sets[0] == cache_key
        and (
            class_index_dict is None
            or all(
                a in class_index_dict and b in class_index_dict
                for s in cached_sets[1]
                for a, b in s
            )
        )
    ):
        return cached_sets[1]

    candidates = [
        c
        for c in cm.names
        if _is_named_class(c)
        and (class_index_dict is None or c in class_index_dict)
    ]

    t_pos: List[Tuple[str, str]] = []
    seen_pos: set[Tuple[str, str]] = set()
    if bundle.dataset.testing is not None:
        from mowl.projection import TaxonomyProjector

        proj = TaxonomyProjector()
        raw_edges = proj.project(bundle.dataset.testing)
        for e in sorted(raw_edges, key=lambda edge: (str(edge.src), str(edge.dst))):
            sub, sup = str(e.src), str(e.dst)
            if (
                sub != sup
                and _is_named_class(sub)
                and _is_named_class(sup)
                and (class_index_dict is None or (sub in class_index_dict and sup in class_index_dict))
                and (sub, sup) not in seen_pos
            ):
                seen_pos.add((sub, sup))
                t_pos.append((sub, sup))

    t_und: List[Tuple[str, str]] = []
    t_neg: List[Tuple[str, str]] = []
    n_cand = len(candidates)
    if n_cand < 2 or n_sample <= 0:
        return t_pos, t_und, t_neg

    nf2_by_sub = _build_nf2_index(cm)
    rng = np.random.default_rng(seed)
    max_draws = 50 * n_sample
    seen_pairs: set[Tuple[str, str]] = set(seen_pos)

    for _ in range(max_draws):
        if len(t_neg) >= n_sample and len(t_und) >= n_sample:
            break
        i = int(rng.integers(0, n_cand))
        j = int(rng.integers(0, n_cand))
        if i == j:
            continue
        pair = (candidates[i], candidates[j])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if not _fast_joint_satisfiable(cm, nf2_by_sub, pair):
            if len(t_neg) < n_sample:
                t_neg.append(pair)
        elif not cm.entails(pair[0], pair[1]):
            if len(t_und) < n_sample:
                t_und.append(pair)

    setattr(bundle, "_faithfulness_sets_cache", (cache_key, (t_pos, t_und, t_neg)))
    return t_pos, t_und, t_neg


def _score_gci0_pairs(
    module: torch.nn.Module,
    class_index_dict: Mapping[str, int],
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = 32768,
) -> np.ndarray:
    """Score ``(sub, sup)`` pairs with ``module(batch, 'gci0')`` (lower = more plausible)."""
    if not pairs:
        return np.empty((0,), dtype=np.float64)

    device = _module_device(module)
    idx_array = np.array(
        [[class_index_dict[a], class_index_dict[b]] for a, b in pairs],
        dtype=np.int64,
    )
    n = len(idx_array)
    chunks: List[np.ndarray] = []
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch_t = torch.from_numpy(idx_array[start : start + batch_size]).to(
                    device=device, dtype=torch.long
                )
                scores = module(batch_t, "gci0").reshape(-1).detach().cpu().numpy()
                chunks.append(scores.astype(np.float64, copy=False))
    finally:
        if was_training:
            module.train()
    return np.concatenate(chunks, axis=0)


def faithfulness_metrics(
    model: Any,
    bundle: DatasetBundle,
    cm: Optional["CanonicalModel"] = None,
    seed: int = 0,
    n_sample: int = 5000,
) -> Dict[str, Any]:
    """Compute faithfulness ordering metrics over ``T+``, ``T?``, and ``T-`` (Spec §5.3).

    Parameters
    ----------
    model : EmbeddingELModel
        Trained mOWL model exposing ``.module`` and ``.class_index_dict``.
    bundle : DatasetBundle
        Experiment dataset bundle.
    cm : CanonicalModel, optional
        Canonical model of the training ontology. If ``None``, obtained via
        :func:`~experiments.cm_cache.get_canonical_model`.
    seed : int, optional
        Random seed for pair sampling and ``10^5`` pair/triple ordering draws
        (default: ``0``).
    n_sample : int, optional
        Maximum sample size for ``T-`` and ``T?`` (default: ``5000``).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``auc_pos_neg``, ``pair_fvr``, ``fvr``, ``n_pos``,
        ``n_und``, ``n_neg``. When ``n_neg < 50`` (or ``n_pos == 0`` or
        ``n_und == 0``), the three ordering metrics are ``None``.
    """
    if cm is None:
        cm = get_canonical_model(bundle)

    c2i = model.class_index_dict
    t_pos, t_und, t_neg = build_faithfulness_sets(
        bundle, cm, c2i, seed=seed, n_sample=n_sample
    )
    n_pos = len(t_pos)
    n_und = len(t_und)
    n_neg = len(t_neg)

    if n_neg < 50 or n_pos == 0 or n_und == 0:
        return {
            "auc_pos_neg": None,
            "pair_fvr": None,
            "fvr": None,
            "n_pos": int(n_pos),
            "n_und": int(n_und),
            "n_neg": int(n_neg),
        }

    s_pos = _score_gci0_pairs(model.module, c2i, t_pos)
    s_und = _score_gci0_pairs(model.module, c2i, t_und)
    s_neg = _score_gci0_pairs(model.module, c2i, t_neg)

    # Exact Mann-Whitney ROC AUC for P[f(+) < f(-)] + 0.5 * P[f(+) == f(-)]
    combined = np.concatenate([-s_pos, -s_neg])
    ranks = rankdata(combined, method="average")
    u_stat = float(np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2.0)
    auc_pos_neg = float(u_stat / float(n_pos * n_neg))

    rng = np.random.default_rng(seed)
    n_draws = 100_000
    und_idx = rng.integers(0, n_und, size=n_draws)
    neg_idx = rng.integers(0, n_neg, size=n_draws)
    pair_fvr = float(np.mean(s_und[und_idx] >= s_neg[neg_idx]))

    t_pos_idx = rng.integers(0, n_pos, size=n_draws)
    t_und_idx = rng.integers(0, n_und, size=n_draws)
    t_neg_idx = rng.integers(0, n_neg, size=n_draws)
    strict_chain = (s_neg[t_neg_idx] > s_und[t_und_idx]) & (
        s_und[t_und_idx] > s_pos[t_pos_idx]
    )
    fvr = float(np.mean(~strict_chain))

    return {
        "auc_pos_neg": auc_pos_neg,
        "pair_fvr": pair_fvr,
        "fvr": fvr,
        "n_pos": int(n_pos),
        "n_und": int(n_und),
        "n_neg": int(n_neg),
    }


def modelness(
    model: Any,
    *,
    tol: float = 1e-6,
    batch_size: int = 32768,
    margin: float = 0.0,
) -> Dict[str, float]:
    """Compute per-GCI and overall fraction of training axioms with loss ``<= tol`` at ``margin``.

    Temporarily sets the module's margin attribute (``margin`` for ELBE, ``gamma``
    for Box²EL and MultiBoxEL) to ``margin`` (default ``0.0``) during the
    measurement and restores the exact original value afterwards in a ``finally``
    block.

    Parameters
    ----------
    model : EmbeddingELModel
        Trained mOWL model exposing ``.module`` and ``.training_datasets``.
    tol : float, optional
        Loss threshold for an axiom to be considered satisfied (default: ``1e-6``).
    batch_size : int, optional
        Chunk size for scoring large GCI datasets (default: ``32768``).
    margin : float, optional
        Margin value set on the module during evaluation (default: ``0.0``).

    Returns
    -------
    Dict[str, float]
        Dictionary containing ``modelness_<gci>`` for every non-empty GCI dataset
        present in ``model.training_datasets`` and ``modelness_all`` over all
        training axioms.
    """
    module = model.module
    device = _module_device(module)
    eval_margin = float(margin)

    saved_module_attrs: Dict[str, Any] = {}
    for attr in ("gamma", "margin"):
        if hasattr(module, attr):
            saved_module_attrs[attr] = getattr(module, attr)
    saved_model_margin = getattr(model, "margin", None)
    has_model_margin = hasattr(model, "margin")
    was_training = module.training

    out: Dict[str, float] = {}
    total_satisfied = 0
    total_axioms = 0

    try:
        for attr in saved_module_attrs:
            setattr(module, attr, eval_margin)
        if has_model_margin:
            setattr(model, "margin", eval_margin)
        module.eval()

        with torch.no_grad():
            for gci_name, gci_ds in model.training_datasets.items():
                if gci_ds is None or len(gci_ds) == 0:
                    continue
                data = gci_ds[:]
                n_rows = int(data.shape[0])
                sat_gci = 0
                supported = True
                for start in range(0, n_rows, batch_size):
                    chunk = data[start : start + batch_size].to(device=device)
                    try:
                        raw_losses = module(chunk, gci_name)
                    except NotImplementedError:
                        supported = False
                        break
                    if isinstance(raw_losses, Exception) or raw_losses is NotImplemented:
                        supported = False
                        break
                    losses = raw_losses.reshape(-1)
                    sat_gci += int((losses <= tol).sum().item())
                if not supported:
                    continue
                out[f"modelness_{gci_name}"] = float(sat_gci / n_rows)
                total_satisfied += sat_gci
                total_axioms += n_rows
    finally:
        for attr, val in saved_module_attrs.items():
            setattr(module, attr, val)
        if has_model_margin:
            setattr(model, "margin", saved_model_margin)
        if was_training:
            module.train()

    out["modelness_all"] = (
        float(total_satisfied / total_axioms) if total_axioms > 0 else 1.0
    )
    return out


def point_modelness(model: Any) -> Optional[Dict[str, float]]:
    """Return ``{"pos_frac": float, "hardneg_frac": float}`` when objective is ``point``/``mixed``, else ``None``."""
    module = getattr(model, "module", model)
    objective = getattr(module, "objective", getattr(model, "objective", "axiom"))
    if objective not in ("point", "mixed"):
        return None
    ps = getattr(module, "point_supervision", getattr(model, "point_supervision", None))
    if ps is None or not hasattr(module, "point_modelness"):
        return None
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            raw = module.point_modelness(ps)
    finally:
        if was_training:
            module.train()
    return {
        "pos_frac": float(raw["pos_frac"]),
        "hardneg_frac": float(raw["hardneg_frac"]),
    }


def helly_metrics(
    model: Any,
    bundle: Optional[DatasetBundle] = None,
    cm: Optional["CanonicalModel"] = None,
    certificates: Optional[Sequence[HellyCertificate]] = None,
    *,
    tol: float = 1e-6,
    deg_eps: float = 1e-3,
) -> Dict[str, Any]:
    """Compute Helly violation and degeneracy rates across ``certificates`` (Spec §5.3).

    For each :class:`~mowl.geometric.HellyCertificate` ``cert``:
    - ``triple_nonempty``: ``True`` iff there exists a selection of one component
      box per concept name in ``cert.c1 ∪ cert.c2 ∪ cert.c3`` whose common
      intersection is non-empty under corner arithmetic with tolerance ``tol``
      (``max lo <= min hi + tol`` in every coordinate).
    - ``degenerate``: ``True`` iff at least one non-``None`` concept name in
      ``cert.pairwise_names`` has ``min_d offset <= deg_eps`` in every component.

    Parameters
    ----------
    model : EmbeddingELModel
        mOWL model exposing ``.module`` and ``.class_index_dict``.
    bundle : DatasetBundle, optional
        Dataset bundle used to obtain ``cm`` via :func:`~experiments.cm_cache.get_canonical_model`
        when ``cm`` and ``certificates`` are not provided.
    cm : CanonicalModel, optional
        Canonical model of the training ontology.
    certificates : Sequence[HellyCertificate], optional
        Certificates to evaluate. If ``None``, computed via
        ``certificate_triangles(cm)``.
    tol : float, optional
        Corner intersection tolerance (default: ``1e-6``).
    deg_eps : float, optional
        Minimum coordinate half-width threshold for non-degeneracy (default: ``1e-3``).

    Returns
    -------
    Dict[str, Any]
        ``{"helly_violation_rate": Optional[float], "degeneracy_rate": Optional[float], "n_certificates": int, "n_certificates_unmapped": int}``.
        When ``n_certificates == 0``, both ``helly_violation_rate`` and ``degeneracy_rate``
        are ``None``. Certificates with any concept name missing from ``model.class_index_dict``
        are counted in ``n_certificates_unmapped`` and excluded from the violation
        and degeneracy rates; raises ``ValueError`` if all non-empty certificates
        are unmapped.
    """
    if certificates is None:
        if cm is None:
            if bundle is None:
                raise ValueError(
                    "helly_metrics requires at least one of certificates, cm, or bundle."
                )
            cm = get_canonical_model(bundle)
        certificates = certificate_triangles(cm)

    n_certs = len(certificates)
    if n_certs == 0:
        return {
            "helly_violation_rate": None,
            "degeneracy_rate": None,
            "n_certificates": 0,
            "n_certificates_unmapped": 0,
        }

    regions = class_regions(model)
    c2i = model.class_index_dict

    violations = 0
    degeneracies = 0
    n_unmapped = 0

    if isinstance(regions, PrismRegion):
        for cert in certificates:
            all_names = list(
                dict.fromkeys(list(cert.c1) + list(cert.c2) + list(cert.c3))
            )
            pw_names = [p for p in cert.pairwise_names if p is not None]
            if (
                len(all_names) == 0
                or any(name not in c2i for name in all_names)
                or any(p_name not in c2i for p_name in pw_names)
            ):
                n_unmapped += 1
                continue

            inter = regions[c2i[all_names[0]]]
            for name in all_names[1:]:
                inter = inter.intersect(regions[c2i[name]])
            if not inter.is_empty(tol=tol):
                violations += 1

            is_deg = False
            for p_name in pw_names:
                if regions[c2i[p_name]].is_degenerate(deg_eps=deg_eps):
                    is_deg = True
                    break
            if is_deg:
                degeneracies += 1
    else:
        centers, offsets = regions
        lo_all = centers - offsets  # (n, k, d)
        hi_all = centers + offsets  # (n, k, d)
        k = int(centers.shape[1])

        for cert in certificates:
            all_names = list(
                dict.fromkeys(list(cert.c1) + list(cert.c2) + list(cert.c3))
            )
            pw_names = [p for p in cert.pairwise_names if p is not None]
            if (
                len(all_names) == 0
                or any(name not in c2i for name in all_names)
                or any(p_name not in c2i for p_name in pw_names)
            ):
                n_unmapped += 1
                continue

            first_idx = c2i[all_names[0]]
            active: List[Tuple[np.ndarray, np.ndarray]] = [
                (lo_all[first_idx, m], hi_all[first_idx, m]) for m in range(k)
            ]
            for name in all_names[1:]:
                idx_u = c2i[name]
                next_active: List[Tuple[np.ndarray, np.ndarray]] = []
                for cur_lo, cur_hi in active:
                    for m in range(k):
                        new_lo = np.maximum(cur_lo, lo_all[idx_u, m])
                        new_hi = np.minimum(cur_hi, hi_all[idx_u, m])
                        if bool(np.all(new_lo <= new_hi + tol)):
                            next_active.append((new_lo, new_hi))
                active = next_active
                if not active:
                    break
            if len(active) > 0:
                violations += 1

            is_deg = False
            for p_name in pw_names:
                off_p = offsets[c2i[p_name]]  # (k, d)
                valid_comps = np.all(off_p > deg_eps, axis=-1)  # (k,)
                if not bool(np.any(valid_comps)):
                    is_deg = True
                    break
            if is_deg:
                degeneracies += 1

    n_mapped = n_certs - n_unmapped
    if n_mapped == 0:
        raise ValueError(
            f"All {n_certs} Helly certificates have concept names missing from "
            f"model.class_index_dict; the CanonicalModel cache or model is inconsistent."
        )

    return {
        "helly_violation_rate": float(violations / n_mapped),
        "degeneracy_rate": float(degeneracies / n_mapped),
        "n_certificates": int(n_certs),
        "n_certificates_unmapped": int(n_unmapped),
    }


def cost_metrics(
    model: Any,
    timings: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute parameter count, runtime, component usage, and GPU memory metrics (Spec §5.3).

    Parameters
    ----------
    model : EmbeddingELModel or torch.nn.Module
        mOWL model (or its underlying ``.module``).
    timings : Mapping[str, Any], optional
        Timing and memory dictionary recorded by the runner. Recognises
        ``epoch_times``, ``mean_epoch_time_s`` / ``mean_epoch_time``,
        ``total_train_time_s`` / ``total_train_time`` / ``train_time_s`` /
        ``train_time``, ``epochs``, and ``peak_gpu_memory_bytes`` /
        ``peak_gpu_memory``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``param_count``, ``mean_epoch_time_s``,
        ``total_train_time_s``, ``n_components``, ``mean_effective_components``,
        ``peak_gpu_memory_bytes``, and ``peak_gpu_memory_mb``.
    """
    module = getattr(model, "module", model)
    timings_dict = dict(timings) if timings is not None else {}

    param_count = int(sum(p.numel() for p in module.parameters()))
    n_components = int(getattr(module, "n_components", 1))

    if hasattr(module, "effective_components"):
        eff_tensor = module.effective_components()
        mean_eff = float(eff_tensor.to(dtype=torch.float32).mean().item())
    else:
        try:
            regions = class_regions(model)
            if isinstance(regions, PrismRegion):
                non_deg = (np.sum(regions.supports, axis=-1) > 1e-3) & np.all(
                    (regions.upper - regions.lower) > 1e-3, axis=-1
                )
                mean_eff = float(np.mean(non_deg))
            else:
                _, offsets = regions
                mean_eff = float(np.mean(np.all(offsets[:, 0, :] > 1e-3, axis=-1)))
        except TypeError:
            mean_eff = float(n_components)

    epoch_times = timings_dict.get("epoch_times")
    total_train_time_s: Optional[float] = None
    for key in (
        "total_train_time_s",
        "total_train_time",
        "train_time_s",
        "train_time",
    ):
        if timings_dict.get(key) is not None:
            total_train_time_s = float(timings_dict[key])
            break
    if total_train_time_s is None:
        if epoch_times is not None and len(epoch_times) > 0:
            total_train_time_s = float(np.sum(epoch_times))
        else:
            total_train_time_s = 0.0

    mean_epoch_time_s: Optional[float] = None
    for key in ("mean_epoch_time_s", "mean_epoch_time"):
        if timings_dict.get(key) is not None:
            mean_epoch_time_s = float(timings_dict[key])
            break
    if mean_epoch_time_s is None:
        if epoch_times is not None and len(epoch_times) > 0:
            mean_epoch_time_s = float(np.mean(epoch_times))
        elif timings_dict.get("epochs") and float(timings_dict["epochs"]) > 0:
            mean_epoch_time_s = float(
                total_train_time_s / float(timings_dict["epochs"])
            )
        else:
            mean_epoch_time_s = 0.0

    peak_bytes: Optional[int] = None
    for key in ("peak_gpu_memory_bytes", "peak_gpu_memory"):
        if timings_dict.get(key) is not None:
            peak_bytes = int(timings_dict[key])
            break
    if peak_bytes is None:
        if torch.cuda.is_available():
            peak_bytes = int(torch.cuda.max_memory_allocated())
        else:
            peak_bytes = 0

    return {
        "param_count": param_count,
        "mean_epoch_time_s": mean_epoch_time_s,
        "total_train_time_s": total_train_time_s,
        "n_components": n_components,
        "mean_effective_components": mean_eff,
        "peak_gpu_memory_bytes": peak_bytes,
        "peak_gpu_memory_mb": float(peak_bytes / (1024.0 * 1024.0)),
    }


def abox_metrics(
    model: Any,
    bundle: DatasetBundle,
    *,
    split: str = "test",
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute the full ABox evaluation block for ``bundle`` (Spec §3.1, §3.2).

    Evaluates:
    - Filtered candidate ranking (both ``rankable_only`` and ``all_positives``
      plus coverage accounting),
    - CAFA-style protein-centric :math:`F_{\\max}` for ``yeast-function``
      (excluding KB-entailed cells from both prediction and truth sets; ``None``
      on ``helis``),
    - ``hierarchy_violation_rate`` over entailed candidate subsumptions,
    - Three-way faithfulness metrics on ``label`` (``faithfulness_kb``) and, when
      present, ``label_gold`` (``faithfulness_gold``), with ``auc_pos_und``
      (``AUC(+/?)``) pooled across evaluation individuals,
    - ``abox_modelness`` on training assertions (``None`` when no assertion is
      scored),
    - Missing entity counts (``n_missing_individuals``, ``n_missing_candidates``).
    """
    from experiments.abox_eval import (
        abox_modelness,
        count_missing_entities,
        fmax,
        get_candidate_hierarchy,
        hierarchy_violation_rate,
        load_abox_split_data,
        ranking,
        score_candidates,
        three_way_metrics,
    )

    split_data = load_abox_split_data(bundle, split=split)
    individuals = split_data["individuals"]
    candidates = split_data["candidates"]
    cand_to_idx = split_data["cand_to_idx"]

    missing_counts = count_missing_entities(model, individuals, candidates)
    scores = score_candidates(model, individuals, candidates)

    rank_res = ranking(
        scores,
        split_data["positives"],
        skip_mask=split_data["skip_mask"],
    )

    hierarchy = get_candidate_hierarchy(bundle.paths["ontology"], candidates)
    entailed_pairs: List[Tuple[int, int]] = []
    for c_iri in candidates:
        c_idx = cand_to_idx[c_iri]
        for sup_iri in hierarchy.get(c_iri, ()):
            sup_idx = cand_to_idx.get(sup_iri, -1)
            if sup_idx >= 0 and sup_idx != c_idx:
                entailed_pairs.append((c_idx, sup_idx))

    hvr = hierarchy_violation_rate(scores, entailed_pairs, seed=seed)

    if bundle.name == "yeast-function":
        fmax_val: Optional[float] = fmax(
            scores,
            split_data["true_sets"],
            hierarchy,
            candidates=candidates,
            entailed_mask=(split_data["label_codes"] == 1),
        )
    else:
        fmax_val = None

    faith_kb = three_way_metrics(scores, split_data["label_codes"], seed=seed)
    if split_data["label_gold_codes"] is not None:
        faith_gold: Optional[Dict[str, Any]] = three_way_metrics(
            scores, split_data["label_gold_codes"], seed=seed
        )
    else:
        faith_gold = None

    abox_mod = abox_modelness(model)

    return {
        "rank_mrr": rank_res["mrr"],
        "rank_hits@1": rank_res["hits@1"],
        "rank_hits@5": rank_res["hits@5"],
        "rank_hits@10": rank_res["hits@10"],
        "rank_auc": rank_res["auc"],
        "rank_all_mrr": rank_res["all_mrr"],
        "rank_all_hits@1": rank_res["all_hits@1"],
        "rank_all_hits@5": rank_res["all_hits@5"],
        "rank_all_hits@10": rank_res["all_hits@10"],
        "rank_all_auc": rank_res["all_auc"],
        "rank_coverage": rank_res["coverage"],
        "n_positives_total": rank_res["n_positives_total"],
        "n_positives_rankable": rank_res["n_positives_rankable"],
        "n_positives_unrankable": rank_res["n_positives_unrankable"],
        "ranking_rankable": rank_res["rankable_only"],
        "ranking_all": rank_res["all_positives"],
        "fmax": fmax_val,
        "hierarchy_violation_rate": hvr,
        "abox_modelness": abox_mod,
        "faithfulness_kb": faith_kb,
        "faithfulness_gold": faith_gold,
        "auc_pos_neg": faith_kb["auc_pos_neg"],
        "auc_pos_und": faith_kb["auc_pos_und"],
        "pair_fvr": faith_kb["pair_fvr"],
        "fvr": faith_kb["fvr"],
        "n_pos": faith_kb["n_pos"],
        "n_und": faith_kb["n_und"],
        "n_neg": faith_kb["n_neg"],
        "auc_pos_neg_gold": faith_gold["auc_pos_neg"] if faith_gold else None,
        "auc_pos_und_gold": faith_gold["auc_pos_und"] if faith_gold else None,
        "pair_fvr_gold": faith_gold["pair_fvr"] if faith_gold else None,
        "fvr_gold": faith_gold["fvr"] if faith_gold else None,
        "n_missing_individuals": missing_counts["n_missing_individuals"],
        "n_missing_candidates": missing_counts["n_missing_candidates"],
    }

