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
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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


def class_regions(model: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``(centers, offsets)`` NumPy arrays of shape ``(n, k, d)`` from ``model``.

    Parameters
    ----------
    model : mOWL model or torch.nn.Module
        An ``ELBE``, ``BoxSquaredEL``, or ``MultiBoxEL`` model (or its underlying
        ``.module``).

    Returns
    -------
    centers : np.ndarray
        Float32 array of shape ``(n_classes, k, d)``.
    offsets : np.ndarray
        Non-negative float32 array of shape ``(n_classes, k, d)`` (computed via
        ``abs``).

    Guarantees
    ----------
    - For ``ELBE`` and ``BoxSquaredEL``, ``k == 1``.
    - For ``MultiBoxEL``, ``k == module.n_components``.
    - ``offsets >= 0`` everywhere.

    Non-guarantees
    --------------
    - Raises ``TypeError`` if ``model`` exposes neither ``class_center`` nor
      ``class_embed`` with ``class_offset``.
    """
    module = getattr(model, "module", model)

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
        if (
            cached_eval is not None
            and (
                str(getattr(cached_eval, "device", None)) != resolved_device
                or getattr(cached_eval, "batch_size", None) != batch_size
            )
        ):
            cached_eval = None

    if cached_eval is None:
        cached_eval = SubsumptionEvaluator(
            bundle.dataset, batch_size=batch_size, device=resolved_device
        )
        testing_data = cached_eval.create_tuples(bundle.dataset.testing)
        filter_data = None
        if filter_ontologies:
            import torch as th

            filter_data = th.cat(
                [cached_eval.create_tuples(ont) for ont in filter_ontologies], dim=0
            )
            cached_eval.update_filtering_labels(filter_data)
        setattr(cached_eval, "_cached_testing_data", testing_data)
        setattr(bundle, "_subsumption_evaluator", cached_eval)

    try:
        if hasattr(cached_eval, "_cached_testing_data"):
            raw_metrics = cached_eval.compute_ranking_metrics(
                module,
                getattr(cached_eval, "_cached_testing_data"),
                filter_data=None,
                mode="head_centric",
            )
        else:
            raw_metrics = cached_eval.evaluate(
                module,
                bundle.dataset.testing,
                filter_ontologies=filter_ontologies,
            )
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
) -> Dict[str, float]:
    """Compute per-GCI and overall fraction of training axioms with loss ``<= tol`` at margin 0.

    Temporarily sets the module's margin attribute (``margin`` for ELBE, ``gamma``
    for Box²EL and MultiBoxEL) to ``0.0`` during the measurement and restores the
    exact original value afterwards in a ``finally`` block.

    Parameters
    ----------
    model : EmbeddingELModel
        Trained mOWL model exposing ``.module`` and ``.training_datasets``.
    tol : float, optional
        Loss threshold for an axiom to be considered satisfied (default: ``1e-6``).
    batch_size : int, optional
        Chunk size for scoring large GCI datasets (default: ``32768``).

    Returns
    -------
    Dict[str, float]
        Dictionary containing ``modelness_<gci>`` for every non-empty GCI dataset
        present in ``model.training_datasets`` and ``modelness_all`` over all
        training axioms.
    """
    module = model.module
    device = _module_device(module)

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
            setattr(module, attr, 0.0)
        if has_model_margin:
            setattr(model, "margin", 0.0)
        module.eval()

        with torch.no_grad():
            for gci_name, gci_ds in model.training_datasets.items():
                if gci_ds is None or len(gci_ds) == 0:
                    continue
                data = gci_ds[:]
                n_rows = int(data.shape[0])
                sat_gci = 0
                for start in range(0, n_rows, batch_size):
                    chunk = data[start : start + batch_size].to(device=device)
                    losses = module(chunk, gci_name).reshape(-1)
                    sat_gci += int((losses <= tol).sum().item())
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
        ``{"helly_violation_rate": float, "degeneracy_rate": float, "n_certificates": int, "n_certificates_unmapped": int}``.
        Certificates with any concept name missing from ``model.class_index_dict``
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
            "helly_violation_rate": 0.0,
            "degeneracy_rate": 0.0,
            "n_certificates": 0,
            "n_certificates_unmapped": 0,
        }

    centers, offsets = class_regions(model)
    lo_all = centers - offsets  # (n, k, d)
    hi_all = centers + offsets  # (n, k, d)
    k = int(centers.shape[1])
    c2i = model.class_index_dict

    violations = 0
    degeneracies = 0
    n_unmapped = 0

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
        _, offsets = class_regions(model)
        mean_eff = float(np.mean(np.all(offsets[:, 0, :] > 1e-3, axis=-1)))

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
