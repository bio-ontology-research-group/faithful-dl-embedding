"""E-OMQA / Query2Box evaluation protocol: loaders for the released pickles and the filtered ranking metrics.

Metrics follow ``code_q2b_onto/models.py::test_step`` (Andresel et al. 2023; Ren et al. 2020): for every test
query, every hard answer is ranked against all entities with the easy answers and the other hard answers filtered
out (rank = 1 + number of non-answer entities scoring strictly higher, ties broken as ``torch.argsort`` does, i.e.
by score only). Reported per query shape: MRR, HITS@1, HITS@3, HITS@10 (macro over queries, each query contributing
the mean over its hard answers), plus the unweighted mean over shapes.

Query structures are the Query2Box tuples, e.g. ``('e', ('r',))``, ``(('e', ('r',)), ('e', ('r',)))``,
``(('e', ('r',)), ('e', ('r',)), ('u',))``, ``((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))``.
"""
from __future__ import annotations

import collections
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch as th

from experiments.omqa.build_eomqa import norm, safe_load

QUERY_NAMES: Dict[Any, str] = {
    ("e", ("r",)): "1p",
    ("e", ("r", "r")): "2p",
    ("e", ("r", "r", "r")): "3p",
    (("e", ("r",)), ("e", ("r",))): "2i",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
    ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
    (("e", ("r", "r")), ("e", ("r",))): "pi",
    (("e", ("r",)), ("e", ("r",)), ("u",)): "2u",
    ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up",
}


def load_split(data_root: Path, test_case: str, part: str = "test") -> Tuple[Dict[Any, List[Any]], Dict[Any, Set[int]], Dict[Any, Set[int]]]:
    """Load ``<data_root>/<test_case>/<part>-{queries,easy-answers,hard-answers}.pkl``.

    Returns ``(queries_by_structure, easy, hard)`` with queries normalised to Python ints and the answer dicts keyed
    by the normalised query.
    """
    d = data_root / test_case
    queries = safe_load(d / f"{part}-queries.pkl")
    easy = safe_load(d / f"{part}-easy-answers.pkl")
    hard = safe_load(d / f"{part}-hard-answers.pkl")
    qbs: Dict[Any, List[Any]] = {}
    easy_n: Dict[Any, Set[int]] = {}
    hard_n: Dict[Any, Set[int]] = {}
    for structure, qs in queries.items():
        lst = []
        for q in qs:
            qn = norm(q)
            lst.append(qn)
            easy_n[qn] = {int(a) for a in (easy.get(q) if q in easy else easy.get(qn, set()))}
            hard_n[qn] = {int(a) for a in (hard.get(q) if q in hard else hard.get(qn, set()))}
        qbs[norm(structure) if not isinstance(structure, str) else structure] = lst
    return qbs, easy_n, hard_n


def filtered_ranks(scores: th.Tensor, easy: Sequence[int], hard: Sequence[int]) -> th.Tensor:
    """Filtered ranks (1-based) of the hard answers given scores over all entities (Query2Box convention).

    ``rank(a) = 1 + #{x not in easy ∪ hard : score[x] > score[a]} + #{x not in easy ∪ hard : score[x] == score[a],
    x < a}`` — the second term reproduces ``argsort`` tie-breaking by index as closely as a deterministic rule can.
    """
    scores = scores.detach()
    mask = th.ones_like(scores, dtype=th.bool)
    idx = th.tensor(list(easy) + list(hard), dtype=th.long, device=scores.device)
    if idx.numel():
        mask[idx] = False
    hard_t = th.tensor(list(hard), dtype=th.long, device=scores.device)
    s_hard = scores[hard_t]  # (H,)
    others = scores[mask]  # (N',)
    other_idx = th.nonzero(mask).squeeze(1)
    greater = (others.unsqueeze(0) > s_hard.unsqueeze(1)).sum(1)
    ties = ((others.unsqueeze(0) == s_hard.unsqueeze(1)) & (other_idx.unsqueeze(0) < hard_t.unsqueeze(1))).sum(1)
    return (1 + greater + ties).to(th.float)


class PreparedSplit(NamedTuple):
    """Answer sets of one query structure as flat CPU tensors (built once; sliced and padded per batch on device).

    ``queries[i]`` has hard answers ``hard_flat[hard_off[i]:hard_off[i+1]]`` and filter set (easy ∪ hard)
    ``filt_flat[filt_off[i]:filt_off[i+1]]``. Queries without hard answers are dropped (they never counted).
    """
    queries: List[Any]
    hard_flat: th.Tensor
    hard_off: th.Tensor
    filt_flat: th.Tensor
    filt_off: th.Tensor


def prepare(queries_by_structure: Dict[Any, List[Any]], easy: Dict[Any, Set[int]], hard: Dict[Any, Set[int]]) -> Dict[Any, PreparedSplit]:
    """Convert the Python answer sets of :func:`load_split` into :class:`PreparedSplit` per structure (once per run)."""
    out: Dict[Any, PreparedSplit] = {}
    for structure, qs in queries_by_structure.items():
        qs = [q for q in qs if hard.get(q)]
        h_lists = [sorted(hard[q]) for q in qs]
        f_lists = [sorted(easy.get(q, ())) + h for q, h in zip(qs, h_lists)]
        h_len = np.fromiter((len(h) for h in h_lists), dtype=np.int64, count=len(h_lists))
        f_len = np.fromiter((len(f) for f in f_lists), dtype=np.int64, count=len(f_lists))
        h_off = th.from_numpy(np.concatenate([[0], np.cumsum(h_len)]).astype(np.int64))
        f_off = th.from_numpy(np.concatenate([[0], np.cumsum(f_len)]).astype(np.int64))
        # np.fromiter over a chained iterator is ~20x faster than th.tensor(list) for tens of millions of ints
        h_flat = th.from_numpy(np.fromiter(itertools.chain.from_iterable(h_lists), dtype=np.int64, count=int(h_len.sum())))
        f_flat = th.from_numpy(np.fromiter(itertools.chain.from_iterable(f_lists), dtype=np.int64, count=int(f_len.sum())))
        out[structure] = PreparedSplit(qs, h_flat, h_off, f_flat, f_off)
    return out


def load_prepared(data_root: Path, test_case: str, part: str = "test", cache: bool = True) -> Dict[Any, PreparedSplit]:
    """:func:`prepare` of :func:`load_split`, cached as ``<data_root>/<test_case>/<part>.prepared.pt`` (tensors +
    plain tuples only, loaded with ``weights_only=True``). Building the cache from the released pickles takes
    ~30 s per split on a laptop; loading it takes well under a second.
    """
    cpath = Path(data_root) / test_case / f"{part}.prepared.pt"
    if cache and cpath.exists():
        raw = th.load(cpath, weights_only=True)
        return {k: PreparedSplit(*v) for k, v in raw.items()}
    prepared = prepare(*load_split(Path(data_root), test_case, part))
    if cache:
        tmp = cpath.with_suffix(".pt.tmp")
        th.save({k: tuple(v) for k, v in prepared.items()}, tmp)
        tmp.replace(cpath)
    return prepared


def _pad_slice(flat: th.Tensor, off: th.Tensor, lo: int, hi: int, pad_value: int, device: th.device) -> Tuple[th.Tensor, th.Tensor]:
    """Rows ``lo:hi`` of a flat/offset list as a padded ``(hi - lo, max_len)`` tensor on ``device`` plus the lengths."""
    o = off[lo:hi + 1].to(device)
    lengths = o[1:] - o[:-1]
    width = int(lengths.max().item()) if lengths.numel() else 0
    vals = flat[int(off[lo]):int(off[hi])].to(device)
    pos = th.arange(width, device=device).unsqueeze(0)  # (1, W)
    valid = pos < lengths.unsqueeze(1)  # (B, W)
    idx = (o[:-1] - int(off[lo])).unsqueeze(1) + pos  # flat index of each slot
    padded = th.full((hi - lo, width), pad_value, dtype=th.long, device=device)
    padded[valid] = vals[idx[valid]]
    return padded, lengths


def _ranks_padded(scores: th.Tensor, a_pad: th.Tensor, n_hard: th.Tensor, f_pad: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Core of :func:`filtered_ranks_batch` on padded tensors (``a_pad`` padded with 0, ``f_pad`` with -1)."""
    B, N = scores.shape
    dev = scores.device
    _, order = th.sort(scores, dim=1, descending=True, stable=True)
    pos = th.empty_like(order)
    pos.scatter_(1, order, th.arange(N, device=dev).unsqueeze(0).expand(B, N))  # unfiltered rank - 1 (ties: lower index first)
    pos_a = pos.gather(1, a_pad)  # (B, Hmax)
    pos_f = th.where(f_pad >= 0, pos.gather(1, f_pad.clamp_min(0)), th.full_like(f_pad, N + 1))
    pos_f, _ = th.sort(pos_f, dim=1)
    above = th.searchsorted(pos_f, pos_a, right=False)  # #{filtered f : pos[f] < pos[a]}; a itself is not counted
    ranks = (1 + pos_a - above).to(th.float)
    valid = th.arange(a_pad.shape[1], device=dev).unsqueeze(0) < n_hard.unsqueeze(1)
    q_of_pair = th.arange(B, device=dev).unsqueeze(1).expand_as(valid)[valid]
    return ranks[valid], q_of_pair


def filtered_ranks_batch(scores: th.Tensor, easy: Sequence[Sequence[int]], hard: Sequence[Sequence[int]]) -> Tuple[th.Tensor, th.Tensor]:
    """Vectorised :func:`filtered_ranks` for a batch of queries; runs on the device of ``scores``.

    ``scores`` is ``(B, N)``; ``easy[i]``/``hard[i]`` are the answer ids of query ``i``. Returns ``(ranks, query)``
    over all ``P = sum_i |hard[i]|`` (query, hard answer) pairs, where ``ranks[p]`` is exactly what
    ``filtered_ranks(scores[i], easy[i], hard[i])`` returns for that answer (same tie rule) and ``query[p] = i``.

    Method: one stable descending sort per query gives every entity's unfiltered position ``pos`` (ties by lower
    index); the filtered rank of a hard answer ``a`` is ``1 + pos[a] - #{f in easy ∪ hard, f != a : pos[f] < pos[a]}``,
    the subtraction done by ``searchsorted`` on the sorted filtered positions. Cost O(B·N log N), independent of
    the number of answers (E-OMQA ``pi`` queries have thousands of hard answers each).
    """
    scores = scores.detach()
    dev = scores.device
    pad = th.nn.utils.rnn.pad_sequence
    hard_t = [th.as_tensor(list(h), dtype=th.long) for h in hard]
    filt_t = [th.as_tensor(list(e) + list(h), dtype=th.long) for e, h in zip(easy, hard)]
    n_hard = th.tensor([t.numel() for t in hard_t], dtype=th.long, device=dev)
    if int(n_hard.sum()) == 0:
        return th.zeros(0, device=dev), th.zeros(0, dtype=th.long, device=dev)
    a_pad = pad(hard_t, batch_first=True, padding_value=0).to(dev)
    f_pad = pad(filt_t, batch_first=True, padding_value=-1).to(dev)
    return _ranks_padded(scores, a_pad, n_hard, f_pad)

def evaluate(
    score_fn: Callable[[Any, List[Any]], th.Tensor],
    queries_by_structure: Union[Dict[Any, List[Any]], Dict[Any, PreparedSplit]],
    easy: Optional[Dict[Any, Set[int]]] = None,
    hard: Optional[Dict[Any, Set[int]]] = None,
    batch_size: int = 256,
    structures: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the protocol. ``score_fn(structure, queries) -> (B, n_entities)`` scores (higher = better).

    ``queries_by_structure`` is either the first element of :func:`load_split` (with ``easy``/``hard``) or the
    output of :func:`prepare` (answer sets already converted; no Python-side work per evaluation).
    Returns ``{shape: {"mrr", "hits1", "hits3", "hits10", "n"}}`` plus ``"avg"`` (mean over shapes).
    """
    prepared: Dict[Any, PreparedSplit]
    if queries_by_structure and all(isinstance(v, PreparedSplit) for v in queries_by_structure.values()):
        prepared = queries_by_structure  # type: ignore[assignment]
    else:
        assert easy is not None and hard is not None
        prepared = prepare(queries_by_structure, easy, hard)  # type: ignore[arg-type]
    out: Dict[str, Dict[str, float]] = {}
    for structure, ps in prepared.items():
        name = QUERY_NAMES.get(structure, str(structure))
        if structures is not None and name not in set(structures):
            continue
        qs = ps.queries
        sums = {k: 0.0 for k in ("mrr", "hits1", "hits3", "hits10")}
        n_q = 0
        for i in range(0, len(qs), batch_size):
            batch = qs[i:i + batch_size]
            scores = score_fn(structure, batch).detach()  # (B, N), any device; ranking runs where the scores are
            a_pad, n_hard = _pad_slice(ps.hard_flat, ps.hard_off, i, i + len(batch), 0, scores.device)
            f_pad, _ = _pad_slice(ps.filt_flat, ps.filt_off, i, i + len(batch), -1, scores.device)
            r, qi = _ranks_padded(scores, a_pad, n_hard, f_pad)
            cnt = th.bincount(qi, minlength=len(batch)).to(r.dtype)
            per_q = th.stack([th.bincount(qi, weights=w, minlength=len(batch)) / cnt
                              for w in (1.0 / r, (r <= 1).to(r.dtype), (r <= 3).to(r.dtype), (r <= 10).to(r.dtype))])
            tot = per_q.sum(1).tolist()  # one host sync per batch
            for k, v in zip(("mrr", "hits1", "hits3", "hits10"), tot):
                sums[k] += v
            n_q += len(batch)
        if n_q:
            out[name] = {k: v / n_q for k, v in sums.items()}
            out[name]["n"] = float(n_q)
    if out:
        out["avg"] = {k: float(np.mean([m[k] for m in out.values()])) for k in ("mrr", "hits1", "hits3", "hits10")}
    return out


def _self_test() -> None:
    s = th.tensor([0.9, 0.5, 0.7, 0.1, 0.7])
    # hard = {3}: entities scoring higher: 0, 1, 2, 4 -> rank 5; with easy {0}: rank 4
    assert filtered_ranks(s, [], [3]).tolist() == [5.0]
    assert filtered_ranks(s, [0], [3]).tolist() == [4.0]
    # hard = {4}: 0 higher; 2 ties with lower index -> rank 3; hard {2}: 0 higher, 4 ties with higher index -> rank 2
    assert filtered_ranks(s, [], [4]).tolist() == [3.0]
    assert filtered_ranks(s, [], [2]).tolist() == [2.0]
    # two hard answers filter each other
    assert filtered_ranks(s, [], [2, 4]).tolist() == [2.0, 2.0]
    # batched version agrees with the per-query one on random scores with many ties
    g = th.Generator().manual_seed(0)
    sc = th.randint(0, 5, (40, 300), generator=g).float()
    easy = [sorted(set(th.randint(0, 300, (3,), generator=g).tolist())) for _ in range(40)]
    hard = [sorted(set(th.randint(0, 300, (4,), generator=g).tolist()) - set(e)) for e in easy]
    rb, qb = filtered_ranks_batch(sc, easy, hard)
    ref = th.cat([filtered_ranks(sc[i], easy[i], hard[i]) for i in range(40)])
    assert th.equal(rb, ref), (rb, ref)
    assert qb.tolist() == [i for i in range(40) for _ in hard[i]]
    # prepared/padded path equals the list path (drop queries without hard answers, as prepare does)
    keep = [i for i in range(40) if hard[i]]
    qbs = {("e", ("r",)): keep}
    prep = prepare(qbs, {i: set(easy[i]) for i in keep}, {i: set(hard[i]) for i in keep})
    ps = prep[("e", ("r",))]
    a_pad, n_hard = _pad_slice(ps.hard_flat, ps.hard_off, 0, len(keep), 0, sc.device)
    f_pad, _ = _pad_slice(ps.filt_flat, ps.filt_off, 0, len(keep), -1, sc.device)
    rp, qp = _ranks_padded(sc[keep], a_pad, n_hard, f_pad)
    assert th.equal(rp, th.cat([filtered_ranks(sc[i], easy[i], hard[i]) for i in keep]))
    res = evaluate(lambda s_, qs: sc[qs], prep)
    res2 = evaluate(lambda s_, qs: sc[qs], qbs, {i: set(easy[i]) for i in keep}, {i: set(hard[i]) for i in keep})
    assert res == res2, (res, res2)
    print("protocol self-test ok")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m experiments.omqa.protocol --data-root D [--cases ...] [--parts ...]`` builds the prepared caches."""
    import argparse
    import time
    ap = argparse.ArgumentParser(description="build <case>/<part>.prepared.pt caches (or run the self-test with no arguments)")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--cases", nargs="*", default=["test_ind_ded", "test_ded", "test_ind"])
    ap.add_argument("--parts", nargs="*", default=["valid", "test"])
    a = ap.parse_args(argv)
    if a.data_root is None:
        _self_test()
        return 0
    for case in a.cases:
        for part in a.parts:
            t0 = time.perf_counter()
            pr = load_prepared(a.data_root, case, part)
            print(f"[prepare] {case}/{part}: {sum(len(v.queries) for v in pr.values())} queries, "
                  f"{sum(v.hard_flat.numel() for v in pr.values())} hard answers, {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
