"""Train GroundedEL (translation-box roles) on a query-answering benchmark and evaluate with its protocol.

Benchmarks: ``--benchmark eomqa`` (E-OMQA LUBM/NELL; tables from ``build_eomqa.py``; Q2B protocol on the test
cases ``test_ind`` / ``test_ded`` / ``test_ind_ded``) and ``--benchmark tar`` (TAR YAGO4/DBpedia; tables from
``build_tar.py``; TAR protocol, entity- and concept-level answers).

Training: the KB tables (TBox + ABox) are the only training signal by default (model construction); the
model's own per-term losses (:meth:`GroundedEL.compute_batch_losses`) sample ``--batch`` rows per axiom type per
step, so one epoch = ``ceil(n_role_assertions / batch)`` steps. τ is annealed geometrically over the first
``--tau-epochs`` epochs; Adam with cosine LR decay to ``--lr-final``. Optional ``--query-loss {none,plain,onto}``:
multi-hop training queries from ``train_plain`` or ``train_onto`` (shapes ``--query-shapes``, default 2p/3p/2i/3i)
supervised with the witness max-min margin loss :func:`mowl.nn.el.grounded.queries.query_witness_loss`
(``--query-weight``, ``--query-batch``, ``--query-neg``, ``--query-k``, ``--query-margin``).
Checkpoint selection (``--select``): ``best`` keeps the epoch with the best validation score (``--valid-metric``
mrr or hits3 on ``--valid-shapes``, default all shapes; E-OMQA: ``valid-*`` of the first test case; TAR: test items
[0:1000]); ``last`` tests the final epoch (validation is only logged). ``--save-last`` also writes ``last.pt``.

Scores: entity candidates by crisp depth in the compiled multi-box (``mowl.nn.el.grounded.queries``); concept
candidates by overlap depth. ``--exact`` additionally reports the element-semantics variant.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

import mowl

if not getattr(mowl, "_jvm_started", False):  # mowl.models imports OWLAPI classes; JVM needed only for the import
    try:
        mowl.init_jvm(os.environ.get("MOWL_JVM_MEMORY", "4g"))
    except Exception:  # pragma: no cover - already started
        pass

from mowl.models import GroundedEL
from mowl.nn.el.grounded import queries as Q

try:  # the IRI GroundedEL uses to recognise owl:Thing
    from mowl.owlapi.defaults import TOP as TOP_IRI
except ImportError:  # pragma: no cover
    TOP_IRI = "http://www.w3.org/2002/07/owl#Thing"

from experiments.omqa import protocol as omqa_protocol
from experiments.omqa.build_eomqa import norm, safe_load
from experiments.tar import protocol as tar_protocol


def _rename_top(class_index: Dict[str, int]) -> Dict[str, int]:
    return {(TOP_IRI if k == "owl:Thing" else k): v for k, v in class_index.items()}


class Scorer:
    """Compile and score Query2Box-style queries against a trained GroundedEL (translation mode)."""

    def __init__(self, model: GroundedEL, tables: Dict[str, Any], benchmark: str, exact: bool = False, max_boxes: int = 64,
                 scorer: str = "region", beam_k: int = 64):
        self.model = model
        self.module = model.module
        self.benchmark = benchmark
        self.n_unknown_anchor = 0  # anchors not in the KB (counted, scored from a fallback element)
        self.exact = exact
        self.max_boxes = max_boxes
        self.scorer = scorer  # "region": crisp depth in the compiled multi-box; "beam": max-min over top-k witnesses
        self.beam_k = beam_k
        self.device = model.device
        nb_c = model.nb_classes
        if benchmark == "eomqa":
            e2i = tables["entity_to_individual"]
            e2c = tables["entity_to_class"]
            n_ent = max(max(e2i), max(e2c) if e2c else 0) + 1
            ent2el = th.full((n_ent,), -1, dtype=th.long)
            cls_of_ent = th.full((n_ent,), -1, dtype=th.long)
            for e, i in e2i.items():
                ent2el[e] = nb_c + i
            for e, c in e2c.items():
                cls_of_ent[e] = c
                ent2el[e] = c  # class entities are the class's own element
            r2role = tables["relation_to_role"]
            n_rel = max(list(r2role) + [tables.get("type_inv_rel") or 0, tables.get("type_fwd_rel") or 0]) + 1
            rel2role = th.full((n_rel,), -1, dtype=th.long)
            for r, k in r2role.items():
                rel2role[r] = k
            # the compiler works in module role ids; the type relations are not roles, so they get sentinel ids
            # beyond nb_rels (compile_query special-cases them before any role lookup)
            nb_r = int(self.module.nb_rels)
            self._ds_type_rel = tables.get("type_fwd_rel")
            self._ds_type_inv_rel = tables.get("type_inv_rel")
            self.type_rel = nb_r if self._ds_type_rel is not None else None
            self.type_inv_rel = nb_r + 1 if self._ds_type_inv_rel is not None else None
            if self._ds_type_rel is not None:
                rel2role[self._ds_type_rel] = self.type_rel
            if self._ds_type_inv_rel is not None:
                rel2role[self._ds_type_inv_rel] = self.type_inv_rel
            self.rel2role_list = rel2role.tolist()
            self.ent2el, self.cls_of_ent, self.rel2role = ent2el.to(self.device), cls_of_ent.to(self.device), rel2role.to(self.device)
            self.n_entities = n_ent
            # candidate elements: every dataset entity in order (class entities score through their class element);
            # entity ids that never occur as individual or class have no element and are masked out
            self.cand_valid = self.ent2el >= 0
            self.cand_elements = self.ent2el.clamp_min(0)
            self.ent2el_cpu = ent2el
            self.fallback_entity = int(th.nonzero(ent2el >= 0)[0].item())
        else:  # tar: ids are IRIs
            self.ind_index = tables["individual_index_dict"]
            self.cls_index = {(TOP_IRI if k == "owl:Thing" else k): v for k, v in tables["class_index_dict"].items()}
            self.role_index = tables["object_property_index_dict"]
            self.n_entities = len(self.ind_index)
            self.cand_elements = th.arange(nb_c, nb_c + self.n_entities, device=self.device)
            self.cand_valid = th.ones((self.n_entities,), dtype=th.bool, device=self.device)
            self.class_ids = th.tensor([v for k, v in sorted(self.cls_index.items(), key=lambda kv: kv[1]) if k != TOP_IRI], device=self.device)
            self.type_rel = self.type_inv_rel = None

    # --- id mapping -------------------------------------------------------------------------------------------
    def _map_query(self, q: Any) -> Any:
        """Map dataset ids/IRIs to (element id or class id for anchors, role id for relations)."""
        if isinstance(q, tuple) and len(q) == 2 and not isinstance(q[0], tuple) and isinstance(q[1], tuple):
            e, rels = q
            return (self._map_anchor(e), tuple(self._map_rel(r) for r in rels))
        if isinstance(q, tuple) and all(not isinstance(x, tuple) for x in q):
            return tuple(self._map_rel(r) for r in q)  # relation chain after a sub-query (pi/ip/up), or ('u',)
        if isinstance(q, tuple):
            return tuple(self._map_query(x) if isinstance(x, tuple) else x for x in q)
        return q

    def _map_anchor(self, e: Any) -> int:
        if self.benchmark == "eomqa":
            e = int(e)  # dataset entity id; compile_query uses entity_to_element / class_of_entity
            if e >= self.n_entities or self.ent2el_cpu[e] < 0:
                self.n_unknown_anchor += 1
                return self.fallback_entity
            return e
        k = self.ind_index.get(e)
        if k is None:  # anchor outside the KB (should not happen after the table rebuild): count, use element 0
            self.n_unknown_anchor += 1
            return 0
        return int(k)

    def _map_rel(self, r: Any) -> Any:
        if r == "u" or r == -1:
            return r
        if self.benchmark == "eomqa":
            k = self.rel2role_list[int(r)]
            if k < 0:
                raise KeyError(f"dataset relation {r} has no module role")
            return int(k)
        return int(self.role_index[r])

    # --- scoring ------------------------------------------------------------------------------------------------
    @th.no_grad()
    def region(self, structure: Any, queries: List[Any]) -> Any:
        mapped = [self._map_query(q) for q in queries]
        kw = dict(max_boxes=self.max_boxes, exact=self.exact)
        if self.benchmark == "eomqa":
            kw.update(type_rel=self.type_rel, type_inv_rel=self.type_inv_rel, entity_to_element=self.ent2el,
                      class_of_entity=self.cls_of_ent)
        return Q.compile_query(self.module, structure, mapped, **kw)

    @th.no_grad()
    def score_entities(self, structure: Any, queries: List[Any]) -> th.Tensor:
        if self.scorer == "beam":
            mapped = [self._map_query(q) for q in queries]
            kw: Dict[str, Any] = dict(k=self.beam_k, candidate_element_ids=self.cand_elements)
            if self.benchmark == "eomqa":
                kw.update(type_rel=self.type_rel, type_inv_rel=self.type_inv_rel, entity_to_element=self.ent2el,
                          class_of_entity=self.cls_of_ent)
            sc = Q.score_query_beam(self.module, structure, mapped, **kw)
        else:
            R = self.region(structure, queries)
            sc = Q.score_entities(self.module, R, candidate_element_ids=self.cand_elements)
        sc = sc.masked_fill(~self.cand_valid.unsqueeze(0), -1e6)
        return sc  # stays on the model device; the E-OMQA protocol ranks there (GPU-utilisation rule)

    @th.no_grad()
    def score_concepts(self, structure: Any, queries: List[Any]) -> th.Tensor:
        R = self.region(structure, queries)
        return Q.score_concepts(self.module, R, class_ids=self.class_ids)


def evaluate_eomqa(scorer: Scorer, prepared: Dict[Any, omqa_protocol.PreparedSplit], batch_size: int = 256) -> Dict[str, Any]:
    return omqa_protocol.evaluate(lambda s, qs: scorer.score_entities(s, qs), prepared, batch_size=batch_size)


class TrainQuerySampler:
    """Load E-OMQA training queries once and sample multi-hop query batches with positive and negative answer elements."""

    def __init__(self, data_root: Path, split: str, shapes: Sequence[str], scorer: Scorer,
                 batch_size: int = 1024, n_neg: int = 4, k: int = 16, margin: float = 0.0, seed: int = 0):
        self.scorer = scorer
        self.batch_size = max(1, int(batch_size))
        self.n_neg = max(0, int(n_neg))
        self.k = max(1, int(k))
        self.margin = float(margin)
        self.rng = np.random.default_rng(int(seed) + 2026)

        name_to_struct = {v: k for k, v in omqa_protocol.QUERY_NAMES.items()}
        train_dir = Path(data_root) / f"train_{split}"
        raw_queries = safe_load(train_dir / "train-queries.pkl")
        self.raw_answers = safe_load(train_dir / "train-answers.pkl")

        norm_queries: Dict[Any, List[Any]] = {}
        for st, qs in raw_queries.items():
            st_norm = norm(st) if not isinstance(st, str) else st
            norm_queries[st_norm] = list(qs)

        self.shapes: List[Tuple[str, Any, List[Any]]] = []
        self.cache: Dict[str, Dict[int, Optional[Tuple[Any, np.ndarray]]]] = {}
        for sh in shapes:
            if sh not in name_to_struct:
                raise ValueError(f"Unknown --query-shapes entry {sh!r}; expected one of {sorted(name_to_struct)}")
            st = name_to_struct[sh]
            q_list = norm_queries.get(st, [])
            if q_list:
                self.shapes.append((sh, st, q_list))
                self.cache[sh] = {}

        if scorer.benchmark == "eomqa":
            self.ent2el_np = scorer.ent2el_cpu.numpy()
            valid_cands = scorer.cand_elements[scorer.cand_valid].detach().cpu().numpy()
            self.valid_cand_elements = np.unique(valid_cands).astype(np.int64)
        else:
            self.ent2el_np = None
            self.valid_cand_elements = scorer.cand_elements.detach().cpu().numpy().astype(np.int64)

    def _get_mapped(self, shape_name: str, q_list: List[Any], idx: int) -> Optional[Tuple[Any, np.ndarray]]:
        sh_cache = self.cache[shape_name]
        if idx in sh_cache:
            return sh_cache[idx]
        q_raw = q_list[idx]
        ans_raw = self.raw_answers.get(q_raw)
        if ans_raw is None:
            ans_raw = self.raw_answers.get(norm(q_raw))
        if not ans_raw:
            sh_cache[idx] = None
            return None
        if self.ent2el_np is not None:
            n_ent = self.ent2el_np.shape[0]
            el_list = [int(self.ent2el_np[int(a)]) for a in ans_raw if 0 <= int(a) < n_ent and self.ent2el_np[int(a)] >= 0]
        else:
            nb_c = int(self.scorer.model.nb_classes)
            el_list = [nb_c + int(self.scorer.ind_index[a]) for a in ans_raw if a in self.scorer.ind_index]
        if not el_list:
            sh_cache[idx] = None
            return None
        ans_arr = np.unique(np.asarray(el_list, dtype=np.int64))
        mapped_q = self.scorer._map_query(q_raw)
        res = (mapped_q, ans_arr)
        sh_cache[idx] = res
        return res

    def step_loss(self, module: Any) -> th.Tensor:
        if not self.shapes:
            return module.elem_pos.weight.sum() * 0.0
        n_shapes = len(self.shapes)
        n_cands = int(self.valid_cand_elements.shape[0])
        total_b = 0
        weighted_loss: Optional[th.Tensor] = None

        for i, (shape_name, structure, q_list) in enumerate(self.shapes):
            b_i = (self.batch_size // n_shapes) + (1 if i < (self.batch_size % n_shapes) else 0)
            if b_i <= 0 or not q_list:
                continue
            n_q = len(q_list)
            mapped_qs: List[Any] = []
            pos_ans = np.empty((b_i,), dtype=np.int64)
            neg_ans = np.empty((b_i, self.n_neg), dtype=np.int64) if self.n_neg > 0 else None

            draw_idx = self.rng.integers(0, n_q, size=b_i)
            for row in range(b_i):
                idx = int(draw_idx[row])
                item = self._get_mapped(shape_name, q_list, idx)
                tries = 0
                while item is None and tries < 8:
                    idx = int(self.rng.integers(0, n_q))
                    item = self._get_mapped(shape_name, q_list, idx)
                    tries += 1
                if item is None:
                    continue
                mq, ans_arr = item
                mapped_qs.append(mq)
                k_row = len(mapped_qs) - 1
                if ans_arr.shape[0] == 1:
                    pos_ans[k_row] = ans_arr[0]
                else:
                    pos_ans[k_row] = ans_arr[int(self.rng.integers(0, ans_arr.shape[0]))]
                if neg_ans is not None and self.n_neg > 0:
                    cand_picks = self.valid_cand_elements[self.rng.integers(0, n_cands, size=self.n_neg)]
                    if ans_arr.shape[0] < n_cands:
                        for c_j in range(self.n_neg):
                            # Fast binary search on sorted ans_arr to exclude positive answers
                            pos_idx = int(np.searchsorted(ans_arr, cand_picks[c_j]))
                            while pos_idx < ans_arr.shape[0] and ans_arr[pos_idx] == cand_picks[c_j]:
                                cand_picks[c_j] = self.valid_cand_elements[int(self.rng.integers(0, n_cands))]
                                pos_idx = int(np.searchsorted(ans_arr, cand_picks[c_j]))
                    neg_ans[k_row] = cand_picks

            b_eff = len(mapped_qs)
            if b_eff == 0:
                continue
            pos_sub = pos_ans[:b_eff]
            neg_sub = neg_ans[:b_eff] if neg_ans is not None else None
            loss_i = Q.query_witness_loss(
                module,
                structure,
                mapped_qs,
                pos_sub,
                neg_sub,
                k=self.k,
                margin=self.margin,
                type_rel=self.scorer.type_rel,
                type_inv_rel=self.scorer.type_inv_rel,
                entity_to_element=self.scorer.ent2el if self.scorer.benchmark == "eomqa" else None,
                class_of_entity=self.scorer.cls_of_ent if self.scorer.benchmark == "eomqa" else None,
            )
            assert isinstance(loss_i, th.Tensor)
            w_term = loss_i * float(b_eff)
            weighted_loss = w_term if weighted_loss is None else (weighted_loss + w_term)
            total_b += b_eff

        if weighted_loss is None or total_b == 0:
            return module.elem_pos.weight.sum() * 0.0
        return weighted_loss / float(total_b)


def evaluate_tar(scorer: Scorer, input_dir: Path, lo: int, hi: int, query_types: Sequence[str] = tar_protocol.QUERY_TYPES) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ent_to_index = scorer.ind_index
    cls_names = [k for k, v in sorted(scorer.cls_index.items(), key=lambda kv: kv[1]) if k != TOP_IRI]
    cls_to_index = {c: i for i, c in enumerate(cls_names)}
    for ans_type, to_index, fn in (("e", ent_to_index, scorer.score_entities), ("c", cls_to_index, scorer.score_concepts)):
        res: Dict[str, Dict[str, float]] = {}
        for qt in query_types:
            items = tar_protocol.load_test(input_dir, qt, ans_type, lo, hi)
            structure = tar_protocol.STRUCTURES[qt]
            logs: Dict[str, List[float]] = {"mrr": [], "hits1": [], "hits3": [], "hits10": [], "hits50": []}
            for i in range(0, len(items), 64):
                batch = items[i:i + 64]
                scores = fn(structure, [q for q, _, _ in batch]).cpu()
                for (q, ans, filt), s in zip(batch, scores):
                    targets = list(ans) if isinstance(ans, (set, frozenset, list)) else [ans]
                    filt_idx = [to_index[a] for a in filt if a in to_index]
                    for a in targets:
                        if a not in to_index:
                            continue
                        r = tar_protocol.filtered_rank(s, to_index[a], filt_idx)
                        logs["mrr"].append(1.0 / r)
                        for k in (1, 3, 10, 50):
                            logs[f"hits{k}"].append(float(r <= k))
            if logs["mrr"]:
                res[qt] = {k: float(np.mean(v)) for k, v in logs.items()}
                res[qt]["n"] = float(len(logs["mrr"]))
        if res:
            res["avg"] = {k: float(np.mean([m[k] for m in res.values()])) for k in ("mrr", "hits1", "hits3", "hits10", "hits50")}
        out[ans_type] = res
    return out


def build_model(tables: Dict[str, Any], args: argparse.Namespace) -> GroundedEL:
    kw: Dict[str, Any] = dict(
        class_index_dict=_rename_top(tables["class_index_dict"]),
        object_property_index_dict=tables["object_property_index_dict"],
        individual_index_dict=tables["individual_index_dict"],
        gci0=tables.get("gci0"), gci3=tables.get("gci3"), role_inclusion=tables.get("role_inclusion"),
        class_assertion=tables.get("class_assertion"), object_property_assertion=tables.get("object_property_assertion"),
        embed_dim=args.dim, n_components=args.K, n_role_components=args.Kr, batch_size=args.batch,
        learning_rate=args.lr, tau0=args.tau0, tau1=args.tau1, eps_min=args.eps_min, eps_role=args.eps_role,
        lambda_reg=args.lambda_reg, n_region_samples=args.n_region_samples, device=args.device, seed=args.seed,
        role_mode="translation", inverse_of=tables.get("inverse_of"), log_every=0,
        # model-level corrupted-fact negatives and dual weights (mOWL 3511f40); getattr: older results.json args lack them
        neg_weight=getattr(args, "neg_weight", 0.0), neg_k=getattr(args, "neg_k", 1),
        constraint_mode=getattr(args, "constraint_mode", "fixed"), dual_rho=getattr(args, "dual_rho", 1.0),
        dual_max=getattr(args, "dual_max", 1e3), dual_target=getattr(args, "dual_target", 0.0),
    )
    for k in ("gci0", "gci3", "role_inclusion", "class_assertion", "object_property_assertion"):
        if kw[k] is not None and int(kw[k].shape[0]) == 0:
            kw[k] = None
    return GroundedEL.from_tables(**kw)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--benchmark", choices=["eomqa", "tar"], required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--data-root", required=True, help="eomqa: data_<kb>/ with test_*/ ; tar: input/ dir")
    ap.add_argument("--test-cases", nargs="*", default=["test_ind_ded", "test_ded", "test_ind"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--Kr", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--tau0", type=float, default=1.0)
    ap.add_argument("--tau1", type=float, default=0.01)
    ap.add_argument("--tau-epochs", type=int, default=20)
    ap.add_argument("--eps-min", type=float, default=1e-2)
    ap.add_argument("--eps-role", type=float, default=1e-2)
    ap.add_argument("--neg-weight", type=float, default=0.0, help="weight of corrupted-fact negatives (0 = off)")
    ap.add_argument("--neg-k", type=int, default=1, help="corrupted facts per batch row")
    ap.add_argument("--constraint-mode", choices=["fixed", "lagrangian"], default="fixed",
                    help="lagrangian: dual-ascent weights on the fact/axiom/region terms (model.dual_update once per epoch)")
    ap.add_argument("--dual-rho", type=float, default=1.0)
    ap.add_argument("--dual-max", type=float, default=1e3)
    ap.add_argument("--dual-target", type=float, default=0.0)
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--n-region-samples", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--valid-shapes", nargs="*", default=None, help="restrict validation to these shapes (speed)")
    ap.add_argument("--max-boxes", type=int, default=64)
    ap.add_argument("--exact", action="store_true", help="also report the element-semantics scorer at the end")
    ap.add_argument("--scorer", choices=["region", "beam"], default="region", help="entity scorer for validation and test")
    ap.add_argument("--beam-k", type=int, default=64, help="witnesses per projection for --scorer beam")
    ap.add_argument("--query-loss", choices=["none", "plain", "onto"], default="none",
                    help="supervise multi-hop training queries via query_witness_loss (plain=train_plain, onto=train_onto)")
    ap.add_argument("--query-shapes", nargs="*", default=["2p", "3p", "2i", "3i"],
                    help="query shapes trained by --query-loss")
    ap.add_argument("--query-weight", type=float, default=0.1, help="weight of query_witness_loss added to total loss")
    ap.add_argument("--query-batch", type=int, default=1024, help="training queries per step, spread over --query-shapes")
    ap.add_argument("--query-neg", type=int, default=4, help="sampled negative answers per training query")
    ap.add_argument("--query-k", type=int, default=16, help="top-k witnesses per intermediate variable in query_witness_loss")
    ap.add_argument("--query-margin", type=float, default=0.0, help="hinge margin for query_witness_loss")
    ap.add_argument("--eval-only", default=None, help="skip training: load this checkpoint (best.pt/last.pt) and run the test stages")
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--max-epoch-seconds", type=float, default=0.0, help="stop training after this wall time (0: off)")
    ap.add_argument("--select", choices=["best", "last"], default="best", help="checkpoint to test: best validation score or final epoch")
    ap.add_argument("--valid-metric", choices=["mrr", "hits3"], default="mrr", help="validation score used for --select best")
    ap.add_argument("--save-last", action="store_true", help="also save the final-epoch state as last.pt")
    args = ap.parse_args(argv)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = th.load(args.tables, weights_only=False)
    t0 = time.perf_counter()
    model = build_model(tables, args)
    model.init_module()
    module = model.module
    params = [p for p in module.parameters() if p.requires_grad]
    opt = th.optim.Adam(params, lr=args.lr)
    n_assert = int(tables["object_property_assertion"].shape[0]) if tables.get("object_property_assertion") is not None else 0
    steps_per_epoch = max(1, math.ceil(n_assert / args.batch))
    nb_c, n_ind = int(model.nb_classes), len(tables["individual_index_dict"])
    print(f"[qa] {args.name}: classes {model.nb_classes} individuals {model.nb_elements - model.nb_classes} roles {model.nb_rels} "
          f"role assertions {n_assert} steps/epoch {steps_per_epoch} device {args.device} load {time.perf_counter() - t0:.1f}s", flush=True)

    data_root = Path(args.data_root)
    scorer = Scorer(model, tables, args.benchmark, exact=False, max_boxes=args.max_boxes, scorer=args.scorer, beam_k=args.beam_k)

    # E-OMQA splits are loaded and converted once, before training (the per-evaluation Python work would idle the GPU)
    valid_prep: Optional[Dict[Any, omqa_protocol.PreparedSplit]] = None
    test_prep: Dict[str, Dict[Any, omqa_protocol.PreparedSplit]] = {}
    if args.benchmark == "eomqa":
        t1 = time.perf_counter()
        valid_prep = omqa_protocol.load_prepared(data_root, args.test_cases[0], "valid")
        for tc in args.test_cases:
            test_prep[tc] = omqa_protocol.load_prepared(data_root, tc, "test")
        print(f"[qa] query splits loaded in {time.perf_counter() - t1:.0f}s", flush=True)

    query_sampler: Optional[TrainQuerySampler] = None
    if args.query_loss != "none" and not args.eval_only:
        tq0 = time.perf_counter()
        query_sampler = TrainQuerySampler(
            data_root, args.query_loss, args.query_shapes, scorer,
            batch_size=args.query_batch, n_neg=args.query_neg, k=args.query_k, margin=args.query_margin, seed=args.seed,
        )
        counts_str = " ".join(f"{sh}:{len(qs)}" for sh, _, qs in query_sampler.shapes)
        print(f"[qa] train_{args.query_loss} queries ({counts_str}) loaded in {time.perf_counter() - tq0:.1f}s", flush=True)

    def validate() -> Tuple[float, Dict[str, Any]]:
        module.eval()
        if args.benchmark == "eomqa":
            res = omqa_protocol.evaluate(lambda s, qs: scorer.score_entities(s, qs), valid_prep, structures=args.valid_shapes)
            mrr = res.get("avg", {}).get(args.valid_metric, 0.0)
        else:
            res = evaluate_tar(scorer, data_root, 0, 1000, query_types=args.valid_shapes or tar_protocol.QUERY_TYPES)
            mrr = float(np.mean([res[a]["avg"][args.valid_metric] for a in ("e", "c") if "avg" in res.get(a, {})]))
        module.train()
        return mrr, res

    history: List[Dict[str, Any]] = []
    best_mrr, best_epoch, best_state = -1.0, 0, None
    tau_total = max(1, args.tau_epochs)
    train_t0 = time.perf_counter()
    if args.eval_only:
        ck = th.load(args.eval_only, map_location=model.device, weights_only=False)
        module.load_state_dict({k: v.to(model.device) for k, v in ck["state_dict"].items()})
        best_epoch, best_mrr = int(ck.get("epoch", -1)), float(ck.get("val_mrr", float("nan")))
        print(f"[qa] eval-only: loaded {args.eval_only} (epoch {best_epoch}); scorer={args.scorer} beam_k={args.beam_k}", flush=True)
    for epoch in range(0 if args.eval_only else args.epochs):
        ep_t0 = time.perf_counter()
        cur_tau = model.step_tau(min(epoch, tau_total - 1), tau_total)
        lr = model.scheduled_lr(epoch, args.epochs, args.lr, args.lr_final, "cosine") if hasattr(model, "scheduled_lr") else args.lr
        for g in opt.param_groups:
            g["lr"] = lr
        sums: Dict[str, th.Tensor] = {}  # accumulated on device; one host sync per epoch (GPU-utilisation rule)
        for _ in range(steps_per_epoch):
            losses = model.compute_batch_losses()  # includes the "neg" term and the dual weights w_* when enabled
            if query_sampler is not None:
                q_loss = query_sampler.step_loss(module)
                losses["query"] = q_loss.detach()
                losses["total"] = losses["total"] + args.query_weight * q_loss
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            opt.step()
            model.clamp_dead_logits()
            for k, v in losses.items():
                if th.is_tensor(v) and v.ndim == 0:
                    sums[k] = sums[k] + v.detach() if k in sums else v.detach().clone()
        rec: Dict[str, Any] = {"epoch": epoch + 1, "tau": cur_tau, "lr": lr, "epoch_sec": time.perf_counter() - ep_t0}
        rec.update({k: float(v.item()) / steps_per_epoch for k, v in sums.items()})
        if args.constraint_mode == "lagrangian":
            rec["dual"] = model.dual_update({k: rec[k] for k in ("fact", "axioms", "region") if k in rec})
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            vt0 = time.perf_counter()
            mrr, vres = validate()
            rec["val_mrr"] = mrr
            rec["val"] = vres
            rec["val_sec"] = time.perf_counter() - vt0
            if mrr > best_mrr:
                best_mrr, best_epoch = mrr, epoch + 1
                if args.select == "best":
                    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                    th.save({"state_dict": best_state, "args": vars(args), "epoch": best_epoch, "val_mrr": best_mrr}, out_dir / "best.pt")
        history.append(rec)
        print(f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float) and k not in ("epoch_sec", "val_sec", "lr", "w_fact", "w_axioms", "w_region"))
              + (" w=(%.1f,%.1f,%.1f)" % (rec["dual"]["fact"], rec["dual"]["axioms"], rec["dual"]["region"]) if "dual" in rec else "")
              + f" lr={lr:.2e} time={rec['epoch_sec']:.1f}s" + (f" val_mrr={rec['val_mrr']:.4f} (best {best_mrr:.4f} @ {best_epoch}) val_time={rec['val_sec']:.0f}s" if "val_mrr" in rec else ""), flush=True)
        if args.max_epoch_seconds and time.perf_counter() - train_t0 > args.max_epoch_seconds:
            print("[qa] wall-time budget reached, stopping", flush=True)
            break

    last_epoch = len(history)
    if args.eval_only:
        args.select, best_state, last_epoch = "best", None, best_epoch
    if not args.eval_only and (args.save_last or args.select == "last"):
        th.save({"state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()}, "args": vars(args), "epoch": last_epoch},
                out_dir / "last.pt")
    if args.select == "best" and best_state is not None:
        module.load_state_dict({k: v.to(model.device) for k, v in best_state.items()})
    tested_epoch = best_epoch if args.select == "best" else last_epoch
    print(f"[qa] testing epoch {tested_epoch} (select={args.select}, valid metric {args.valid_metric}; best val {best_mrr:.4f} @ {best_epoch})", flush=True)
    module.eval()
    results: Dict[str, Any] = {"name": args.name, "args": vars(args), "best_epoch": best_epoch, "best_val_mrr": best_mrr,
                               "tested_epoch": tested_epoch, "select": args.select, "valid_metric": args.valid_metric,
                               "train_wall_sec": time.perf_counter() - train_t0, "history": history}

    def save() -> None:  # written after every evaluation stage so that a later failure keeps the earlier results
        results["n_unknown_anchor"] = scorer.n_unknown_anchor
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))

    if args.benchmark == "eomqa":
        for tc in args.test_cases:
            results[f"test_{tc}"] = evaluate_eomqa(scorer, test_prep[tc])
            print(f"[test {tc}] " + json.dumps({k: round(v["hits3"], 4) for k, v in results[f'test_{tc}'].items()}), flush=True)
            save()
        if args.exact:
            ex = Scorer(model, tables, args.benchmark, exact=True, max_boxes=args.max_boxes)
            for tc in args.test_cases:
                results[f"test_{tc}_exact"] = evaluate_eomqa(ex, test_prep[tc])
                print(f"[test {tc} exact] " + json.dumps({k: round(v["hits3"], 4) for k, v in results[f'test_{tc}_exact'].items()}), flush=True)
                save()
    else:
        results["test"] = evaluate_tar(scorer, data_root, 1000, 2000)
        print("[test e] " + json.dumps({k: round(v["mrr"], 4) for k, v in results["test"]["e"].items()}), flush=True)
        print("[test c] " + json.dumps({k: round(v["mrr"], 4) for k, v in results["test"]["c"].items()}), flush=True)
        save()
        if args.exact:
            ex = Scorer(model, tables, args.benchmark, exact=True, max_boxes=args.max_boxes)
            results["test_exact"] = evaluate_tar(ex, data_root, 1000, 2000)
            print("[test e exact] " + json.dumps({k: round(v["mrr"], 4) for k, v in results["test_exact"]["e"].items()}), flush=True)
            print("[test c exact] " + json.dumps({k: round(v["mrr"], 4) for k, v in results["test_exact"]["c"].items()}), flush=True)
            save()
    try:
        results["crisp_report"] = model.evaluate_model(use_reasoner=False, n_eval=args.n_eval)
    except Exception as exc:  # pragma: no cover
        results["crisp_report"] = {"error": repr(exc)}
    save()
    print(f"[done] {args.name} best epoch {best_epoch} val MRR {best_mrr:.4f}; results {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
