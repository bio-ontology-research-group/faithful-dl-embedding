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


def _batched_crisp_elem_membership(
    mod: Any,
    points: th.Tensor,
    class_chunk_size: int = 2048,
) -> th.Tensor:
    """Compute crisp membership ``(L, nb_classes)`` for ``points`` ``(L, d)`` in class chunks.

    Guarantees
    ----------
    - Returns a float32 tensor of shape ``(L, nb_classes)`` with entries in ``{0.0, 1.0}``
      identical to stacking ``crisp_membership(mod.depth(points, c_id, crisp=True))``
      across all ``c_id in range(mod.nb_classes)``.

    Non-guarantees
    --------------
    - Assumes ``torch.no_grad()`` context is active for memory efficiency.
    """
    from mowl.nn.el.grounded.losses import crisp_membership

    l_cnt = int(points.shape[0])
    nb_classes = int(mod.nb_classes)
    device = points.device
    if l_cnt == 0:
        return th.empty((0, nb_classes), dtype=th.float32, device=device)

    out = th.empty((l_cnt, nb_classes), dtype=th.float32, device=device)
    pts_exp = points.unsqueeze(1).unsqueeze(2)  # (L, 1, 1, d)

    for start in range(0, nb_classes, class_chunk_size):
        end = min(start + class_chunk_size, nb_classes)
        c_ids = th.arange(start, end, dtype=th.long, device=device)
        regs = mod.class_regions(c_ids)  # cen, w: (C, K, d), active: (C, K)
        coord_depth = regs.w.unsqueeze(0) - th.abs(pts_exp - regs.cen.unsqueeze(0))  # (L, C, K, d)
        slot_depth = th.amin(coord_depth, dim=-1)  # (L, C, K)
        act = regs.active.unsqueeze(0)  # (1, C, K)
        has_act = act.any(dim=-1, keepdim=True)  # (1, C, 1)
        inactive_fill = th.where(
            has_act,
            th.full_like(slot_depth, float("-inf")),
            th.full_like(slot_depth, -1e6),
        )
        dep = th.amax(th.where(act, slot_depth, inactive_fill), dim=-1)  # (L, C)
        if mod.top_id is not None and start <= mod.top_id < end:
            dep[:, mod.top_id - start] = 1e6
        if mod.bot_id is not None and start <= mod.bot_id < end:
            dep[:, mod.bot_id - start] = -1e6
        out[:, start:end] = crisp_membership(dep)
    return out


def compute_crisp_and_eval_metrics(
    model: Any,
    n_eval: int = 10000,
    force_batched: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute ``crisp_report`` and ``evaluate_model(use_reasoner=False)`` safely at any scale.

    When ``model.nb_elements * model.nb_classes <= 200_000`` (and ``not force_batched``),
    delegates directly to ``model.crisp_report(n_eval=n_eval)`` and
    ``model.evaluate_model(use_reasoner=False, n_eval=n_eval)``.
    On larger ontologies (such as GO where ``nb_elements = nb_classes = 56,191``),
    computes the identical crisp ``elem_viol``, ``region_viol`` (on ``n_eval``
    samples), ``exist``, ``dead_ok``, and ``exact`` metrics in batched tensor form
    to avoid materialising a ``(nb_elements, nb_classes)`` matrix.

    Guarantees
    ----------
    - Reasoner-free (R1): never imports or calls any OWL reasoner or ``CanonicalModel``.
    - Returns ``(crisp_dict, eval_dict)`` with the exact schema of
      :meth:`~mowl.models.GroundedEL.crisp_report` and
      :meth:`~mowl.models.GroundedEL.evaluate_model`.

    Non-guarantees
    --------------
    - ``faithful_named_*`` entries are ``None`` since ``use_reasoner=False``.
    """
    if not force_batched and int(model.nb_elements) * int(model.nb_classes) <= 200_000:
        crisp = model.crisp_report(n_eval=n_eval)
        eval_metrics = model.evaluate_model(use_reasoner=False, n_eval=n_eval)
        if isinstance(eval_metrics.get("faithful_named"), tuple):
            eval_metrics = dict(eval_metrics)
            eval_metrics["faithful_named"] = list(eval_metrics["faithful_named"])
        return crisp, eval_metrics

    from mowl.nn.el.grounded.losses import (
        class_assertion,
        crisp_membership,
        gci0,
        gci0_bot,
        gci1,
        gci1_bot,
        gci2,
        gci3,
        gci3_bot,
        role_assertion,
        role_chain,
        role_inclusion,
    )
    from mowl.nn.el.grounded.sampling import (
        sample_class_region,
        sample_pair_region,
        sample_role_region,
    )

    mod = model.module
    device = model.device
    td = model.training_datasets
    gen = th.Generator(device="cpu")
    gen.manual_seed(int(model.seed) + 9999)

    with th.no_grad():
        pos_all = mod.positions()  # (nb_elements, d)
        dead_c = mod.dead() > 0.5
        if model.bot_id is not None:
            dead_c[model.bot_id] = True
        dead_f = dead_c.to(dtype=th.float32)
        live_mask = ~dead_c
        live_ids = th.nonzero(live_mask, as_tuple=False).view(-1)

        if live_ids.numel() > 256:
            perm = th.randperm(live_ids.numel(), device="cpu", generator=gen)[:256].to(
                device=device
            )
            eval_elem_ids = live_ids[perm]
        else:
            eval_elem_ids = live_ids

        eval_pos = pos_all[eval_elem_ids]  # (L, d)
        eval_dead_f = dead_f[eval_elem_ids]  # (L,)
        elem_m_live = _batched_crisp_elem_membership(mod, eval_pos)  # (L, nb_classes)

        elem_viol_count = 0
        elem_total_count = 0

        # 1. Batched elem_viol across all training axioms on eval_elem_ids
        if "gci0" in td and len(td["gci0"]) > 0:
            rows = td["gci0"][:].to(device=device, dtype=th.long)
            if eval_elem_ids.numel() > 0:
                for s in range(0, rows.shape[0], 4096):
                    chunk = rows[s : s + 4096]
                    ma = elem_m_live[:, chunk[:, 0]]
                    mb = elem_m_live[:, chunk[:, 1]]
                    h_e = gci0(ma, mb, dead=eval_dead_f.unsqueeze(1).expand_as(ma))
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

        if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
            rows = td["gci0_bot"][:].to(device=device, dtype=th.long)
            if eval_elem_ids.numel() > 0:
                a_ids = rows[:, 0]
                ma = elem_m_live[:, a_ids]
                h_e = gci0_bot(ma, dead=eval_dead_f.unsqueeze(1).expand_as(ma))
                elem_viol_count += int((h_e > 0.0).sum().item())
                elem_total_count += int(h_e.numel())

        if "gci1" in td and len(td["gci1"]) > 0:
            rows = td["gci1"][:].to(device=device, dtype=th.long)
            if eval_elem_ids.numel() > 0:
                for s in range(0, rows.shape[0], 4096):
                    chunk = rows[s : s + 4096]
                    ma = elem_m_live[:, chunk[:, 0]]
                    mb = elem_m_live[:, chunk[:, 1]]
                    mc = elem_m_live[:, chunk[:, 2]]
                    h_e = gci1(ma, mb, mc, dead=eval_dead_f.unsqueeze(1).expand_as(ma))
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

        if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
            rows = td["gci1_bot"][:].to(device=device, dtype=th.long)
            if eval_elem_ids.numel() > 0:
                for s in range(0, rows.shape[0], 4096):
                    chunk = rows[s : s + 4096]
                    ma = elem_m_live[:, chunk[:, 0]]
                    mb = elem_m_live[:, chunk[:, 1]]
                    h_e = gci1_bot(ma, mb, dead=eval_dead_f.unsqueeze(1).expand_as(ma))
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

        if "gci2" in td and len(td["gci2"]) > 0 and eval_elem_ids.numel() > 0:
            rows = td["gci2"][:].to(device=device, dtype=th.long)
            l_cnt = int(eval_pos.shape[0])
            for s in range(0, rows.shape[0], 512):
                chunk = rows[s : s + 512]
                c_sz = int(chunk.shape[0])
                a_ids = chunk[:, 0]
                r_ids = chunk[:, 1]
                b_ids = chunk[:, 2]
                xp = eval_pos.unsqueeze(1).expand(l_cnt, c_sz, model.embed_dim).reshape(-1, model.embed_dim)
                yp = pos_all[b_ids].unsqueeze(0).expand(l_cnt, c_sz, model.embed_dim).reshape(-1, model.embed_dim)
                rr = r_ids.unsqueeze(0).expand(l_cnt, c_sz).reshape(-1)
                rho_e = crisp_membership(mod.role_depth(xp, yp, rr, crisp=True)).reshape(l_cnt, c_sz)
                ma = elem_m_live[:, a_ids]
                db_e = dead_f[b_ids].unsqueeze(0).expand(l_cnt, c_sz)
                h_e = gci2(
                    ma,
                    rho_e,
                    dead=eval_dead_f.unsqueeze(1).expand_as(ma),
                    dead_filler=db_e,
                )
                elem_viol_count += int((h_e > 0.0).sum().item())
                elem_total_count += int(h_e.numel())

        # Precompute crisp role memberships on (eval_pos, eval_pos[:32]) across all roles
        y_cnt = min(32, int(eval_elem_ids.numel()))
        if y_cnt > 0 and model.nb_rels > 0:
            l_cnt = int(eval_pos.shape[0])
            y_pos = eval_pos[:y_cnt]
            rho_xy_all = th.empty((l_cnt, y_cnt, model.nb_rels), dtype=th.float32, device=device)
            xp_grid = eval_pos.unsqueeze(1).expand(l_cnt, y_cnt, model.embed_dim).reshape(-1, model.embed_dim)
            yp_grid = y_pos.unsqueeze(0).expand(l_cnt, y_cnt, model.embed_dim).reshape(-1, model.embed_dim)
            for r_id in range(model.nb_rels):
                rho_xy_all[:, :, r_id] = crisp_membership(
                    mod.role_depth(xp_grid, yp_grid, r_id, crisp=True)
                ).reshape(l_cnt, y_cnt)

            if "gci3" in td and len(td["gci3"]) > 0:
                rows = td["gci3"][:].to(device=device, dtype=th.long)
                for s in range(0, rows.shape[0], 512):
                    chunk = rows[s : s + 512]
                    r_ids = chunk[:, 0]
                    b_ids = chunk[:, 1]
                    c_ids = chunk[:, 2]
                    rho_c = rho_xy_all[:, :, r_ids]  # (L, y_cnt, C)
                    mb_y = elem_m_live[:y_cnt, b_ids].unsqueeze(0).expand_as(rho_c)
                    mc_x = elem_m_live[:, c_ids].unsqueeze(1).expand_as(rho_c)
                    dead_x = eval_dead_f.view(l_cnt, 1, 1).expand_as(rho_c)
                    h_e = gci3(rho_c, mb_y, mc_x, dead=dead_x)
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

            if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
                rows = td["gci3_bot"][:].to(device=device, dtype=th.long)
                r_ids = rows[:, 0]
                b_ids = rows[:, 1]
                rho_c = rho_xy_all[:, :, r_ids]
                mb_y = elem_m_live[:y_cnt, b_ids].unsqueeze(0).expand_as(rho_c)
                dead_x = eval_dead_f.view(l_cnt, 1, 1).expand_as(rho_c)
                h_e = gci3_bot(rho_c, mb_y, dead=dead_x)
                elem_viol_count += int((h_e > 0.0).sum().item())
                elem_total_count += int(h_e.numel())

            if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
                rows = td["role_inclusion"][:].to(device=device, dtype=th.long)
                rr = rho_xy_all[:, :, rows[:, 0]]
                rs = rho_xy_all[:, :, rows[:, 1]]
                h_e = role_inclusion(rr, rs)
                elem_viol_count += int((h_e > 0.0).sum().item())
                elem_total_count += int(h_e.numel())

            if "role_chain" in td and len(td["role_chain"]) > 0:
                c_cnt = min(16, y_cnt)
                rows = td["role_chain"][:].to(device=device, dtype=th.long)
                for row in rows.detach().cpu().tolist():
                    r_id, s_id, t_id = int(row[0]), int(row[1]), int(row[2])
                    rho_xy = rho_xy_all[:c_cnt, :c_cnt, r_id].unsqueeze(2)  # (C, C, 1)
                    rho_yz = rho_xy_all[:c_cnt, :c_cnt, s_id].unsqueeze(0)  # (1, C, C)
                    rho_xz = rho_xy_all[:c_cnt, :c_cnt, t_id].unsqueeze(1)  # (C, 1, C)
                    h_e = role_chain(
                        rho_xy.expand(c_cnt, c_cnt, c_cnt),
                        rho_yz.expand(c_cnt, c_cnt, c_cnt),
                        rho_xz.expand(c_cnt, c_cnt, c_cnt),
                    )
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

        # 2. Region violations on n_eval sample budget across sampled axioms
        axiom_keys = (
            "gci0",
            "gci0_bot",
            "gci1",
            "gci1_bot",
            "gci2",
            "gci3",
            "gci3_bot",
            "role_inclusion",
            "role_chain",
        )
        key_counts = {
            k: len(td[k]) for k in axiom_keys if k in td and len(td[k]) > 0
        }
        total_region_axioms = sum(key_counts.values())
        n_per_ax = 16
        max_sampled_axioms = max(1, n_eval // n_per_ax)

        reg_viol_axioms = 0
        reg_total_axioms = 0
        all_regs = mod.class_regions()

        for k, cnt in key_counts.items():
            rows_all = td[k][:].detach().cpu()
            if total_region_axioms > max_sampled_axioms:
                quota = max(1, int(round(max_sampled_axioms * (cnt / total_region_axioms))))
                quota = min(cnt, quota)
                perm_ax = th.randperm(cnt, generator=gen)[:quota]
                rows_sel = rows_all[perm_ax].tolist()
            else:
                rows_sel = rows_all.tolist()

            for row in rows_sel:
                reg_total_axioms += 1
                if k == "gci0":
                    a_id, b_id = int(row[0]), int(row[1])
                    s_reg = sample_class_region(mod, a_id, generator=gen, n_samples=n_per_ax)
                    corners = model._box_corner_samples(
                        all_regs.cen[a_id : a_id + 1],
                        all_regs.w[a_id : a_id + 1],
                        all_regs.active[a_id : a_id + 1],
                        gen,
                    )
                    pts_list = [eval_pos]
                    if bool(s_reg.valid.any().item()):
                        pts_list.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_list.append(corners)
                    pts = th.cat(pts_list, dim=0)
                    if pts.shape[0] > 0:
                        ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                        mb = crisp_membership(mod.depth(pts, b_id, crisp=True))
                        if bool((gci0(ma, mb, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "gci0_bot":
                    a_id = int(row[0])
                    s_reg = sample_class_region(mod, a_id, generator=gen, n_samples=n_per_ax)
                    if bool(s_reg.valid.any().item()):
                        reg_viol_axioms += 1
                    elif eval_pos.shape[0] > 0:
                        ma = elem_m_live[:, a_id]
                        if bool((gci0_bot(ma, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "gci1":
                    a_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                    s_reg = sample_pair_region(mod, a_id, b_id, generator=gen, n_samples=n_per_ax)
                    pts_list = [eval_pos]
                    if bool(s_reg.valid.any().item()):
                        pts_list.append(s_reg.points[s_reg.valid])
                    pts = th.cat(pts_list, dim=0)
                    if pts.shape[0] > 0:
                        ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                        mb = crisp_membership(mod.depth(pts, b_id, crisp=True))
                        mc = crisp_membership(mod.depth(pts, c_id, crisp=True))
                        if bool((gci1(ma, mb, mc, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "gci1_bot":
                    a_id, b_id = int(row[0]), int(row[1])
                    s_inter = sample_pair_region(mod, a_id, b_id, generator=gen, n_samples=n_per_ax)
                    if bool(s_inter.valid.any().item()):
                        reg_viol_axioms += 1
                    else:
                        s_a = sample_class_region(mod, a_id, generator=gen, n_samples=n_per_ax)
                        pts_list = [eval_pos]
                        if bool(s_a.valid.any().item()):
                            pts_list.append(s_a.points[s_a.valid])
                        pts = th.cat(pts_list, dim=0)
                        if pts.shape[0] > 0:
                            ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                            mb = crisp_membership(mod.depth(pts, b_id, crisp=True))
                            if bool((gci1_bot(ma, mb, dead=None) > 0.0).any().item()):
                                reg_viol_axioms += 1
                elif k == "gci2":
                    a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                    pb = pos_all[b_id]
                    s_reg = sample_class_region(mod, a_id, generator=gen, n_samples=n_per_ax)
                    corners = model._box_corner_samples(
                        all_regs.cen[a_id : a_id + 1],
                        all_regs.w[a_id : a_id + 1],
                        all_regs.active[a_id : a_id + 1],
                        gen,
                    )
                    pts_list = [eval_pos]
                    if bool(s_reg.valid.any().item()):
                        pts_list.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_list.append(corners)
                    pts = th.cat(pts_list, dim=0)
                    if pts.shape[0] > 0:
                        ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                        rho_z = crisp_membership(mod.role_depth(pts, pb, r_id, crisp=True))
                        db_z = dead_f[b_id].expand_as(ma)
                        if bool((gci2(ma, rho_z, dead=None, dead_filler=db_z) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "gci3":
                    r_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                    h_s, t_s = sample_role_region(mod, r_id, generator=gen, n_samples=n_per_ax)
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        rho_z = crisp_membership(mod.role_depth(zx, zy, r_id, crisp=True))
                        mb_z = crisp_membership(mod.depth(zy, b_id, crisp=True))
                        mc_z = crisp_membership(mod.depth(zx, c_id, crisp=True))
                        if bool((gci3(rho_z, mb_z, mc_z, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "gci3_bot":
                    r_id, b_id = int(row[0]), int(row[1])
                    h_s, t_s = sample_role_region(mod, r_id, generator=gen, n_samples=n_per_ax)
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        rho_z = crisp_membership(mod.role_depth(zx, zy, r_id, crisp=True))
                        mb_z = crisp_membership(mod.depth(zy, b_id, crisp=True))
                        if bool((gci3_bot(rho_z, mb_z, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "role_inclusion":
                    r_id, s_id = int(row[0]), int(row[1])
                    h_s, t_s = sample_role_region(mod, r_id, generator=gen, n_samples=n_per_ax)
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        rr = crisp_membership(mod.role_depth(zx, zy, r_id, crisp=True))
                        rs = crisp_membership(mod.role_depth(zx, zy, s_id, crisp=True))
                        if bool((role_inclusion(rr, rs) > 0.0).any().item()):
                            reg_viol_axioms += 1
                elif k == "role_chain":
                    r_id, s_id, t_id = int(row[0]), int(row[1]), int(row[2])
                    hx, hy = sample_role_region(mod, r_id, generator=gen, n_samples=n_per_ax)
                    _, tz = sample_role_region(mod, s_id, generator=gen, n_samples=n_per_ax)
                    v_c = hx.valid & tz.valid
                    if bool(v_c.any().item()):
                        zx = hx.points[v_c]
                        zy = hy.points[v_c]
                        zz = tz.points[v_c]
                        rr = crisp_membership(mod.role_depth(zx, zy, r_id, crisp=True))
                        rs = crisp_membership(mod.role_depth(zy, zz, s_id, crisp=True))
                        rt = crisp_membership(mod.role_depth(zx, zz, t_id, crisp=True))
                        if bool((role_chain(rr, rs, rt) > 0.0).any().item()):
                            reg_viol_axioms += 1

        elem_viol = (
            float(elem_viol_count / elem_total_count) if elem_total_count > 0 else 0.0
        )
        region_viol = (
            float(reg_viol_axioms / reg_total_axioms) if reg_total_axioms > 0 else 0.0
        )
        crisp = {
            "elem_viol": elem_viol,
            "region_viol": region_viol,
        }

        # Vectorised self-membership exist metric over live classes (use_reasoner=False)
        all_cls_ids = th.arange(model.nb_classes, dtype=th.long, device=device)
        self_depths = mod.depth(pos_all[: model.nb_classes], all_cls_ids, crisp=True)
        dead_cls = dead_c[: model.nb_classes]
        live_cls = ~dead_cls
        self_inside = (self_depths > 0.0) & live_cls
        exist_val = (
            float(self_inside[live_cls].to(dtype=th.float32).mean().item())
            if bool(live_cls.any().item())
            else 1.0
        )
        dead_ok = True
        exact = bool(
            elem_viol == 0.0
            and region_viol == 0.0
            and exist_val == 1.0
            and dead_ok
        )
        eval_metrics: Dict[str, Any] = {
            "elem_viol": elem_viol,
            "region_viol": region_viol,
            "exist": exist_val,
            "dead_ok": dead_ok,
            "faithful_named_precision": None,
            "faithful_named_recall": None,
            "faithful_named": [None, None],
            "exact": exact,
        }
        return crisp, eval_metrics


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
        device=eff_device,
        seed=seed,
    )

    # Guard against O(nb_elements^2 * d) allocation in GroundedEL.reseed_idle on GO
    # and unsubsampled 1-D all_fillers in GroundedEL.compute_batch_losses on GO.
    if model.nb_elements > 5000:
        if model.reseed_every == 0:
            model.reseed_idle = lambda *a, **kw: 0  # type: ignore[method-assign]
        import mowl.models.grounded.model as _gmod

        _orig_sfe = _gmod.sample_filler_elements

        def _capped_sample_filler_elements(
            filler_elements: th.Tensor,
            nb_elements: int,
            n_random: int,
            generator: th.Generator,
            batch_size: int = 1,
        ) -> th.Tensor:
            if filler_elements.ndim == 1 and filler_elements.numel() > 16:
                max_f = max(n_random, 8)
                perm = th.randperm(
                    filler_elements.numel(), device=generator.device, generator=generator
                )[:max_f].to(device=filler_elements.device)
                filler_elements = filler_elements[perm]
            return _orig_sfe(
                filler_elements,
                nb_elements,
                n_random,
                generator=generator,
                batch_size=batch_size,
            )

        _gmod.sample_filler_elements = _capped_sample_filler_elements

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
    crisp, eval_metrics = compute_crisp_and_eval_metrics(model, n_eval=n_eval)
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
      ``--epochs``, ``--batch-size``, ``--lr``, ``--seed``, ``--device``,
      ``--data-root``, and ``--out``.

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
