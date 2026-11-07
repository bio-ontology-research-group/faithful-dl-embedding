#!/usr/bin/env python3
"""Profile the GroundedEL TBox step (``compute_batch_losses``) of the E3 model.

Usage::

    python -m experiments.deepgose.profile_tbox --ont cc --tbox-batch 1024 --device cuda \
        [--steps 10] [--data-root DIR] [--cache-dir DIR] [--out profile.json]

Reports wall time per TBox step and per data (BCE) step, the fraction of wall time
during which CUDA kernels run (GPU busy fraction, from the torch profiler), the number
of CUDA kernel launches and host-device syncs per step, and the top operators / Python
functions by time. A step is "GPU-bound" when the busy fraction is close to 1.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import torch as th

from experiments.deepgose.data import load_deepgose_data
from experiments.deepgose.model import build_deepgose_grounded_model


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        th.cuda.synchronize()


def _time_steps(fn, n: int, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ont", default="cc")
    ap.add_argument("--tbox-batch", type=int, default=1024)
    ap.add_argument("--protein-batch", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/data/u/hohndor/gfe/data/deepgo2/data"))
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--backward", action="store_true", help="include backward pass in the timed TBox step")
    args = ap.parse_args(argv)

    device = args.device if (not args.device.startswith("cuda") or th.cuda.is_available()) else "cpu"
    data = load_deepgose_data(args.data_root, args.ont, cache_dir=args.cache_dir)
    model = build_deepgose_grounded_model(
        tables=data.tables, embed_dim=args.dim, n_components=args.K, n_role_components=args.K,
        lambda_tbox=1.0, tbox_batch_size=args.tbox_batch, in_dim=int(data.train_features.shape[1]),
        seed=0, device=device,
    )
    model.module.train()
    g = model.grounded
    print(f"[profile] ont={args.ont} nb_classes={g.nb_classes} nb_elements={g.nb_elements} nb_rels={g.nb_rels} "
          f"tbox_batch={args.tbox_batch} d={args.dim} K={args.K} device={device}", flush=True)
    print("[profile] axiom counts:", {k: len(v) for k, v in g.training_datasets.items()}, flush=True)

    feats = data.train_features[: args.protein_batch].to(device)
    lbls = data.train_labels[: args.protein_batch].to(device)

    def tbox_step():
        losses = g.compute_batch_losses()
        if args.backward:
            losses["total"].backward()
        return losses

    def data_step():
        loss, _ = model.compute_data_loss(feats, lbls)
        if args.backward:
            loss.backward()
        return loss

    for _ in range(3):
        tbox_step(); data_step()
    t_tbox = _time_steps(tbox_step, args.steps, device)
    t_data = _time_steps(data_step, args.steps, device)
    print(f"[profile] wall per TBox step: {t_tbox*1000:.1f} ms   per data step ({args.protein_batch} proteins): {t_data*1000:.1f} ms"
          f"   ratio {t_tbox / max(t_data, 1e-9):.1f}x", flush=True)

    # torch profiler: GPU busy fraction and launch counts for the TBox step
    acts = [th.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(th.profiler.ProfilerActivity.CUDA)
    n_prof = 3
    with th.profiler.profile(activities=acts, record_shapes=False, with_stack=False) as prof:
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n_prof):
            tbox_step()
        _sync(device)
        wall = time.perf_counter() - t0
    ka = prof.key_averages()
    cuda_self_us = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka)
    n_kernels = sum(e.count for e in ka if getattr(e, "device_type", None) is not None and str(getattr(e, "device_type", "")).endswith("CUDA"))
    sync_names = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "Memcpy DtoH", "aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::unique")
    syncs = {n: sum(e.count for e in ka if e.key == n) // n_prof for n in sync_names}
    launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel")) // n_prof
    busy = cuda_self_us / 1e6 / wall if wall > 0 else float("nan")
    print(f"[profile] GPU busy fraction during TBox steps: {busy:.3f}   kernel launches/step: {launches}   "
          f"per-step sync-ish ops: {syncs}", flush=True)
    print("[profile] top ops by self device time:")
    print(ka.table(sort_by="self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total", row_limit=20))
    print("[profile] top ops by self CPU time:")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=20))

    # cProfile: Python-level hotspots
    pr = cProfile.Profile()
    _sync(device)
    pr.enable()
    for _ in range(3):
        tbox_step()
    _sync(device)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
    print("[profile] cProfile (cumulative):")
    print(s.getvalue())
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(20)
    print("[profile] cProfile (tottime):")
    print(s2.getvalue())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "ont": args.ont, "tbox_batch": args.tbox_batch, "device": device, "backward": args.backward,
            "ms_per_tbox_step": t_tbox * 1000, "ms_per_data_step": t_data * 1000,
            "gpu_busy_fraction": busy, "kernel_launches_per_step": launches, "syncs_per_step": syncs,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
