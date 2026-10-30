"""Sequential sweep orchestrator for learnable geometric embedding experiments (Task 8, Spec §5.5).

CLI usage
---------
::

    python -m experiments.sweep --suite {synthetic,pizza,real,all} [--smoke] [--dry-run] [--out ...]

Guarantees
----------
- ``enumerate_cells(suite, smoke=False)`` enumerates ``(dataset, method, dim, seed)``
  tuples in deterministic order across ``methods_for_dataset(dataset)`` (11 methods
  for ``tk3..tk6``, ``go``, ``foodon``; 12 methods including ``box2el-milp`` for
  ``pizza``), suite datasets, ``DATASET_DIMS[dataset]``, and
  ``DATASET_SEEDS[dataset]`` (or ``[0]`` when ``smoke=True``).
- Skips any cell whose output JSON ``{dataset}__{method}__d{dim}__s{seed}.json``
  already exists in ``--out`` (unless ``--force`` is passed).
- Reuses the loaded :class:`~experiments.datasets.DatasetBundle` and cached
  :class:`~mowl.geometric.CanonicalModel` across all methods/dims/seeds of the
  same dataset so large ontologies (GO, FoodOn) are loaded and saturated/reloaded
  only once per sweep.
- Logs progress to stdout and to ``experiments/logs/sweep_<suite>[_smoke].log``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import traceback
from typing import Dict, List, Optional, Sequence

import torch

from experiments.cm_cache import get_canonical_model
from experiments.config import (
    ABOX_DATASETS,
    DATASETS,
    DATASET_DIMS,
    DATASET_EPOCHS,
    DATASET_SEEDS,
    DEFAULT_HYPERPARAMS,
    LOGS_DIR,
    LR_GRID,
    METHODS,
    OBJECTIVE_DATASETS,
    OBJECTIVE_METHODS,
    PRISM_METHODS,
    RAW_RESULTS_DIR,
    RUNS_DIR,
    SMOKE_RESULTS_DIR,
    SYNTHETIC_DATASETS,
    methods_for_dataset,
)
from experiments.datasets import load_dataset
from experiments.run import execute_run, run_filename

SUITE_DATASETS: Dict[str, List[str]] = {
    "synthetic": list(SYNTHETIC_DATASETS),
    "objectives": list(OBJECTIVE_DATASETS),
    "pizza": ["pizza"],
    "real": ["go", "foodon"],
    "abox": list(ABOX_DATASETS),
    "all": list(DATASETS),
}


@dataclass(frozen=True)
class SweepCell:
    """One ``(dataset, method, dim, seed, epochs, lr)`` experiment cell."""

    dataset: str
    method: str
    dim: int
    seed: int
    epochs: int
    lr: float = float(DEFAULT_HYPERPARAMS["learning_rate"])

    @property
    def filename(self) -> str:
        """Canonical JSON filename for this cell."""
        return run_filename(self.dataset, self.method, self.dim, self.seed, lr=self.lr)


def enumerate_cells(
    suite: str = "all",
    *,
    smoke: bool = False,
    epochs_override: Optional[int] = None,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    dims: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    lrs: Optional[Sequence[float]] = None,
) -> List[SweepCell]:
    """Enumerate all ``SweepCell`` entries for ``suite`` in deterministic order."""
    if datasets is None:
        if suite not in SUITE_DATASETS:
            raise ValueError(
                f"Unknown suite {suite!r}; expected one of {list(SUITE_DATASETS)}"
            )
        ds_list = list(SUITE_DATASETS[suite])
    else:
        ds_list = list(datasets)

    if lrs is not None:
        lr_list = [float(x) for x in lrs]
    elif suite == "objectives":
        lr_list = [float(DEFAULT_HYPERPARAMS["learning_rate"])] if smoke else [float(x) for x in LR_GRID]
    else:
        lr_list = [float(DEFAULT_HYPERPARAMS["learning_rate"])]

    cells: List[SweepCell] = []
    for ds_name in ds_list:
        ds_dims = [int(d) for d in dims] if dims is not None else DATASET_DIMS[ds_name]
        ds_seeds = (
            [int(s) for s in seeds]
            if seeds is not None
            else ([0] if smoke else DATASET_SEEDS[ds_name])
        )
        epochs = (
            int(epochs_override)
            if epochs_override is not None
            else (3 if smoke else int(DATASET_EPOCHS[ds_name]))
        )
        if methods is not None:
            allowed_for_ds = set(methods_for_dataset(ds_name))
            ds_methods = []
            for m_id in methods:
                if m_id not in METHODS:
                    raise ValueError(
                        f"Unknown method {m_id!r}; expected one of {list(METHODS)}"
                    )
                if m_id not in allowed_for_ds:
                    raise ValueError(
                        f"Method {m_id!r} is not allowed on dataset {ds_name!r}; "
                        f"allowed methods for {ds_name!r}: {sorted(allowed_for_ds)}"
                    )
                ds_methods.append(m_id)
        elif suite == "objectives":
            ds_methods = [
                m for m in (OBJECTIVE_METHODS + PRISM_METHODS) if m in methods_for_dataset(ds_name)
            ]
        else:
            obj_prism_set = set(OBJECTIVE_METHODS) | set(PRISM_METHODS)
            ds_methods = [
                m for m in methods_for_dataset(ds_name) if m not in obj_prism_set
            ]

        for lr_val in lr_list:
            for dim in ds_dims:
                for seed in ds_seeds:
                    for method_id in ds_methods:
                        cells.append(
                            SweepCell(
                                dataset=ds_name,
                                method=method_id,
                                dim=int(dim),
                                seed=int(seed),
                                epochs=epochs,
                                lr=float(lr_val),
                            )
                        )
    return cells


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``experiments/sweep.py``."""
    parser = argparse.ArgumentParser(
        description="Run or dry-run a sweep of learnable geometric embedding experiments."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=list(SUITE_DATASETS.keys()),
        help="Experiment suite to run (default: all).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Explicit dataset list override.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Explicit method list override.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=None,
        help="Explicit embedding dimension list override.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list override.",
    )
    parser.add_argument(
        "--lr",
        "--lrs",
        dest="lrs",
        type=float,
        nargs="+",
        default=None,
        help="Explicit learning rate list override.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 3 epochs, seed 0 only, output to results/smoke/ by default.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional explicit epoch count override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all planned cells and their skip/pending status without running.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells even if their output JSON already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cuda' or 'cpu'; defaults to 'cuda' if available else 'cpu').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for JSON outputs (default: results/smoke/ if --smoke else results/raw/).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory for model checkpoints (default: ~/tmp/gfe-runs/).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOGS_DIR,
        help="Directory for sweep log files (default: experiments/logs/).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``experiments/sweep.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(
        args.out
        if args.out is not None
        else (SMOKE_RESULTS_DIR if args.smoke else RAW_RESULTS_DIR)
    ).expanduser().resolve()

    cells = enumerate_cells(
        args.suite,
        smoke=bool(args.smoke),
        epochs_override=args.epochs,
        datasets=args.datasets,
        methods=args.methods,
        dims=args.dims,
        seeds=args.seeds,
        lrs=args.lrs,
    )

    if args.dry_run:
        n_existing = sum(
            1 for c in cells if (out_dir / c.filename).exists() and not args.force
        )
        print(
            f"[sweep --dry-run] suite={args.suite} smoke={args.smoke} "
            f"total_cells={len(cells)} existing={n_existing} pending={len(cells) - n_existing} "
            f"out={out_dir}"
        )
        for idx, c in enumerate(cells, start=1):
            status = (
                "SKIP"
                if (out_dir / c.filename).exists() and not args.force
                else "RUN"
            )
            print(
                f"  [{idx:3d}/{len(cells):3d}] {status:4s} {c.filename} (epochs={c.epochs})"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    log_path = log_dir / f"sweep_{args.suite}{suffix}.log"

    resolved_device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    def _log(msg: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    _log(
        f"Starting sweep suite={args.suite} smoke={args.smoke} total_cells={len(cells)} "
        f"device={resolved_device} out={out_dir}"
    )

    # Group execution by dataset so each dataset bundle and CanonicalModel are loaded once
    current_ds_name: Optional[str] = None
    current_bundle = None
    current_cm = None

    n_ran = 0
    n_skipped = 0
    n_failed = 0
    t_sweep0 = time.monotonic()

    oom_types = tuple(
        t
        for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(t, type) and issubclass(t, BaseException)
    )

    def _is_oom(exc: BaseException) -> bool:
        if oom_types and isinstance(exc, oom_types):
            return True
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            return True
        return False

    for idx, cell in enumerate(cells, start=1):
        target_json = out_dir / cell.filename
        if target_json.exists() and not args.force:
            n_skipped += 1
            _log(f"[{idx:3d}/{len(cells):3d}] SKIP {cell.filename} (exists)")
            continue

        t_cell0 = time.monotonic()
        eff_bs = int(DEFAULT_HYPERPARAMS["batch_size"])
        oom_retries = 0
        max_oom_retries = 3
        payload = None

        while True:
            try:
                if cell.dataset != current_ds_name:
                    if current_bundle is not None:
                        setattr(current_bundle, "_subsumption_evaluator", None)
                        setattr(current_bundle, "_normalized_gcis_cache", None)
                        setattr(current_bundle, "_exact_box_init_cache", None)
                    current_bundle = None
                    current_cm = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    t_ds0 = time.monotonic()
                    current_bundle = load_dataset(cell.dataset)
                    current_cm = get_canonical_model(current_bundle)
                    current_ds_name = cell.dataset
                    _log(
                        f"Loaded dataset={cell.dataset} and CanonicalModel "
                        f"({len(current_cm.names)} names) in {time.monotonic() - t_ds0:.2f}s"
                    )

                payload = execute_run(
                    dataset=cell.dataset,
                    method=cell.method,
                    dim=cell.dim,
                    seed=cell.seed,
                    epochs=cell.epochs,
                    batch_size=eff_bs,
                    learning_rate=cell.lr,
                    lr=cell.lr,
                    device=resolved_device,
                    out_dir=out_dir,
                    runs_dir=args.runs_dir,
                    bundle=current_bundle,
                    cm=current_cm,
                )
                break
            except Exception as exc:
                torch.cuda.empty_cache()
                if _is_oom(exc) and oom_retries < max_oom_retries and eff_bs > 1:
                    oom_retries += 1
                    eff_bs = max(1, eff_bs // 2)
                    first_line = (str(exc).splitlines() or [""])[0]
                    _log(
                        f"[{idx:3d}/{len(cells):3d}] OOM RETRY {oom_retries}/{max_oom_retries} "
                        f"{cell.filename} | eff_bs={eff_bs} | {type(exc).__name__}: {first_line}"
                    )
                    continue

                n_failed += 1
                first_line = (str(exc).splitlines() or [""])[0]
                tb_str = traceback.format_exc()
                _log(
                    f"[{idx:3d}/{len(cells):3d}] FAILED {cell.filename} | "
                    f"{type(exc).__name__}: {first_line}"
                )
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(tb_str if tb_str.endswith("\n") else tb_str + "\n")

                failed_path = out_dir / f"{cell.filename}.failed.json"
                failed_payload = {
                    "dataset": cell.dataset,
                    "method": cell.method,
                    "dim": cell.dim,
                    "seed": cell.seed,
                    "epochs": cell.epochs,
                    "lr": cell.lr,
                    "effective_batch_size": eff_bs,
                    "oom_retries": oom_retries,
                    "exception_type": type(exc).__name__,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "message": str(exc),
                    "traceback": tb_str,
                }
                failed_path.write_text(
                    json.dumps(failed_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                break

        if payload is None:
            continue

        failed_marker = out_dir / f"{cell.filename}.failed.json"
        if failed_marker.exists():
            failed_marker.unlink()

        n_ran += 1
        m = payload["metrics"]
        cfg = payload["config"]
        elapsed = time.monotonic() - t_cell0
        hvr_val = m.get("helly_violation_rate")
        hvr_str = f"{hvr_val:.4f}" if hvr_val is not None else "null"
        fmrr_val = m.get("rank_f_mrr")
        fmrr_str = f"{fmrr_val:.4f}" if fmrr_val is not None else "null"
        _log(
            f"[{idx:3d}/{len(cells):3d}] DONE {cell.filename} | "
            f"wall={elapsed:.2f}s | train={payload['timings']['total_train_time_s']:.2f}s | "
            f"modelness={m.get('modelness_all', 0.0):.4f} | "
            f"hvr={hvr_str} | "
            f"f_mrr={fmrr_str} | "
            f"gpu_mb={m.get('peak_gpu_memory_mb', 0.0):.1f} | "
            f"eff_bs={cfg['effective_batch_size']}"
        )

    total_elapsed = time.monotonic() - t_sweep0
    _log(
        f"Completed sweep suite={args.suite} smoke={args.smoke}: "
        f"ran={n_ran} skipped={n_skipped} failed={n_failed} total={len(cells)} in {total_elapsed:.2f}s"
    )
    if n_ran == 0 and n_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
