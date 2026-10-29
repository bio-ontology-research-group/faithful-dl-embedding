"""Post-hoc evaluation of model-ness at the training margin (Task 10).

CLI usage
---------
::

    python -m experiments.posthoc --raw experiments/results/raw --ckpt ~/tmp/gfe-runs [--only-missing]

Guarantees
----------
- For every raw JSON ``<raw>/<run>.json`` that has a matching checkpoint
  ``<ckpt>/<run>.pt``, rebuilds the model on CPU via :func:`experiments.run._build_model`
  (reusing the dataset bundle and :class:`~mowl.geometric.CanonicalModel` across runs of
  the same dataset), loads the saved ``state_dict``, and evaluates
  :func:`experiments.metrics.modelness` at the run's training margin (``config.margin``,
  tolerance ``1e-6``).
- Writes ``metrics.modelness_train_margin``, ``metrics.modelness_train_margin_<gci>``,
  and ``metrics.posthoc = {"version": 1, "git": <hash>}`` additively into the JSON.
- Records ``config.margin_semantics = "relaxing"`` for ``BoxSquaredEL`` / ``MultiBoxEL``
  and ``"tightening"`` for ``ELBE``.
- Rewrites each JSON atomically via a temporary file and ``os.replace``.
- Idempotent and runs entirely on CPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from experiments.cm_cache import get_canonical_model
from experiments.config import (
    DEFAULT_HYPERPARAMS,
    METHODS,
    RAW_RESULTS_DIR,
    REPO_ROOT,
    RUNS_DIR,
)
from experiments.datasets import DatasetBundle, load_dataset
from experiments.metrics import modelness
from experiments.run import _build_model, _git_rev_parse

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric import CanonicalModel

POSTHOC_VERSION = 1


def margin_semantics_for_model(model_cls_name: str) -> str:
    """Return ``'relaxing'`` for Box²EL/MultiBoxEL and ``'tightening'`` for ELBE."""
    if model_cls_name in ("BoxSquaredEL", "MultiBoxEL"):
        return "relaxing"
    if model_cls_name == "ELBE":
        return "tightening"
    raise ValueError(
        f"Unsupported model class {model_cls_name!r} for margin_semantics."
    )


def compute_modelness_train_margin(
    model: Any,
    *,
    train_margin: float,
    tol: float = 1e-6,
    batch_size: int = 32768,
) -> Dict[str, float]:
    """Evaluate :func:`experiments.metrics.modelness` at ``train_margin`` and map keys to ``modelness_train_margin*``."""
    raw = modelness(
        model,
        tol=tol,
        batch_size=batch_size,
        margin=float(train_margin),
    )
    out: Dict[str, float] = {
        "modelness_train_margin": float(raw["modelness_all"]),
    }
    for k, v in raw.items():
        if k == "modelness_all" or not k.startswith("modelness_"):
            continue
        gci_name = k[len("modelness_") :]
        out[f"modelness_train_margin_{gci_name}"] = float(v)
    return out


def _is_already_complete(payload: Dict[str, Any]) -> bool:
    """Return ``True`` iff ``payload`` already contains all Task 10 post-hoc fields."""
    cfg = payload.get("config", {})
    m = payload.get("metrics", {})
    if "margin_semantics" not in cfg:
        return False
    if "modelness_train_margin" not in m:
        return False
    ph = m.get("posthoc")
    if not isinstance(ph, dict) or ph.get("version") != POSTHOC_VERSION:
        return False
    for k in m:
        if k.startswith("modelness_") and k != "modelness_all" and not k.startswith("modelness_train_margin"):
            gci_name = k[len("modelness_") :]
            if f"modelness_train_margin_{gci_name}" not in m:
                return False
    return True


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a sibling temporary file and ``os.replace``."""
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except BaseException:  # pragma: no cover
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def process_raw_directory(
    raw_dir: Path = RAW_RESULTS_DIR,
    ckpt_dir: Path = RUNS_DIR,
    *,
    only_missing: bool = False,
    min_age_seconds: float = 0.0,
) -> Dict[str, int]:
    """Run the post-hoc pass over ``raw_dir`` using checkpoints in ``ckpt_dir`` on CPU.

    Returns a summary dictionary with keys ``updated``, ``skipped_missing_ckpt``,
    ``skipped_existing``, ``skipped_young``, and ``total_json``.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()

    if not raw_dir.exists():
        return {
            "updated": 0,
            "skipped_missing_ckpt": 0,
            "skipped_existing": 0,
            "skipped_young": 0,
            "total_json": 0,
        }

    json_paths = sorted(raw_dir.glob("*.json"))
    now = time.time()
    git_hash = _git_rev_parse(REPO_ROOT)

    by_dataset: Dict[str, List[Tuple[Path, Path, Dict[str, Any]]]] = defaultdict(list)
    skipped_missing_ckpt = 0
    skipped_existing = 0
    skipped_young = 0

    for jp in json_paths:
        if min_age_seconds > 0.0:
            age_s = now - jp.stat().st_mtime
            if age_s < min_age_seconds:
                skipped_young += 1
                continue

        ckpt_path = ckpt_dir / f"{jp.stem}.pt"
        if not ckpt_path.exists():
            skipped_missing_ckpt += 1
            continue

        payload = json.loads(jp.read_text(encoding="utf-8"))
        if only_missing and _is_already_complete(payload):
            skipped_existing += 1
            continue

        ds_name = str(payload["config"]["dataset"])
        by_dataset[ds_name].append((jp, ckpt_path, payload))

    updated = 0
    for ds_name, items in by_dataset.items():
        bundle: DatasetBundle = load_dataset(ds_name)
        cm: "CanonicalModel" = get_canonical_model(bundle)

        for jp, ckpt_path, payload in items:
            cfg = payload["config"]
            method_id = str(cfg["method"])
            spec = METHODS.get(method_id, {})
            model_cls_name = str(cfg.get("model") or spec.get("model"))
            dim = int(cfg["dim"])
            seed = int(cfg["seed"])
            eff_bs = int(
                cfg.get(
                    "effective_batch_size",
                    cfg.get("batch_size", DEFAULT_HYPERPARAMS["batch_size"]),
                )
            )
            lr = float(
                cfg.get("learning_rate", DEFAULT_HYPERPARAMS["learning_rate"])
            )
            train_margin = float(
                cfg.get("margin", DEFAULT_HYPERPARAMS["margin"])
            )
            delta = float(cfg.get("delta", DEFAULT_HYPERPARAMS["delta"]))
            reg_factor = float(
                cfg.get("reg_factor", DEFAULT_HYPERPARAMS["reg_factor"])
            )
            num_negs = int(
                cfg.get("num_negs", DEFAULT_HYPERPARAMS["num_negs"])
            )

            model, _ = _build_model(
                bundle=bundle,
                cm=cm,
                method_id=method_id,
                dim=dim,
                seed=seed,
                batch_size=eff_bs,
                learning_rate=lr,
                margin=train_margin,
                delta=delta,
                reg_factor=reg_factor,
                num_negs=num_negs,
                device="cpu",
                checkpoint_path=ckpt_path,
            )

            state_dict = torch.load(
                ckpt_path, map_location="cpu", weights_only=True
            )
            model.module.load_state_dict(state_dict)
            model.module.to("cpu")
            model.module.eval()

            m_train = compute_modelness_train_margin(
                model,
                train_margin=train_margin,
                tol=1e-6,
                batch_size=eff_bs,
            )
            del model

            cfg["margin_semantics"] = margin_semantics_for_model(model_cls_name)
            metrics_dict = payload.setdefault("metrics", {})
            metrics_dict.update(m_train)
            existing_ph = metrics_dict.get("posthoc")
            if (
                isinstance(existing_ph, dict)
                and existing_ph.get("version") == POSTHOC_VERSION
                and isinstance(existing_ph.get("git"), str)
                and existing_ph.get("git")
            ):
                ph_git = existing_ph["git"]
            else:
                ph_git = git_hash
            metrics_dict["posthoc"] = {
                "version": POSTHOC_VERSION,
                "git": ph_git,
            }

            _atomic_write_json(jp, payload)
            updated += 1

    return {
        "updated": updated,
        "skipped_missing_ckpt": skipped_missing_ckpt,
        "skipped_existing": skipped_existing,
        "skipped_young": skipped_young,
        "total_json": len(json_paths),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``experiments/posthoc.py``."""
    parser = argparse.ArgumentParser(
        description="Compute post-hoc modelness_train_margin and margin_semantics on raw JSON results."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_RESULTS_DIR,
        help="Directory containing raw JSON run files (default: experiments/results/raw).",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing <run>.pt checkpoint files (default: ~/tmp/gfe-runs).",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip JSON files that already contain post-hoc modelness_train_margin fields.",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=0.0,
        help="Skip JSON files modified more recently than this many seconds ago (default: 0.0).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``experiments/posthoc.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = process_raw_directory(
        raw_dir=args.raw,
        ckpt_dir=args.ckpt,
        only_missing=bool(args.only_missing),
        min_age_seconds=float(args.min_age_seconds),
    )
    print(
        f"[posthoc] updated={summary['updated']} "
        f"skipped_missing_ckpt={summary['skipped_missing_ckpt']} "
        f"skipped_existing={summary['skipped_existing']} "
        f"skipped_young={summary['skipped_young']} "
        f"total_json={summary['total_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
