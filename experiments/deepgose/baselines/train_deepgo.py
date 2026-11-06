"""DeepGO-SE ``deepgozero_esm`` baseline training script with ``dgl`` dependency stripped.

Adapted from ``~/Public/software/deepgo2/train.py``:
- Removes the top-level ``dgl`` import and stubs ``dgl`` / ``dgl.nn`` so
  ``deepgo.models.DeepGOModel`` and ``deepgo.data.load_normal_forms`` import
  cleanly in a PyTorch-only virtual environment.
- Honours ``--data-root``, ``--out``, ``--seed``, ``--model-id``, ``--device``,
  and smoke subset flags ``--max-train-proteins`` and ``--min-term-annotations``.
- Writes ``{out}/{ont}/{model_name}_{model_id}.pt``,
  ``{out}/{ont}/test_predictions_{model_name}_{model_id}.pkl`` (propagated scores in
  ``test_df['preds']`` as in DeepGO-SE, plus raw scores in ``test_df['raw_preds']``),
  and ``{out}/{ont}/metrics_{model_name}_{model_id}.json`` (recording ``best_val_bce``)
  so :mod:`experiments.deepgose.evaluate` scores baselines identically to GroundedEL.

Guarantees
----------
- Reasoner-free (R1): never imports or calls any OWL reasoner or
  :class:`~mowl.geometric.completion.CanonicalModel`.
- Never writes temporary files under ``/tmp``.
"""

from __future__ import annotations

from functools import partial
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import random
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import click as ck
import numpy as np
import pandas as pd
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR

# NumPy 2.x compatibility for deepgo.metrics
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

# Stub dgl / dgl.nn so read-only deepgo/data.py and deepgo/models.py import without DGL installed
if "dgl" not in sys.modules:
    _dgl_stub = types.ModuleType("dgl")
    _dgl_nn_stub = types.ModuleType("dgl.nn")
    _dgl_nn_stub.GATConv = object  # type: ignore[attr-defined]
    _dgl_nn_stub.GraphConv = object  # type: ignore[attr-defined]
    _dgl_stub.nn = _dgl_nn_stub  # type: ignore[attr-defined]
    sys.modules["dgl"] = _dgl_stub
    sys.modules["dgl.nn"] = _dgl_nn_stub

DEEPGO2_DIR = Path(
    os.environ.get("DEEPGO2_DIR", "/home/leechuck/Public/software/deepgo2")
)
if DEEPGO2_DIR.is_dir() and str(DEEPGO2_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(DEEPGO2_DIR.resolve()))

from deepgo.data import get_data, load_data, load_normal_forms
from deepgo.metrics import compute_roc
from deepgo.models import DeepGOModel
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.utils import Ontology, propagate_annots


class _CallableCommand(ck.Command):
    """Click Command subclass allowing ``main(argv_list)`` or ``main(**kwargs)`` without ``SystemExit``."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], (list, tuple)):
            return self.main(args=[str(x) for x in args[0]], standalone_mode=False)
        if (
            kwargs
            and "args" not in kwargs
            and "standalone_mode" not in kwargs
            and "prog_name" not in kwargs
        ):
            cli_args: List[str] = []
            param_map = {p.name: p for p in self.params}
            for k, v in kwargs.items():
                if v is None:
                    continue
                p = param_map.get(k)
                opt = p.opts[0] if (p is not None and p.opts) else f"--{k.replace('_', '-')}"
                if isinstance(v, bool):
                    if v:
                        cli_args.append(opt)
                else:
                    cli_args.extend([opt, str(v)])
            return self.main(args=cli_args, standalone_mode=False)
        return super().__call__(*args, **kwargs)


def _load_baseline_data(
    data_root: str,
    ont: str,
    terms_file: str,
    features_length: Optional[int],
    features_column: str,
    test_data_file: str,
    max_train_proteins: Optional[int] = None,
    min_term_annotations: Optional[int] = None,
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    Tuple[th.Tensor, th.Tensor],
    Tuple[th.Tensor, th.Tensor],
    Tuple[th.Tensor, th.Tensor],
    pd.DataFrame,
]:
    """Load DeepGO-SE split data with optional smoke subsetting matching ``experiments.deepgose.data``."""
    terms_df = pd.read_pickle(terms_file)
    all_terms = [str(t) for t in np.asarray(terms_df["gos"]).flatten()]

    iprs_path = f"{data_root}/{ont}/interpros.pkl"
    if os.path.exists(iprs_path):
        ipr_df = pd.read_pickle(iprs_path)
        iprs = [str(x) for x in np.asarray(ipr_df["interpros"]).flatten()]
        iprs_dict = {v: k for k, v in enumerate(iprs)}
    else:
        iprs_dict = {}
    if features_column == "interpros":
        features_length = len(iprs_dict)

    train_df = pd.read_pickle(f"{data_root}/{ont}/train_data.pkl")
    valid_df = pd.read_pickle(f"{data_root}/{ont}/valid_data.pkl")
    test_df = pd.read_pickle(f"{data_root}/{ont}/{test_data_file}")

    if max_train_proteins is not None:
        if max_train_proteins < 1:
            raise ValueError(
                f"max_train_proteins must be >= 1, got {max_train_proteins}"
            )
        train_df = train_df.iloc[: int(max_train_proteins)].reset_index(drop=True)

    if min_term_annotations is not None and min_term_annotations > 0:
        counts: Dict[str, int] = {t: 0 for t in all_terms}
        for annots in train_df["prop_annotations"].values:
            for g in annots:
                if g in counts:
                    counts[g] += 1
        terms = [t for t in all_terms if counts[t] >= int(min_term_annotations)]
        if not terms:
            raise ValueError(
                f"No terms remain for ont={ont!r} with min_term_annotations={min_term_annotations}"
            )
    else:
        terms = all_terms

    terms_dict = {v: i for i, v in enumerate(terms)}
    print("Terms", len(terms))

    assert features_length is not None
    train_data = get_data(
        train_df, iprs_dict, terms_dict, features_length, features_column
    )
    valid_data = get_data(
        valid_df, iprs_dict, terms_dict, features_length, features_column
    )
    test_data = get_data(
        test_df, iprs_dict, terms_dict, features_length, features_column
    )
    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df


@ck.command(cls=_CallableCommand)
@ck.option(
    "--data-root",
    "-dr",
    default="/mnt/data1/gfe/data/deepgo2/data",
    help="Data root folder containing {mf,bp,cc}/, go.norm, go.obo.",
)
@ck.option(
    "--out",
    "-o",
    default="/mnt/data/u/hohndor/gfe/runs/deepgose",
    help="Output directory for checkpoints, predictions, and metrics.",
)
@ck.option(
    "--ont",
    "-ont",
    default="cc",
    type=ck.Choice(["mf", "bp", "cc"]),
    help="GO subontology.",
)
@ck.option(
    "--model-name",
    "-m",
    type=ck.Choice(
        ["deepgozero", "deepgozero_plus", "deepgozero_esm", "deepgozero_esm_plus"]
    ),
    default="deepgozero_esm",
    help="Prediction model name.",
)
@ck.option(
    "--model-id",
    "-mi",
    default=0,
    type=int,
    help="Model index / id.",
)
@ck.option(
    "--seed",
    "-s",
    default=0,
    type=int,
    help="Random seed.",
)
@ck.option(
    "--test-data-name",
    "-td",
    default="test",
    type=ck.Choice(["test", "nextprot", "valid"]),
    help="Test data set name.",
)
@ck.option(
    "--batch-size",
    "-bs",
    default=37,
    type=int,
    help="Batch size for training.",
)
@ck.option(
    "--epochs",
    "-ep",
    "-e",
    default=128,
    type=int,
    help="Training epochs.",
)
@ck.option(
    "--load",
    "-ld",
    is_flag=True,
    help="Load existing checkpoint instead of training.",
)
@ck.option(
    "--device",
    "-d",
    default="cuda:0",
    help="Device for training (falls back to cpu if cuda unavailable).",
)
@ck.option(
    "--max-train-proteins",
    default=None,
    type=int,
    help="Optional cap on training proteins (smoke/tests only).",
)
@ck.option(
    "--min-term-annotations",
    default=None,
    type=int,
    help="Optional minimum positive training annotations per term (smoke/tests only).",
)
def main(
    data_root: str,
    out: str,
    ont: str,
    model_name: str,
    model_id: Optional[int],
    seed: int,
    test_data_name: str,
    batch_size: int,
    epochs: int,
    load: bool,
    device: str,
    max_train_proteins: Optional[int] = None,
    min_term_annotations: Optional[int] = None,
) -> Dict[str, Any]:
    """Train and evaluate the ``DeepGOModel`` (``deepgozero_esm``) baseline on ``ont``."""
    eff_device = device
    if eff_device.startswith("cuda") and not th.cuda.is_available():
        eff_device = "cpu"

    eff_seed = int(seed)
    random.seed(eff_seed)
    np.random.seed(eff_seed)
    th.manual_seed(eff_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(eff_seed)

    base_model_name = model_name
    eff_model_id = 0 if model_id is None else int(model_id)
    run_model_name = f"{base_model_name}_{eff_model_id}"

    if base_model_name.find("plus") != -1:
        go_norm_file = f"{data_root}/go-plus.norm"
    else:
        go_norm_file = f"{data_root}/go.norm"
    if not os.path.exists(go_norm_file):
        go_norm_file = f"{data_root}/{ont}/go.norm"

    go_file = f"{data_root}/go.obo"
    if not os.path.exists(go_file):
        go_file = f"{data_root}/{ont}/go.obo"

    out_ont_dir = Path(out) / ont
    out_ont_dir.mkdir(parents=True, exist_ok=True)

    model_file = str(out_ont_dir / f"{run_model_name}.pt")
    terms_file = f"{data_root}/{ont}/terms.pkl"
    out_file = str(out_ont_dir / f"{test_data_name}_predictions_{run_model_name}.pkl")
    default_out_file = str(out_ont_dir / "test_predictions.pkl")
    metrics_file = str(out_ont_dir / f"metrics_{run_model_name}.json")
    default_metrics_file = str(out_ont_dir / "metrics.json")
    pf_file = str(out_ont_dir / f"valid_{run_model_name}.pf")

    # Load Gene Ontology and Normalized axioms
    go = Ontology(go_file, with_rels=True)

    # Load the datasets
    if base_model_name.find("esm") != -1:
        features_length: Optional[int] = 2560
        features_column = "esm2"
    else:
        features_length = None
        features_column = "interpros"

    test_data_file = f"{test_data_name}_data.pkl"
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = (
        _load_baseline_data(
            data_root=data_root,
            ont=ont,
            terms_file=terms_file,
            features_length=features_length,
            features_column=features_column,
            test_data_file=test_data_file,
            max_train_proteins=max_train_proteins,
            min_term_annotations=min_term_annotations,
        )
    )
    n_terms = len(terms_dict)
    terms_list = list(terms_dict.keys())
    if features_column == "interpros":
        features_length = len(iprs_dict)
    assert features_length is not None

    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    valid_labels_np = valid_labels.detach().cpu().numpy()
    test_labels_np = test_labels.detach().cpu().numpy()

    # Load normal forms
    nf1_raw, nf2_raw, nf3_raw, nf4_raw, relations, zero_classes = load_normal_forms(
        go_norm_file, terms_dict
    )
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    nf1 = th.LongTensor(nf1_raw).to(eff_device)
    nf2 = th.LongTensor(nf2_raw).to(eff_device)
    nf3 = th.LongTensor(nf3_raw).to(eff_device)
    nf4 = th.LongTensor(nf4_raw).to(eff_device)
    normal_forms = (nf1, nf2, nf3, nf4)

    # Create DataLoaders
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True
    )
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False
    )
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False
    )

    loss_func = nn.BCELoss()
    net = DeepGOModel(features_length, n_terms, n_zeros, n_rels, eff_device).to(
        eff_device
    )
    print(net)

    optimizer = th.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    best_loss = 10000.0
    best_epoch = 0
    if not load:
        print("Training the model")
        for epoch in range(epochs):
            net.train()
            train_loss = 0.0
            train_elloss = 0.0
            train_steps = max(1, int(math.ceil(len(train_labels) / batch_size)))
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(eff_device)
                batch_labels = batch_labels.to(eff_device)
                logits = net(batch_features)
                loss = F.binary_cross_entropy(logits, batch_labels)
                el_loss = net.el_loss(normal_forms)
                total_loss = loss + el_loss
                train_loss += loss.detach().item()
                train_elloss = el_loss.detach().item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            train_loss /= train_steps

            print("Validation")
            net.eval()
            with th.no_grad():
                valid_steps = max(1, int(math.ceil(len(valid_labels_np) / batch_size)))
                valid_loss = 0.0
                preds_val: List[np.ndarray] = []
                for batch_features, batch_labels in valid_loader:
                    batch_features = batch_features.to(eff_device)
                    batch_labels = batch_labels.to(eff_device)
                    logits = net(batch_features)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    valid_loss += batch_loss.detach().item()
                    preds_val.append(logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                preds_val_arr = (
                    np.concatenate(preds_val, axis=0)
                    if preds_val
                    else np.zeros_like(valid_labels_np)
                )
                roc_auc = (
                    float(compute_roc(valid_labels_np, preds_val_arr))
                    if np.unique(valid_labels_np).size > 1
                    else 0.0
                )
                print(
                    f"Epoch {epoch}: Loss - {train_loss}, EL Loss: {train_elloss}, "
                    f"Valid loss - {valid_loss}, AUC - {roc_auc}"
                )

            print("EL Loss", train_elloss)
            if valid_loss < best_loss:
                best_loss = float(valid_loss)
                best_epoch = epoch + 1
                print("Saving model")
                th.save(net.state_dict(), model_file)

            scheduler.step()

    # Loading best model
    print("Loading the best model")
    net.load_state_dict(
        th.load(model_file, map_location=eff_device, weights_only=False)
    )
    net.eval()

    with th.no_grad():
        valid_steps = max(1, int(math.ceil(len(valid_labels_np) / batch_size)))
        valid_loss = 0.0
        for batch_features, batch_labels in valid_loader:
            batch_features = batch_features.to(eff_device)
            batch_labels = batch_labels.to(eff_device)
            logits = net(batch_features)
            batch_loss = F.binary_cross_entropy(logits, batch_labels)
            valid_loss += batch_loss.detach().item()
        valid_loss /= valid_steps

    with th.no_grad():
        test_steps = max(1, int(math.ceil(len(test_labels_np) / batch_size)))
        test_loss = 0.0
        preds_list: List[np.ndarray] = []
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(eff_device)
            batch_labels = batch_labels.to(eff_device)
            logits = net(batch_features)
            batch_loss = F.binary_cross_entropy(logits, batch_labels)
            test_loss += batch_loss.detach().cpu().item()
            preds_list.append(logits.detach().cpu().numpy())
        test_loss /= test_steps
        preds_arr = (
            np.concatenate(preds_list, axis=0)
            if preds_list
            else np.zeros((0, n_terms), dtype=np.float32)
        )
        roc_auc = (
            float(compute_roc(test_labels_np, preds_arr))
            if np.unique(test_labels_np).size > 1
            else 0.0
        )
        print(
            f"Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}"
        )

    with open(pf_file, "w", encoding="utf-8") as f:
        f.write(
            f"Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}\n"
        )

    raw_preds = [row.copy() for row in preds_arr]
    for t_id in terms_dict:
        go.get_ancestors(t_id)
    if len(preds_arr) > 64 and len(terms_dict) > 256:
        with Pool(min(32, os.cpu_count() or 1)) as p:
            prop_preds = p.map(
                partial(propagate_annots, go=go, terms_dict=terms_dict),
                [row.copy() for row in preds_arr],
            )
    else:
        prop_preds = [
            propagate_annots(row.copy(), go=go, terms_dict=terms_dict)
            for row in preds_arr
        ]

    out_df = test_df.copy()
    out_df["raw_preds"] = raw_preds
    out_df["preds"] = prop_preds
    out_df.to_pickle(out_file)
    out_df.to_pickle(default_out_file)

    metrics_payload: Dict[str, Any] = {
        "ont": ont,
        "seed": eff_seed,
        "model_id": eff_model_id,
        "model_name": base_model_name,
        "best_val_bce": float(valid_loss if load else best_loss),
        "best_val_bce_epoch": int(best_epoch),
        "test_bce": float(test_loss),
        "test_roc_auc": float(roc_auc),
        "n_terms": int(n_terms),
        "terms": terms_list,
        "checkpoint_path": model_file,
        "predictions_path": out_file,
    }
    for m_target in (metrics_file, default_metrics_file):
        with open(m_target, "w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    return metrics_payload


if __name__ == "__main__":
    main()
