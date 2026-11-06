"""DeepGO-SE ``mlp_esm`` baseline training script with ``dgl`` dependency stripped.

Adapted from ``~/Public/software/deepgo2/train_mlp.py`` and ``train.py``:
- Removes the top-level ``dgl`` import and stubs ``dgl`` / ``dgl.nn`` so
  ``deepgo.models.MLPModel`` and ``deepgo.data.load_data`` import cleanly in a
  PyTorch-only virtual environment.
- Uses the DeepGO-SE ESM training hyperparameters (Adam ``5e-4``,
  ``MultiStepLR(milestones=[5, 20], gamma=0.1)``, batch size 37, 128 epochs,
  checkpoint by validation BCE).
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
from typing import Any, Dict, List, Optional

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

from deepgo.metrics import compute_roc
from deepgo.models import MLPModel
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.utils import Ontology, propagate_annots

from experiments.deepgose.baselines.train_deepgo import (
    _CallableCommand,
    _load_baseline_data,
)


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
    default="mlp_esm",
    type=ck.Choice(["mlp", "mlp_esm"]),
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
    type=ck.Choice(["test", "nextprot"]),
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
    """Train and evaluate the ``MLPModel`` (``mlp_esm``) baseline on ``ont``."""
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

    go = Ontology(go_file, with_rels=True)
    loss_func = nn.BCELoss()

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

    net = MLPModel(features_length, n_terms, eff_device).to(eff_device)
    print(net)

    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data

    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True
    )
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False
    )
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False
    )

    valid_labels_np = valid_labels.detach().cpu().numpy()
    test_labels_np = test_labels.detach().cpu().numpy()

    optimizer = th.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    best_loss = 10000.0
    best_epoch = 0
    if not load:
        print("Training the model")
        for epoch in range(epochs):
            net.train()
            train_loss = 0.0
            train_steps = max(1, int(math.ceil(len(train_labels) / batch_size)))
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(eff_device)
                batch_labels = batch_labels.to(eff_device)
                logits = net(batch_features)
                loss = F.binary_cross_entropy(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()

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
                    f"Epoch {epoch}: Loss - {train_loss}, "
                    f"Valid loss - {valid_loss}, AUC - {roc_auc}"
                )
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
        print(f"Test Loss - {test_loss}, AUC - {roc_auc}")

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
