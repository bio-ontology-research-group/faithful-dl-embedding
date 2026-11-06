"""DeepGO-SE ``deepgozero_esm`` baseline training script with ``dgl`` dependency stripped.

Adapted from ``~/Public/software/deepgo2/train.py``:
- Removes the top-level ``dgl`` import and stubs ``dgl`` / ``dgl.nn`` so
  ``deepgo.models.DEEPGOZeroModel`` and ``deepgo.data.load_normal_forms`` import
  cleanly in a PyTorch-only virtual environment.
- Honours ``--data-root``, ``--out``, and ``--seed``.
- Writes ``{out}/{ont}/test_predictions_{model_name}_{model_id}.pkl`` (raw scores
  in ``test_df['preds']``) and ``{out}/{ont}/metrics_{model_name}_{model_id}.json``
  (recording ``best_val_bce``) so :mod:`experiments.deepgose.evaluate` scores
  baselines identically to GroundedEL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types
from typing import Optional

import click as ck
import numpy as np
import pandas as pd
import torch as th
from torch.optim.lr_scheduler import MultiStepLR

# NumPy 2.x compatibility for deepgo.metrics
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

# Stub dgl / dgl.nn so read-only deepgo/data.py and deepgo/models.py import without DGL installed
if "dgl" not in sys.modules:
    _dgl_stub = types.ModuleType("dgl")
    _dgl_nn_stub = types.ModuleType("dgl.nn")
    _dgl_nn_stub.GATConv = object  # type: ignore[attr-defined]
    _dgl_stub.nn = _dgl_nn_stub  # type: ignore[attr-defined]
    sys.modules["dgl"] = _dgl_stub
    sys.modules["dgl.nn"] = _dgl_nn_stub

DEEPGO2_DIR = Path(
    os.environ.get("DEEPGO2_DIR", "/home/leechuck/Public/software/deepgo2")
)
if DEEPGO2_DIR.is_dir() and str(DEEPGO2_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(DEEPGO2_DIR.resolve()))

from deepgo.data import load_data, load_normal_forms
from deepgo.metrics import compute_metrics
from deepgo.models import DEEPGOZeroModel
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.utils import Ontology, propagate_annots


@ck.command()
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
    "--batch-size",
    "-bs",
    default=37,
    type=int,
    help="Batch size for training.",
)
@ck.option(
    "--epochs",
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
    default="cuda",
    help="Device for training (falls back to cpu if cuda unavailable).",
)
def main(
    data_root: str,
    out: str,
    ont: str,
    model_name: str,
    model_id: int,
    seed: int,
    batch_size: int,
    epochs: int,
    load: bool,
    device: str,
) -> None:
    """Train and evaluate the ``deepgozero_esm`` baseline on ``ont``."""
    if "gat" in model_name:
        raise ValueError(
            f"GAT models require dgl; use deepgozero_esm instead (got {model_name!r})"
        )
    eff_device = device
    if eff_device.startswith("cuda") and not th.cuda.is_available():
        eff_device = "cpu"

    th.manual_seed(int(seed))
    np.random.seed(int(seed))

    go_file = f"{data_root}/go.obo"
    if not os.path.exists(go_file):
        go_file = f"{data_root}/{ont}/go.obo"
    norm_file = f"{data_root}/go.norm"
    if not os.path.exists(norm_file):
        norm_file = f"{data_root}/{ont}/go.norm"

    out_ont_dir = Path(out) / ont
    out_ont_dir.mkdir(parents=True, exist_ok=True)

    model_file = str(out_ont_dir / f"{model_name}_{model_id}.th")
    terms_file = f"{data_root}/{ont}/terms.pkl"
    out_file = str(out_ont_dir / f"test_predictions_{model_name}_{model_id}.pkl")
    default_out_file = str(out_ont_dir / "test_predictions.pkl")
    metrics_file = str(out_ont_dir / f"metrics_{model_name}_{model_id}.json")

    go = Ontology(go_file, with_rels=True)
    loss_func = th.nn.BCELoss()

    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file
    )
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)

    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        norm_file, terms_dict
    )
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    normal_forms = (nf1, nf2, nf3, nf4)
    valid_steps = int(math.ceil(len(valid_data[0]) / batch_size)) if len(valid_data[0]) > 0 else 1
    test_steps = int(math.ceil(len(test_data[0]) / batch_size)) if len(test_data[0]) > 0 else 1

    net = DEEPGOZeroModel(
        n_iprs, n_terms, n_zeros, n_rels, device=eff_device, model_name=model_name
    ).to(eff_device)

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
            train_elloss = 0.0
            lmbda = 0.1
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(eff_device)
                batch_labels = batch_labels.to(eff_device)
                logits = net(batch_features)
                loss = loss_func(logits, batch_labels)
                el_loss = net.el_loss(normal_forms)
                total_loss = loss + el_loss
                train_loss += loss.detach().item()
                train_elloss = el_loss.detach().item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            train_loss /= max(train_steps, 1)
            print("Validation")
            net.eval()
            with th.no_grad():
                valid_loss = 0.0
                for batch_features, batch_labels in valid_loader:
                    batch_features = batch_features.to(eff_device)
                    batch_labels = batch_labels.to(eff_device)
                    logits = net(batch_features)
                    loss = loss_func(logits, batch_labels)
                    valid_loss += loss.detach().item()
                valid_loss /= max(valid_steps, 1)
                if valid_loss < best_loss:
                    best_loss = float(valid_loss)
                    best_epoch = epoch + 1
                    print("Saving model")
                    th.save(net.state_dict(), model_file)

            scheduler.step()
            print(
                f"Epoch {epoch}: Loss - {train_loss}, EL Loss: {train_elloss}, Valid loss - {valid_loss}"
            )

    print("Loading the best model")
    net.load_state_dict(th.load(model_file, map_location=eff_device, weights_only=False))
    net.eval()
    with th.no_grad():
        valid_loss = 0.0
        for batch_features, batch_labels in valid_loader:
            batch_features = batch_features.to(eff_device)
            batch_labels = batch_labels.to(eff_device)
            logits = net(batch_features)
            loss = loss_func(logits, batch_labels)
            valid_loss += loss.detach().item()
        valid_loss /= max(valid_steps, 1)

        preds = []
        test_loss = 0.0
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(eff_device)
            batch_labels = batch_labels.to(eff_device)
            logits = net(batch_features)
            batch_loss = loss_func(logits, batch_labels)
            test_loss += batch_loss.detach().cpu().item()
            preds = np.append(preds, logits.detach().cpu().numpy())
        test_loss /= max(test_steps, 1)
        preds = preds.reshape(-1, n_terms)

    # Save raw predictions in test_df['preds'] so evaluate.py can evaluate both raw and propagated
    out_df = test_df.copy()
    out_df["preds"] = list(preds)
    out_df.to_pickle(out_file)
    out_df.to_pickle(default_out_file)

    metrics_payload = {
        "ont": ont,
        "seed": int(seed),
        "model_id": int(model_id),
        "model_name": model_name,
        "best_val_bce": float(valid_loss if load else best_loss),
        "best_val_bce_epoch": int(best_epoch),
        "test_bce": float(test_loss),
        "predictions_path": out_file,
    }
    with open(metrics_file, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


import math  # noqa: E402

if __name__ == "__main__":
    main()
