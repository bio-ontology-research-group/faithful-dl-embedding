"""DeepGO-SE ``mlp_esm`` baseline training script with ``dgl`` dependency stripped.

Adapted from ``~/Public/software/deepgo2/train_mlp.py``:
- Removes the ``dgl`` dependency so ``deepgo.base.MLPModel`` and
  ``deepgo.data.load_data`` import cleanly in a PyTorch-only virtual environment.
- Honours ``--data-root``, ``--out``, and ``--seed``.
- Writes ``{out}/{ont}/test_predictions_{model_name}_{model_id}.pkl`` (raw scores
  in ``test_df['preds']``) and ``{out}/{ont}/metrics_{model_name}_{model_id}.json``
  (recording ``best_val_bce``) so :mod:`experiments.deepgose.evaluate` scores
  baselines identically to GroundedEL.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import types

import click as ck
import numpy as np
import pandas as pd
import torch as th

# NumPy 2.x compatibility for deepgo.metrics
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

# Stub dgl / dgl.nn so read-only deepgo/data.py imports without DGL installed
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

from deepgo.base import MLPModel
from deepgo.data import load_data
from deepgo.metrics import compute_metrics
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
    type=ck.Choice(["test", "next_prot", "af2_test"]),
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
    test_data_name: str,
    batch_size: int,
    epochs: int,
    load: bool,
    device: str,
) -> None:
    """Train and evaluate the ``mlp_esm`` baseline on ``ont``."""
    eff_device = device
    if eff_device.startswith("cuda") and not th.cuda.is_available():
        eff_device = "cpu"

    th.manual_seed(int(seed))
    np.random.seed(int(seed))

    go_file = f"{data_root}/go.obo"
    if not os.path.exists(go_file):
        go_file = f"{data_root}/{ont}/go.obo"

    out_ont_dir = Path(out) / ont
    out_ont_dir.mkdir(parents=True, exist_ok=True)

    model_file = str(out_ont_dir / f"{model_name}_{model_id}.th")
    terms_file = f"{data_root}/{ont}/terms.pkl"
    out_file = str(out_ont_dir / f"test_predictions_{model_name}_{model_id}.pkl")
    default_out_file = str(out_ont_dir / "test_predictions.pkl")
    metrics_file = str(out_ont_dir / f"metrics_{model_name}_{model_id}.json")

    go = Ontology(go_file, with_rels=True)
    loss_func = th.nn.BCELoss()

    test_data_file = f"{test_data_name}_data.pkl"
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root,
        ont,
        terms_file,
        test_data_file=test_data_file,
        use_esm=model_name.endswith("esm"),
    )
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)

    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data

    valid_steps = int(math.ceil(len(valid_labels) / batch_size)) if len(valid_labels) > 0 else 1
    test_steps = int(math.ceil(len(test_labels) / batch_size)) if len(test_labels) > 0 else 1

    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True
    )
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False
    )
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False
    )

    in_length = 2560 if model_name.endswith("esm") else n_iprs
    net = MLPModel(in_length, n_terms, eff_device).to(eff_device)

    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)

    best_loss = 10000.0
    best_epoch = 0
    if not load:
        print("Training the model")
        for epoch in range(epochs):
            net.train()
            train_loss = 0.0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(eff_device)
                batch_labels = batch_labels.to(eff_device)
                logits = net(batch_features)
                loss = loss_func(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()

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

            print(f"Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}")

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


if __name__ == "__main__":
    main()
