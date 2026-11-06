"""Encoder MLP and GroundedEL wiring for DeepGO-SE inductive function prediction (E3).

Builds the protein feature encoder ``2560 -> 1024 -> d`` with ``LayerNorm``,
``ReLU``, and ``Dropout(0.1)``, and wires it into :class:`~mowl.models.GroundedEL`
constructed from prebuilt ``go.norm`` tables via :meth:`GroundedEL.from_tables`
with ``dead_from_bot_only=True``, ``lambda_reg=1.0``, ``eps_min=1e-2``,
``n_adv=2``, and default region sampling.

Guarantees
----------
- Reasoner-free (R1): never imports or calls any OWL reasoner or
  :class:`~mowl.geometric.completion.CanonicalModel`.
- Exposes :func:`build_optimizer_groups` returning two Adam parameter groups:
  group 0 (encoder + ``raw_tau_s``) at ``lr_encoder`` and group 1 (class/role
  boxes and element positions/deadness) at ``lr_boxes``.

Non-guarantees
--------------
- Does not run an optimisation loop itself; see :mod:`experiments.deepgose.train`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

import jpype
import torch as th
import torch.nn as nn

if not jpype.isJVMStarted():
    os.environ.setdefault(
        "_JAVA_OPTIONS",
        "-Xms64m -XX:CompressedClassSpaceSize=64m -XX:ReservedCodeCacheSize=48m -XX:ParallelGCThreads=4 -XX:ConcGCThreads=2",
    )
    import mowl

    mowl.init_jvm(os.environ.get("MOWL_JVM_MEMORY", "1g"))

from mowl.models.grounded import GroundedEL

from experiments.deepgose.data import GONormTables


class ProteinEncoderMLP(nn.Module):
    """Two-layer MLP ``in_dim -> hidden_dim -> embed_dim`` with LayerNorm, ReLU, and Dropout.

    Default architecture: ``2560 -> 1024 -> d`` with ``LayerNorm(1024)``, ``ReLU``,
    and ``Dropout(0.1)``.

    Guarantees
    ----------
    - Maps a 2-D input tensor ``(B, in_dim)`` to ``(B, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        in_dim: int = 2560,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or embed_dim < 1:
            raise ValueError(
                f"Dimensions must be >= 1, got in_dim={in_dim}, hidden_dim={hidden_dim}, embed_dim={embed_dim}"
            )
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.dropout_p = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Map protein features ``(B, in_dim)`` to positions ``(B, embed_dim)``."""
        return self.net(x)


@dataclass
class DeepGOSEGroundedModel:
    """Container holding a table-backed :class:`~mowl.models.GroundedEL` model, term IDs, and ``lambda_tbox``."""

    grounded: GroundedEL
    encoder: ProteinEncoderMLP
    term_class_ids: th.Tensor
    lambda_tbox: float

    @property
    def module(self) -> nn.Module:
        """Return the underlying :class:`~mowl.nn.GroundedELModule`."""
        return self.grounded.module

    @property
    def tau_s(self) -> th.Tensor:
        """Return the learnable score temperature ``tau_s``."""
        return self.grounded.tau_s

    def compute_data_loss(
        self,
        features: th.Tensor,
        labels: th.Tensor,
        max_elements: int = 8_388_608,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute ``(bce_loss, scores)`` on a protein batch via :meth:`GroundedEL.compute_data_loss`."""
        return self.grounded.compute_data_loss(
            features, labels, self.term_class_ids, max_elements=max_elements
        )

    def score(
        self,
        features: th.Tensor,
        max_elements: int = 8_388_608,
    ) -> th.Tensor:
        """Compute raw predicted scores ``(B, n_terms)`` under :func:`torch.no_grad`."""
        return self.grounded.score(
            features, self.term_class_ids, max_elements=max_elements
        )


def build_deepgose_grounded_model(
    tables: GONormTables,
    embed_dim: int = 128,
    n_components: int = 1,
    n_role_components: Optional[int] = None,
    lambda_tbox: float = 1.0,
    tbox_batch_size: int = 4096,
    lr_boxes: float = 1e-2,
    in_dim: int = 2560,
    hidden_dim: int = 1024,
    dropout: float = 0.1,
    lambda_reg: float = 1.0,
    eps_min: float = 1e-2,
    n_adv: int = 2,
    n_region_samples: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> DeepGOSEGroundedModel:
    """Construct :class:`DeepGOSEGroundedModel` from ``tables`` with E3 default hyperparameters.

    Guarantees
    ----------
    - Uses ``dead_from_bot_only=True``, ``lambda_reg=1.0``, ``eps_min=1e-2``,
      ``n_adv=2``, and ``n_region_samples=4`` unless overridden.
    - Seeds PyTorch RNG with ``seed`` prior to constructing the encoder and
      :class:`~mowl.models.GroundedEL`.
    """
    if lambda_tbox < 0.0:
        raise ValueError(f"lambda_tbox must be >= 0, got {lambda_tbox}")
    eff_kr = int(n_components if n_role_components is None else n_role_components)
    th.manual_seed(int(seed))
    encoder = ProteinEncoderMLP(
        embed_dim=embed_dim,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    grounded = GroundedEL.from_tables(
        class_index_dict=tables.class_index_dict,
        object_property_index_dict=tables.object_property_index_dict,
        named_class_index_dict=tables.named_class_index_dict,
        gci0=tables.gci0,
        gci1=tables.gci1,
        gci2=tables.gci2,
        gci3=tables.gci3,
        gci0_bot=tables.gci0_bot,
        gci1_bot=tables.gci1_bot,
        gci3_bot=tables.gci3_bot,
        embed_dim=embed_dim,
        n_components=n_components,
        n_role_components=eff_kr,
        batch_size=tbox_batch_size,
        learning_rate=lr_boxes,
        lambda_reg=lambda_reg,
        eps_min=eps_min,
        eps_role=eps_min,
        eps_dead=1e-3,
        n_adv=n_adv,
        n_region_samples=n_region_samples,
        dead_from_bot_only=True,
        encoder=encoder,
        seed=seed,
        device=device,
    )
    term_ids = tables.term_class_ids.to(device=device, dtype=th.long)
    return DeepGOSEGroundedModel(
        grounded=grounded,
        encoder=encoder,
        term_class_ids=term_ids,
        lambda_tbox=float(lambda_tbox),
    )


def build_optimizer_groups(
    model: DeepGOSEGroundedModel,
    lr_encoder: float = 5e-4,
    lr_boxes: float = 1e-2,
) -> Tuple[th.optim.Adam, List[nn.Parameter], List[nn.Parameter]]:
    """Build two-group Adam optimizer: group 0 (encoder + ``raw_tau_s``) and group 1 (boxes/positions).

    Guarantees
    ----------
    - Every parameter in ``model.module.parameters()`` belongs to exactly one of
      the two parameter groups.
    """
    mod = model.grounded.module
    raw_tau_s = mod.ensure_tau_s()
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc_ids.add(id(raw_tau_s))

    enc_params: List[nn.Parameter] = []
    box_params: List[nn.Parameter] = []
    for p in mod.parameters():
        if id(p) in enc_ids:
            enc_params.append(p)
        else:
            box_params.append(p)

    optimizer = th.optim.Adam(
        [
            {"params": enc_params, "lr": float(lr_encoder), "name": "encoder"},
            {"params": box_params, "lr": float(lr_boxes), "name": "boxes"},
        ]
    )
    return optimizer, enc_params, box_params
