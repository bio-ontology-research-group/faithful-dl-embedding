"""Data loading and ``go.norm`` table builder for DeepGO-SE GroundedEL (E3).

Builds prebuilt normal-form tables directly from ``go.norm`` (no OWL/JVM round
trip), adds auxiliary classes ``Aux_G`` and relation ``hasFunction`` with axioms
:math:`\\text{Aux}_G \\sqsubseteq \\exists \\text{hasFunction}.G` (``gci2``) and
:math:`\\exists \\text{hasFunction}.G \\sqsubseteq \\text{Aux}_G` (``gci3``) for
every predicted term :math:`G`, caches the built tables as a ``.pt`` file under
the run directory, and converts DeepGO-SE pickle splits into ``(esm2, labels)``
float32 tensors.

Guarantees
----------
- Reasoner-free and JVM-free (R1): never imports or calls any OWL reasoner,
  OWLAPI, or :class:`~mowl.geometric.completion.CanonicalModel`.
- Replaces ``'_'`` with ``':'`` in all class and relation identifiers from
  ``go.norm``, matching DeepGO-SE's ``load_normal_forms``.
- Caches built normal-form tables atomically under ``cache_dir``.

Non-guarantees
--------------
- Assumes ``go.norm`` lines follow the four normal forms emitted by DeepGO-SE's
  Groovy ``Normalize.groovy`` script (``SubClassOf`` with optional ``and`` or ``some``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch as th

from mowl.geometric.normal_forms import BOTTOM, TOP

HAS_FUNCTION_REL = "hasFunction"
AUX_PREFIX = "Aux_"


def aux_class_name(go_id: str) -> str:
    """Return the auxiliary class identifier ``Aux_<go_id>`` for predicted term ``go_id``."""
    return f"{AUX_PREFIX}{go_id}"


@dataclass
class GONormTables:
    """Prebuilt normal-form index tables and term index mappings for ``GroundedEL.from_tables``.

    Attributes
    ----------
    class_index_dict : Dict[str, int]
        Bijective mapping from class names (``BOTTOM``, ``TOP``, all GO classes
        in ``go.norm`` / ``terms``, and ``Aux_G`` for every predicted term ``G``)
        to ``0 .. nb_classes - 1``.
    object_property_index_dict : Dict[str, int]
        Bijective mapping from relation names in ``go.norm`` plus ``hasFunction``
        to ``0 .. nb_rels - 1``.
    named_class_index_dict : Dict[str, int]
        Mapping of named GO classes (excluding ``TOP``, ``BOTTOM``, and ``Aux_G``)
        to their indices in ``class_index_dict``.
    gci0 : torch.Tensor
        Shape ``(N0, 2)``, ``torch.long``, rows ``[C, D]`` for :math:`C \\sqsubseteq D`.
    gci1 : torch.Tensor
        Shape ``(N1, 3)``, ``torch.long``, rows ``[C, D, E]`` for :math:`C \\sqcap D \\sqsubseteq E`.
    gci2 : torch.Tensor
        Shape ``(N2, 3)``, ``torch.long``, rows ``[C, R, D]`` for :math:`C \\sqsubseteq \\exists R.D`
        (including :math:`\\text{Aux}_G \\sqsubseteq \\exists \\text{hasFunction}.G`).
    gci3 : torch.Tensor
        Shape ``(N3, 3)``, ``torch.long``, rows ``[R, C, D]`` for :math:`\\exists R.C \\sqsubseteq D`
        (including :math:`\\exists \\text{hasFunction}.G \\sqsubseteq \\text{Aux}_G`).
    gci0_bot : torch.Tensor
        Empty ``(0, 2)`` ``torch.long`` tensor.
    gci1_bot : torch.Tensor
        Empty ``(0, 3)`` ``torch.long`` tensor.
    gci3_bot : torch.Tensor
        Empty ``(0, 3)`` ``torch.long`` tensor.
    terms : List[str]
        Ordered list of predicted GO term IDs (e.g. ``'GO:0005575'``).
    term_class_ids : torch.Tensor
        1-D ``torch.long`` tensor of shape ``(n_terms,)`` giving the ``Aux_G``
        class index for each term ``G`` in ``terms`` (used for scoring
        :math:`s(p, G) = \\sigma(\\operatorname{depth}(p, \\text{Aux}_G) / \\tau_s)`).
    go_term_class_ids : torch.Tensor
        1-D ``torch.long`` tensor of shape ``(n_terms,)`` giving the ``G`` class
        index for each term ``G`` in ``terms``.
    term_subclass_pairs : torch.Tensor
        2-D ``torch.long`` tensor of shape ``(M, 2)`` of ``(sub_term_idx, sup_term_idx)``
        column index pairs ``0 <= sub_term_idx, sup_term_idx < n_terms`` with
        ``sub_term_idx != sup_term_idx`` entailed by ``nf1`` (:math:`G \\sqsubseteq G'`).
    """

    class_index_dict: Dict[str, int]
    object_property_index_dict: Dict[str, int]
    named_class_index_dict: Dict[str, int]
    gci0: th.Tensor
    gci1: th.Tensor
    gci2: th.Tensor
    gci3: th.Tensor
    gci0_bot: th.Tensor
    gci1_bot: th.Tensor
    gci3_bot: th.Tensor
    terms: List[str]
    term_class_ids: th.Tensor
    go_term_class_ids: th.Tensor
    term_subclass_pairs: th.Tensor

    def to_dict(self) -> Dict[str, Any]:
        """Serialise into a plain dictionary suitable for :func:`torch.save`."""
        return {
            "class_index_dict": self.class_index_dict,
            "object_property_index_dict": self.object_property_index_dict,
            "named_class_index_dict": self.named_class_index_dict,
            "gci0": self.gci0,
            "gci1": self.gci1,
            "gci2": self.gci2,
            "gci3": self.gci3,
            "gci0_bot": self.gci0_bot,
            "gci1_bot": self.gci1_bot,
            "gci3_bot": self.gci3_bot,
            "terms": self.terms,
            "term_class_ids": self.term_class_ids,
            "go_term_class_ids": self.go_term_class_ids,
            "term_subclass_pairs": self.term_subclass_pairs,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GONormTables":
        """Reconstruct :class:`GONormTables` from a dictionary loaded via :func:`torch.load`."""
        return cls(
            class_index_dict=dict(d["class_index_dict"]),
            object_property_index_dict=dict(d["object_property_index_dict"]),
            named_class_index_dict=dict(d["named_class_index_dict"]),
            gci0=d["gci0"].to(dtype=th.long),
            gci1=d["gci1"].to(dtype=th.long),
            gci2=d["gci2"].to(dtype=th.long),
            gci3=d["gci3"].to(dtype=th.long),
            gci0_bot=d["gci0_bot"].to(dtype=th.long),
            gci1_bot=d["gci1_bot"].to(dtype=th.long),
            gci3_bot=d["gci3_bot"].to(dtype=th.long),
            terms=list(d["terms"]),
            term_class_ids=d["term_class_ids"].to(dtype=th.long),
            go_term_class_ids=d["go_term_class_ids"].to(dtype=th.long),
            term_subclass_pairs=d["term_subclass_pairs"].to(dtype=th.long),
        )


def parse_go_norm(
    norm_file: Union[str, Path],
    terms: Sequence[str],
    cache_path: Optional[Union[str, Path]] = None,
) -> GONormTables:
    """Parse ``go.norm`` and ``terms`` into :class:`GONormTables` with ``Aux_G`` axioms.

    Guarantees
    ----------
    - Replaces ``'_'`` with ``':'`` in all identifiers read from ``norm_file``.
    - Maps the four normal forms in ``go.norm`` to:
      * ``C SubClassOf D`` -> ``gci0`` ``[C, D]``
      * ``C and D SubClassOf E`` -> ``gci1`` ``[C, D, E]``
      * ``R some C SubClassOf D`` -> ``gci3`` ``[R, C, D]``
      * ``C SubClassOf R some D`` -> ``gci2`` ``[C, R, D]``
    - Appends one ``Aux_G`` class per term ``G`` in ``terms`` together with
      ``Aux_G SubClassOf hasFunction some G`` in ``gci2`` and
      ``hasFunction some G SubClassOf Aux_G`` in ``gci3``.
    - If ``cache_path`` is provided and exists with matching ``terms``, loads
      from ``cache_path``; otherwise writes ``cache_path`` atomically.

    Non-guarantees
    --------------
    - Raises :class:`ValueError` if a line in ``norm_file`` does not contain
      ``SubClassOf``.
    """
    terms_list = [str(t) for t in terms]
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.is_file():
            loaded = th.load(cp, map_location="cpu", weights_only=False)
            if list(loaded.get("terms", [])) == terms_list:
                return GONormTables.from_dict(loaded)

    norm_path = Path(norm_file)
    class_index_dict: Dict[str, int] = {BOTTOM: 0, TOP: 1}
    object_property_index_dict: Dict[str, int] = {}

    def _get_cls(c_id: str) -> int:
        if c_id not in class_index_dict:
            class_index_dict[c_id] = len(class_index_dict)
        return class_index_dict[c_id]

    def _get_rel(r_id: str) -> int:
        if r_id not in object_property_index_dict:
            object_property_index_dict[r_id] = len(object_property_index_dict)
        return object_property_index_dict[r_id]

    # Ensure every predicted GO term is registered as a GO class first
    for t_id in terms_list:
        _get_cls(t_id)

    gci0_rows: List[Tuple[int, int]] = []
    gci1_rows: List[Tuple[int, int, int]] = []
    gci2_rows: List[Tuple[int, int, int]] = []
    gci3_rows: List[Tuple[int, int, int]] = []

    with open(norm_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line_norm = line.replace("_", ":")
            if " SubClassOf " not in line_norm:
                continue
            left, right = line_norm.split(" SubClassOf ", 1)
            left = left.strip()
            right = right.strip()

            if " and " in left:
                # NF2: C and D SubClassOf E -> gci1 [C, D, E]
                c_str, d_str = left.split(" and ", 1)
                c_i = _get_cls(c_str.strip())
                d_i = _get_cls(d_str.strip())
                e_i = _get_cls(right)
                gci1_rows.append((c_i, d_i, e_i))
            elif " some " in left:
                # NF3: R some C SubClassOf D -> gci3 [R, C, D]
                r_str, c_str = left.split(" some ", 1)
                r_i = _get_rel(r_str.strip())
                c_i = _get_cls(c_str.strip())
                d_i = _get_cls(right)
                gci3_rows.append((r_i, c_i, d_i))
            elif " some " in right:
                # NF4: C SubClassOf R some D -> gci2 [C, R, D]
                r_str, d_str = right.split(" some ", 1)
                c_i = _get_cls(left)
                r_i = _get_rel(r_str.strip())
                d_i = _get_cls(d_str.strip())
                gci2_rows.append((c_i, r_i, d_i))
            else:
                # NF1: C SubClassOf D -> gci0 [C, D]
                c_i = _get_cls(left)
                d_i = _get_cls(right)
                gci0_rows.append((c_i, d_i))

    # Named GO classes (before adding Aux_G classes)
    named_class_index_dict: Dict[str, int] = {
        k: v for k, v in class_index_dict.items() if k not in (TOP, BOTTOM)
    }

    # Add hasFunction relation and Aux_G classes + axioms for each predicted term G
    has_func_idx = _get_rel(HAS_FUNCTION_REL)
    term_class_ids_list: List[int] = []
    go_term_class_ids_list: List[int] = []
    for t_id in terms_list:
        g_idx = _get_cls(t_id)
        aux_idx = _get_cls(aux_class_name(t_id))
        go_term_class_ids_list.append(g_idx)
        term_class_ids_list.append(aux_idx)
        # Aux_G <= exists hasFunction.G (gci2: [C, R, D])
        gci2_rows.append((aux_idx, has_func_idx, g_idx))
        # exists hasFunction.G <= Aux_G (gci3: [R, C, D])
        gci3_rows.append((has_func_idx, g_idx, aux_idx))

    # Build direct subclass pairs (sub_term_idx, sup_term_idx) among predicted terms from gci0_rows
    term_to_col: Dict[int, int] = {
        g_idx: col_i for col_i, g_idx in enumerate(go_term_class_ids_list)
    }
    sub_pairs_set: set[Tuple[int, int]] = set()
    for c_i, d_i in gci0_rows:
        if c_i in term_to_col and d_i in term_to_col and c_i != d_i:
            sub_pairs_set.add((term_to_col[c_i], term_to_col[d_i]))
    sub_pairs_sorted = sorted(sub_pairs_set)

    gci0_t = (
        th.tensor(gci0_rows, dtype=th.long)
        if gci0_rows
        else th.empty((0, 2), dtype=th.long)
    )
    gci1_t = (
        th.tensor(gci1_rows, dtype=th.long)
        if gci1_rows
        else th.empty((0, 3), dtype=th.long)
    )
    gci2_t = (
        th.tensor(gci2_rows, dtype=th.long)
        if gci2_rows
        else th.empty((0, 3), dtype=th.long)
    )
    gci3_t = (
        th.tensor(gci3_rows, dtype=th.long)
        if gci3_rows
        else th.empty((0, 3), dtype=th.long)
    )
    term_subclass_pairs_t = (
        th.tensor(sub_pairs_sorted, dtype=th.long)
        if sub_pairs_sorted
        else th.empty((0, 2), dtype=th.long)
    )

    tables = GONormTables(
        class_index_dict=class_index_dict,
        object_property_index_dict=object_property_index_dict,
        named_class_index_dict=named_class_index_dict,
        gci0=gci0_t,
        gci1=gci1_t,
        gci2=gci2_t,
        gci3=gci3_t,
        gci0_bot=th.empty((0, 2), dtype=th.long),
        gci1_bot=th.empty((0, 3), dtype=th.long),
        gci3_bot=th.empty((0, 3), dtype=th.long),
        terms=terms_list,
        term_class_ids=th.tensor(term_class_ids_list, dtype=th.long),
        go_term_class_ids=th.tensor(go_term_class_ids_list, dtype=th.long),
        term_subclass_pairs=term_subclass_pairs_t,
    )

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp_cp = cp.with_suffix(f".pt.tmp.{os.getpid()}")
        th.save(tables.to_dict(), tmp_cp)
        tmp_cp.replace(cp)

    return tables


def dataframe_to_tensors(
    df: pd.DataFrame,
    terms_dict: Mapping[str, int],
) -> Tuple[th.Tensor, th.Tensor]:
    """Convert a DeepGO-SE split DataFrame into ``(esm2_features, multi_hot_labels)`` ``float32`` tensors.

    Matches ``deepgo.data.load_normal_forms`` / ``get_data`` label extraction
    on ``df['esm2']`` and ``df['prop_annotations']``.

    Guarantees
    ----------
    - Returns ``features`` of shape ``(N, 2560)`` (or ``(N, feat_dim)``) and
      ``labels`` of shape ``(N, len(terms_dict))``, both ``torch.float32`` on CPU.
    """
    n_rows = len(df)
    n_terms = len(terms_dict)
    if n_rows == 0:
        return (
            th.empty((0, 2560), dtype=th.float32),
            th.empty((0, n_terms), dtype=th.float32),
        )
    esm_series = df["esm2"].values
    feat_np = np.stack([np.asarray(v, dtype=np.float32) for v in esm_series], axis=0)
    labels_np = np.zeros((n_rows, n_terms), dtype=np.float32)
    for i, annots in enumerate(df["prop_annotations"].values):
        for go_id in annots:
            if go_id in terms_dict:
                labels_np[i, terms_dict[go_id]] = 1.0
    return th.from_numpy(feat_np), th.from_numpy(labels_np)


@dataclass
class DeepGOSESplitData:
    """Loaded protein splits, terms, and prebuilt GO normal-form tables for one sub-ontology."""

    ont: str
    terms: List[str]
    terms_dict: Dict[str, int]
    train_features: th.Tensor
    train_labels: th.Tensor
    valid_features: th.Tensor
    valid_labels: th.Tensor
    test_features: th.Tensor
    test_labels: th.Tensor
    test_df: pd.DataFrame
    tables: GONormTables


def load_deepgose_data(
    data_root: Union[str, Path],
    ont: str,
    cache_dir: Optional[Union[str, Path]] = None,
    max_train_proteins: Optional[int] = None,
    min_term_annotations: Optional[int] = None,
) -> DeepGOSESplitData:
    """Load ``train/valid/test_data.pkl``, ``terms.pkl``, and ``go.norm`` for ``ont``.

    Supports optional smoke/test subsetting via ``max_train_proteins`` (slices
    ``train_df`` to the first ``max_train_proteins`` rows) and
    ``min_term_annotations`` (keeps only terms in ``terms.pkl`` with at least
    ``min_term_annotations`` positive ``prop_annotations`` in the training split).

    Guarantees
    ----------
    - Does not modify any file under ``data_root``.
    - Caches the built :class:`GONormTables` as a ``.pt`` file under ``cache_dir``
      when ``cache_dir`` is provided.
    """
    root = Path(data_root)
    ont_dir = root / ont
    terms_df = pd.read_pickle(ont_dir / "terms.pkl")
    all_terms: List[str] = [str(t) for t in np.asarray(terms_df["gos"]).flatten()]

    train_df: pd.DataFrame = pd.read_pickle(ont_dir / "train_data.pkl")
    valid_df: pd.DataFrame = pd.read_pickle(ont_dir / "valid_data.pkl")
    test_df: pd.DataFrame = pd.read_pickle(ont_dir / "test_data.pkl")

    if cache_dir is not None:
        ic_cache_path = Path(cache_dir) / f"ic_cache_{ont}.pt"
        if not ic_cache_path.is_file():
            obo_path = root / "go.obo"
            if not obo_path.is_file():
                obo_path = ont_dir / "go.obo"
            if obo_path.is_file():
                try:
                    import sys as _sys

                    _dg_dir = Path(
                        os.environ.get(
                            "DEEPGO2_DIR", "/home/leechuck/Public/software/deepgo2"
                        )
                    )
                    if _dg_dir.is_dir() and str(_dg_dir.resolve()) not in _sys.path:
                        _sys.path.insert(0, str(_dg_dir.resolve()))
                    from deepgo.utils import Ontology

                    _go = Ontology(str(obo_path), with_rels=True)
                    _annots = (
                        [set(a) for a in train_df["prop_annotations"].values]
                        + [set(a) for a in valid_df["prop_annotations"].values]
                        + [set(a) for a in test_df["prop_annotations"].values]
                    )
                    _go.calculate_ic(_annots)
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    _tmp_ic = ic_cache_path.with_suffix(f".pt.tmp.{os.getpid()}")
                    th.save(
                        {"ic": dict(_go.ic), "ic_norm": float(_go.ic_norm)}, _tmp_ic
                    )
                    _tmp_ic.replace(ic_cache_path)
                except Exception:
                    pass

    if max_train_proteins is not None:
        if max_train_proteins < 1:
            raise ValueError(f"max_train_proteins must be >= 1, got {max_train_proteins}")
        train_df = train_df.iloc[: int(max_train_proteins)].reset_index(drop=True)

    if min_term_annotations is not None and min_term_annotations > 0:
        counts: Dict[str, int] = {t: 0 for t in all_terms}
        for annots in train_df["prop_annotations"].values:
            for g in annots:
                if g in counts:
                    counts[g] += 1
        filtered_terms = [t for t in all_terms if counts[t] >= int(min_term_annotations)]
        if not filtered_terms:
            raise ValueError(
                f"No terms remain for ont={ont!r} with min_term_annotations={min_term_annotations}"
            )
        terms = filtered_terms
    else:
        terms = all_terms

    terms_dict = {t: i for i, t in enumerate(terms)}
    train_feat, train_lbl = dataframe_to_tensors(train_df, terms_dict)
    valid_feat, valid_lbl = dataframe_to_tensors(valid_df, terms_dict)
    test_feat, test_lbl = dataframe_to_tensors(test_df, terms_dict)

    norm_file = root / "go.norm"
    if not norm_file.is_file():
        norm_file = ont_dir / "go.norm"

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        terms_hash = hashlib.sha1("\n".join(terms).encode("utf-8")).hexdigest()[:10]
        cache_path = Path(cache_dir) / f"go_norm_tables_{ont}_{len(terms)}_{terms_hash}.pt"

    tables = parse_go_norm(norm_file, terms=terms, cache_path=cache_path)
    return DeepGOSESplitData(
        ont=ont,
        terms=terms,
        terms_dict=terms_dict,
        train_features=train_feat,
        train_labels=train_lbl,
        valid_features=valid_feat,
        valid_labels=valid_lbl,
        test_features=test_feat,
        test_labels=test_lbl,
        test_df=test_df,
        tables=tables,
    )
