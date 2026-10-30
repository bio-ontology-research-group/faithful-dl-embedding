"""Tests for WP-A objective grid and WP-C prism method configurations (Task 6)."""
from __future__ import annotations

from experiments.config import (
    ABOX_DATASETS,
    LR_GRID,
    METHODS,
    OBJECTIVE_DATASETS,
    SYNTHETIC_DATASETS,
    methods_for_dataset,
)

EXPECTED_ORIGINAL_METHODS = {
    "elbe",
    "elbe-init",
    "elbe-anchor",
    "box2el",
    "box2el-init",
    "box2el-anchor",
    "box2el-milp",
    "multibox-k1",
    "multibox-k2",
    "multibox-k3",
    "multibox-k4",
    "multibox-k2-init",
    "canonical-mixture",
}

EXPECTED_OBJECTIVE_IDS = [
    f"{backbone}-k{k}-{obj}"
    for backbone in ("multibox", "multielbe")
    for k in (1, 2, 3)
    for obj in ("axiom", "soft", "softdiv", "point", "mixed")
]


def test_objective_datasets_and_lr_grid() -> None:
    """Verify OBJECTIVE_DATASETS equals the four synthetic datasets and LR_GRID is [0.001, 0.01]."""
    assert OBJECTIVE_DATASETS == SYNTHETIC_DATASETS == ["tk3", "tk4", "tk5", "tk6"]
    assert LR_GRID == [0.001, 0.01]


def test_existing_methods_untouched() -> None:
    """Verify all pre-existing method ids and their specifications remain intact."""
    for m_id in EXPECTED_ORIGINAL_METHODS:
        assert m_id in METHODS, f"Missing original method {m_id!r}"

    assert METHODS["elbe"]["model"] == "ELBE"
    assert METHODS["box2el"]["model"] == "BoxSquaredEL"
    assert METHODS["box2el-milp"]["datasets"] == ["pizza"]
    assert METHODS["multibox-k2"]["model"] == "MultiBoxEL"
    assert METHODS["multibox-k2"]["n_components"] == 2
    assert METHODS["canonical-mixture"]["model"] == "CanonicalMixture"
    assert METHODS["canonical-mixture"]["datasets"] == ["yeast-function", "helis"]


def test_objective_method_id_scheme_and_kwargs() -> None:
    """Verify all 30 programmatic objective method ids and their exact kwargs."""
    assert len(EXPECTED_OBJECTIVE_IDS) == 30
    expected_obj_kwargs = {
        "axiom": {},
        "soft": {"component_aggregation": "soft", "tau_anneal": True},
        "softdiv": {
            "component_aggregation": "soft",
            "tau_anneal": True,
            "diversity_reg": 0.1,
        },
        "point": {"objective": "point"},
        "mixed": {"objective": "mixed"},
    }

    for backbone, model_name in (("multibox", "MultiBoxEL"), ("multielbe", "MultiELBE")):
        for k in (1, 2, 3):
            for obj_name, kw in expected_obj_kwargs.items():
                m_id = f"{backbone}-k{k}-{obj_name}"
                assert m_id in METHODS, f"Missing objective method {m_id!r}"
                spec = METHODS[m_id]
                assert spec["model"] == model_name
                assert spec["n_components"] == k
                assert spec["init"] is None
                assert spec["anchor_weight"] == 0.0
                assert spec["objective_kwargs"] == kw
                assert spec["objective"] == kw.get("objective", "axiom")
                assert spec["component_aggregation"] == kw.get(
                    "component_aggregation", "hard"
                )
                assert spec["tau_anneal"] == kw.get("tau_anneal", False)
                assert spec["diversity_reg"] == kw.get("diversity_reg", 0.0)


def test_prism_methods_config() -> None:
    """Verify prism and prism-init method entries in config.METHODS."""
    assert "prism" in METHODS
    assert "prism-init" in METHODS

    p_spec = METHODS["prism"]
    assert p_spec["model"] == "PrismEL"
    assert p_spec["n_components"] == 1
    assert p_spec["init"] is None
    assert p_spec["anchor_weight"] == 0.0

    pi_spec = METHODS["prism-init"]
    assert pi_spec["model"] == "PrismEL"
    assert pi_spec["n_components"] == 1
    assert pi_spec["init"] == "projected"
    assert pi_spec["anchor_weight"] == 0.0


def test_methods_for_dataset_includes_objectives_and_prisms_where_allowed() -> None:
    """Verify objective ids and prism/prism-init are allowed on synthetic, pizza, and abox datasets only."""
    new_ids = set(EXPECTED_OBJECTIVE_IDS) | {"prism", "prism-init"}
    assert len(new_ids) == 32

    for ds in OBJECTIVE_DATASETS + ["pizza"] + ABOX_DATASETS:
        ds_methods = set(methods_for_dataset(ds))
        assert new_ids <= ds_methods, f"Missing new methods for dataset {ds!r}"

    for ds in ("go", "foodon"):
        ds_methods = set(methods_for_dataset(ds))
        assert ds_methods.isdisjoint(new_ids), (
            f"Objective/prism methods should not be in {ds!r}"
        )
