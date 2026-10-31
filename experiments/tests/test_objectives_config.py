"""Tests for WP-A objective grid and WP-C prism method configurations (Task 6)."""
from __future__ import annotations

from experiments.config import (
    ABOX_DATASETS,
    LR_GRID,
    METHODS,
    OBJECTIVE_DATASETS,
    OBJECTIVE_METHODS,
    PRISM_METHODS,
    SYNTHETIC_DATASETS,
    methods_for_dataset,
)
from experiments import sweep

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
    for obj in ("axiom", "soft", "softdiv", "point", "mixed", "hybrid")
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
    assert "elem" in METHODS
    assert METHODS["elem"]["model"] == "ELEm"
    assert METHODS["elem"]["n_components"] == 1
    assert METHODS["elem"]["init"] is None
    assert METHODS["elem"]["anchor_weight"] == 0.0
    for ds in ("pizza", "go", "foodon", "yeast-function", "helis"):
        assert "elem" in methods_for_dataset(ds)
    for ds in SYNTHETIC_DATASETS:
        assert "elem" not in methods_for_dataset(ds)
    abox_cells = sweep.enumerate_cells("abox", smoke=True)
    assert "elem" in {c.method for c in abox_cells}
    syn_cells = sweep.enumerate_cells("synthetic", smoke=True)
    assert "elem" not in {c.method for c in syn_cells}


def test_objective_method_id_scheme_and_kwargs() -> None:
    """Verify all 36 programmatic objective method ids and their exact kwargs."""
    assert len(EXPECTED_OBJECTIVE_IDS) == 36
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
        "hybrid": {"objective": "hybrid"},
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
    assert len(new_ids) == 38

    for ds in OBJECTIVE_DATASETS + ["pizza"] + ABOX_DATASETS:
        ds_methods = set(methods_for_dataset(ds))
        assert new_ids <= ds_methods, f"Missing new methods for dataset {ds!r}"

    for ds in ("go", "foodon"):
        ds_methods = set(methods_for_dataset(ds))
        assert ds_methods.isdisjoint(new_ids), (
            f"Objective/prism methods should not be in {ds!r}"
        )


def test_suites_exclude_new_objective_and_prism_ids_except_objectives_suite() -> None:
    """Verify synthetic/pizza/all suites do NOT include the 38 new ids, while objectives suite owns them."""
    new_ids = set(OBJECTIVE_METHODS) | set(PRISM_METHODS)
    assert len(new_ids) == 38

    for old_suite in ("synthetic", "pizza", "real", "abox", "all"):
        cells = sweep.enumerate_cells(old_suite, smoke=True)
        suite_methods = {c.method for c in cells}
        assert suite_methods.isdisjoint(new_ids), (
            f"Suite {old_suite!r} must not include objective/prism ids, got {suite_methods & new_ids}"
        )

    obj_cells = sweep.enumerate_cells("objectives", smoke=True)
    obj_methods = {c.method for c in obj_cells}
    assert obj_methods == new_ids


def test_structural_methods_config() -> None:
    """Verify STRUCTURAL_BASE_METHODS, STRUCTURAL_EXTRA_METHODS, and STRUCTURAL_METHODS ids, kwargs, K values, dataset restrictions, and suite disjointness."""
    from experiments.config import (
        STRUCTURAL_BASE_METHODS,
        STRUCTURAL_EXTRA_METHODS,
        STRUCTURAL_METHODS,
    )

    expected_base_order = [
        "multibox-k2-struct",
        "multibox-k4-struct",
        "multibox-k8-struct",
        "multibox-k2-struct-reseed",
        "multibox-k4-struct-reseed",
        "multibox-k4-struct-nodef",
        "multibox-k4-struct-sr0",
        "prism-struct",
    ]
    expected_extra_order = [
        "multibox-k3-struct-K27-reseed",
        "multibox-k4-struct-K64",
        "multibox-k4-struct-K64-reseed",
        "multibox-k4-struct-ew10",
        "prism-struct-ew10",
        "prism-struct-sr1e-2",
        "multibox-k3-struct-K81-reseed",
        "multibox-k2-struct-K32-reseed",
    ]
    assert STRUCTURAL_BASE_METHODS == expected_base_order
    assert STRUCTURAL_EXTRA_METHODS == expected_extra_order
    assert STRUCTURAL_METHODS == expected_base_order + expected_extra_order

    default_ds = list(SYNTHETIC_DATASETS) + ["pizza"]
    expected_specs = {
        "multibox-k2-struct": (
            "MultiBoxEL",
            2,
            {"objective": "structural", "max_components": 16},
            default_ds,
        ),
        "multibox-k4-struct": (
            "MultiBoxEL",
            4,
            {"objective": "structural", "max_components": 16},
            default_ds,
        ),
        "multibox-k8-struct": (
            "MultiBoxEL",
            8,
            {"objective": "structural", "max_components": 64},
            default_ds,
        ),
        "multibox-k2-struct-reseed": (
            "MultiBoxEL",
            2,
            {"objective": "structural", "max_components": 16, "reseed_every": 50},
            default_ds,
        ),
        "multibox-k4-struct-reseed": (
            "MultiBoxEL",
            4,
            {"objective": "structural", "max_components": 16, "reseed_every": 50},
            default_ds,
        ),
        "multibox-k4-struct-nodef": (
            "MultiBoxEL",
            4,
            {
                "objective": "structural",
                "max_components": 16,
                "structural_definitions": False,
            },
            default_ds,
        ),
        "multibox-k4-struct-sr0": (
            "MultiBoxEL",
            4,
            {"objective": "structural", "max_components": 16, "size_reg": 0.0},
            default_ds,
        ),
        "prism-struct": (
            "PrismEL",
            1,
            {"objective": "structural"},
            default_ds,
        ),
        "multibox-k3-struct-K27-reseed": (
            "MultiBoxEL",
            3,
            {"objective": "structural", "max_components": 27, "reseed_every": 50},
            default_ds,
        ),
        "multibox-k4-struct-K64": (
            "MultiBoxEL",
            4,
            {"objective": "structural", "max_components": 64},
            default_ds,
        ),
        "multibox-k4-struct-K64-reseed": (
            "MultiBoxEL",
            4,
            {"objective": "structural", "max_components": 64, "reseed_every": 50},
            default_ds,
        ),
        "multibox-k4-struct-ew10": (
            "MultiBoxEL",
            4,
            {"objective": "structural", "max_components": 16, "existence_weight": 10.0},
            default_ds,
        ),
        "prism-struct-ew10": (
            "PrismEL",
            1,
            {"objective": "structural", "existence_weight": 10.0},
            default_ds,
        ),
        "prism-struct-sr1e-2": (
            "PrismEL",
            1,
            {"objective": "structural", "size_reg": 0.01},
            default_ds,
        ),
        "multibox-k3-struct-K81-reseed": (
            "MultiBoxEL",
            3,
            {"objective": "structural", "max_components": 81, "reseed_every": 50},
            ["tk4", "tk5", "pizza"],
        ),
        "multibox-k2-struct-K32-reseed": (
            "MultiBoxEL",
            2,
            {"objective": "structural", "max_components": 32, "reseed_every": 50},
            ["tk4", "tk5", "tk6", "pizza"],
        ),
    }

    for m_id, (model_cls, k_val, obj_kw, exp_ds) in expected_specs.items():
        assert m_id in METHODS
        spec = METHODS[m_id]
        assert spec["model"] == model_cls
        assert spec["n_components"] == k_val
        assert spec["init"] is None
        assert spec["anchor_weight"] == 0.0
        assert spec["objective"] == "structural"
        assert spec["objective_kwargs"] == obj_kw
        assert spec["datasets"] == exp_ds

    for ds in ("tk4", "tk5", "pizza"):
        assert set(STRUCTURAL_METHODS) <= set(methods_for_dataset(ds))
    assert "multibox-k3-struct-K81-reseed" not in methods_for_dataset("tk3")
    assert "multibox-k3-struct-K81-reseed" not in methods_for_dataset("tk6")
    assert "multibox-k2-struct-K32-reseed" not in methods_for_dataset("tk3")
    assert "multibox-k2-struct-K32-reseed" in methods_for_dataset("tk6")
    for ds in ("go", "foodon", "yeast-function", "helis"):
        assert set(STRUCTURAL_METHODS).isdisjoint(set(methods_for_dataset(ds)))

    struct_base_set = set(STRUCTURAL_BASE_METHODS)
    struct_extra_set = set(STRUCTURAL_EXTRA_METHODS)
    struct_all_set = set(STRUCTURAL_METHODS)
    assert struct_base_set.isdisjoint(struct_extra_set)

    for other_suite in ("synthetic", "objectives", "pizza", "real", "abox", "all"):
        cells = sweep.enumerate_cells(other_suite, smoke=True)
        suite_methods = {c.method for c in cells}
        assert suite_methods.isdisjoint(struct_all_set), (
            f"Suite {other_suite!r} must not include structural methods"
        )

    struct_cells = sweep.enumerate_cells("structural", smoke=False)
    assert {c.method for c in struct_cells} == struct_base_set
    assert {c.method for c in struct_cells}.isdisjoint(struct_extra_set)

    struct_extra_cells = sweep.enumerate_cells("structural-extra", smoke=False)
    assert {c.method for c in struct_extra_cells} == struct_extra_set
    assert {c.method for c in struct_extra_cells}.isdisjoint(struct_base_set)


