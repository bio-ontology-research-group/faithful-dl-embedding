"""Tests for Task 10 post-hoc training-margin model-ness pass (experiments/posthoc.py)."""
from __future__ import annotations

import json
from pathlib import Path

import mowl
mowl.init_jvm("8g")

import pytest

from experiments import aggregate, posthoc, run

SCRATCH_DIR = Path("~/tmp/gfe-runs/test-scratch").expanduser().resolve()


@pytest.fixture
def scratch_dir(request: pytest.FixtureRequest) -> Path:
    """Return a clean per-test scratch directory under ~/tmp/gfe-runs/test-scratch/."""
    p = SCRATCH_DIR / request.node.name
    if p.exists():
        for child in sorted(p.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_posthoc_tk3_box2el_adds_keys_idempotent_and_monotone(
    scratch_dir: Path,
) -> None:
    """On a 3-epoch tk3 Box²EL run (CPU), posthoc adds keys, is idempotent, and satisfies modelness_train_margin >= modelness."""
    raw_dir = scratch_dir / "raw"
    ckpt_dir = scratch_dir / "ckpt"

    rc_run = run.main(
        [
            "--dataset",
            "tk3",
            "--method",
            "box2el-init",
            "--dim",
            "2",
            "--seed",
            "0",
            "--epochs",
            "3",
            "--device",
            "cpu",
            "--out",
            str(raw_dir),
            "--runs-dir",
            str(ckpt_dir),
        ]
    )
    assert rc_run in (0, None)

    json_path = raw_dir / "tk3__box2el-init__d2__s0.json"
    ckpt_path = ckpt_dir / "tk3__box2el-init__d2__s0.pt"
    assert json_path.exists()
    assert ckpt_path.exists()

    before = json.loads(json_path.read_text(encoding="utf-8"))
    assert "modelness_train_margin" not in before["metrics"]
    assert "margin_semantics" not in before["config"]

    rc_ph1 = posthoc.main(["--raw", str(raw_dir), "--ckpt", str(ckpt_dir)])
    assert rc_ph1 in (0, None)

    after_text_1 = json_path.read_text(encoding="utf-8")
    after_1 = json.loads(after_text_1)

    # Additive preservation of existing keys
    for k, v in before["metrics"].items():
        assert after_1["metrics"][k] == v

    assert after_1["config"]["margin_semantics"] == "relaxing"
    m = after_1["metrics"]
    assert "modelness_train_margin" in m
    assert "modelness_train_margin_gci0" in m
    assert "modelness_train_margin_gci1" in m
    assert "modelness_train_margin_gci1_bot" in m
    assert "posthoc" in m
    assert m["posthoc"]["version"] == 1
    assert isinstance(m["posthoc"]["git"], str) and len(m["posthoc"]["git"]) >= 7

    for key in (
        "modelness_train_margin",
        "modelness_train_margin_gci0",
        "modelness_train_margin_gci1",
        "modelness_train_margin_gci1_bot",
    ):
        assert 0.0 <= m[key] <= 1.0

    # Relaxed margin (gamma = 0.02 >= 0) can only satisfy more axioms than strict margin 0
    assert m["modelness_train_margin"] >= m["modelness_all"]
    assert m["modelness_train_margin_gci0"] >= m["modelness_gci0"]
    assert m["modelness_train_margin_gci1"] >= m["modelness_gci1"]
    assert m["modelness_train_margin_gci1_bot"] >= m["modelness_gci1_bot"]

    # Idempotency check: second pass produces byte-identical JSON
    rc_ph2 = posthoc.main(["--raw", str(raw_dir), "--ckpt", str(ckpt_dir)])
    assert rc_ph2 in (0, None)
    after_text_2 = json_path.read_text(encoding="utf-8")
    assert after_text_2 == after_text_1

    # --only-missing skips already-populated JSON
    summary_only_missing = posthoc.process_raw_directory(
        raw_dir=raw_dir, ckpt_dir=ckpt_dir, only_missing=True
    )
    assert summary_only_missing["updated"] == 0
    assert summary_only_missing["skipped_existing"] == 1
    assert json_path.read_text(encoding="utf-8") == after_text_1

    # Aggregate table includes Mod@γ column next to Modelness
    md = aggregate.render_dataset_markdown("tk3", [after_1])
    assert "| Modelness | Mod@γ |" in md
    md_missing = aggregate.render_dataset_markdown("tk3", [before])
    assert "| Modelness | Mod@γ |" in md_missing
    assert "| — |" in md_missing


def test_posthoc_tk3_elbe_tightening_margin(scratch_dir: Path) -> None:
    """Verify ELBE records margin_semantics='tightening' and modelness_train_margin <= modelness_all."""
    raw_dir = scratch_dir / "raw"
    ckpt_dir = scratch_dir / "ckpt"

    run.main(
        [
            "--dataset",
            "tk3",
            "--method",
            "elbe",
            "--dim",
            "2",
            "--seed",
            "0",
            "--epochs",
            "3",
            "--device",
            "cpu",
            "--out",
            str(raw_dir),
            "--runs-dir",
            str(ckpt_dir),
        ]
    )
    summary = posthoc.process_raw_directory(raw_dir=raw_dir, ckpt_dir=ckpt_dir)
    assert summary["updated"] == 1

    payload = json.loads((raw_dir / "tk3__elbe__d2__s0.json").read_text(encoding="utf-8"))
    assert payload["config"]["margin_semantics"] == "tightening"
    assert payload["metrics"]["modelness_train_margin"] <= payload["metrics"]["modelness_all"]
