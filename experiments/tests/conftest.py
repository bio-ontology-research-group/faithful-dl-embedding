"""Pytest configuration ensuring all temporary directories resolve under ~/tmp (never /tmp)."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest

_USER_TMP = Path("~/tmp").expanduser().resolve()
_USER_TMP.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(_USER_TMP)
os.environ["TEMP"] = str(_USER_TMP)
os.environ["TMP"] = str(_USER_TMP)
tempfile.tempdir = str(_USER_TMP)


def pytest_configure(config: pytest.Config) -> None:
    """Force pytest basetemp under ~/tmp when not explicitly set on the CLI."""
    basetemp = getattr(config.option, "basetemp", None)
    if not basetemp or str(basetemp).startswith("/tmp"):
        user = os.environ.get("USER", "user")
        target = _USER_TMP / f"pytest-of-{user}"
        target.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(target)
