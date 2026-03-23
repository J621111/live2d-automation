from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

PYTEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "output" / "pytest_tmp_workspace"


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = PYTEST_TEMP_ROOT / f"test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
