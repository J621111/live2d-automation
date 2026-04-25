from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTEST_TEMP_ROOT = Path(
    os.getenv("LIVE2D_PYTEST_TEMP_ROOT", PROJECT_ROOT / ".pytest_tmp_workspace")
).resolve()
os.environ.setdefault("LIVE2D_OUTPUT_ROOT", str(PYTEST_TEMP_ROOT))


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    # Windows tempdir creation is unreliable in this environment.
    # Tests use a managed workspace root instead.
    PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = PYTEST_TEMP_ROOT / f"test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
