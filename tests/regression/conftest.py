from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


def iter_manifest() -> list[dict[str, str]]:
    """Return the committed regression manifest entries."""
    return json.loads((ROOT / "manifest.json").read_text())


@pytest.fixture(params=iter_manifest(), ids=lambda case: case["case_id"])
def case(request: pytest.FixtureRequest) -> Iterator[dict[str, str]]:
    """Parametrize the regression suite over the committed manifest."""
    yield request.param
