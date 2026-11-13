"""jitr v4 runs on jax alone: no numba, and no v3 engine modules."""

import importlib
import subprocess
import sys

import pytest


def test_numba_is_not_imported():
    code = "import sys, jitr, jitr.xs; print('numba' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize(
    "name", ["jitr.quadrature", "jitr.lagrange", "jitr.xs.lane_pn"]
)
def test_v3_modules_are_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
