"""Guard against example notebooks being saved with a locally registered kernel.

The notebook tests run with ``--nbval-current-env`` and so do not care about
the recorded kernelspec, but users opening a notebook in JupyterLab do: a
kernel name that only exists on one developer's machine produces a
"kernel not found" prompt for everyone else.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_default_python3_kernel(path):
    with open(path) as f:
        metadata = json.load(f).get("metadata", {})
    kernel_name = metadata.get("kernelspec", {}).get("name")
    assert kernel_name == "python3", (
        f"{path.name} is saved with kernel {kernel_name!r}; "
        "select the default 'Python 3 (ipykernel)' kernel and re-save."
    )
