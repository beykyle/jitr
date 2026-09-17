"""Guard against example notebooks shipping without working Colab support.

A reader who clicks the "Open in Colab" badge gets a bare ``.ipynb``: no ``jitr``
and none of the sibling data files the notebook reads. ``scripts/add_colab_cells.py``
puts a badge cell and a Colab-guarded bootstrap cell at the top of every notebook to
cover that, so a notebook added or renamed without re-running it would either lose
the badge or carry one pointing at a path Colab renders as a bland "notebook not
found" page -- a silent failure. These tests are the reminder.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

_spec = importlib.util.spec_from_file_location(
    "add_colab_cells", REPO_ROOT / "scripts" / "add_colab_cells.py"
)
add_colab_cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(add_colab_cells)

FIXME = "re-run: python scripts/add_colab_cells.py"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_opens_with_its_own_colab_badge(path):
    cell = json.loads(path.read_text())["cells"][0]
    assert (
        cell["cell_type"] == "markdown"
    ), f"{path.name} does not start with the Colab badge cell; {FIXME}"
    assert "".join(cell["source"]) == add_colab_cells.badge_markdown(
        path.name
    ), f"{path.name} has a stale or borrowed Colab badge; {FIXME}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_bootstraps_colab_before_importing_anything(path):
    cell = json.loads(path.read_text())["cells"][1]
    assert (
        cell["cell_type"] == "code"
    ), f"{path.name} has no Colab setup cell as its first code cell; {FIXME}"
    assert "".join(cell["source"]) == add_colab_cells.setup_source(
        path.name
    ), f"{path.name} has a stale Colab setup cell; {FIXME}"
    # rendered docs hide the cell, and a committed pip log would show up on the site
    assert cell["metadata"].get("tags") == [
        "remove-cell"
    ], f"{path.name} setup cell is missing the remove-cell tag; {FIXME}"
    assert (
        cell["outputs"] == [] and cell["execution_count"] is None
    ), f"{path.name} setup cell was saved with output; clear it and re-save"
