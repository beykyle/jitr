"""Sphinx configuration for the jitr documentation site."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

project = "jitr"
author = "Kyle Beyer"
copyright = "2026, Kyle Beyer"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_use_ivar = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    ".git",
    ".github",
    ".venv",
    ".pytest_cache",
    "**/.pytest_cache",
    "build",
    "dist",
    "_build",
    "_build/**",
    "jupyter_execute",
    "jupyter_execute/**",
    "CHANGELOG.rst",
    "README.md",
    "SUPPORT.rst",
    "assets/jitr_logo.ipynb",
    "examples/notebooks/chex_jitr_validation.ipynb",
    "examples/notebooks/comparison_to_Runge_Kutta.ipynb",
    "examples/notebooks/test_coupled_single_dwba.ipynb",
    "examples/notebooks/test_dom_impl.ipynb",
    "examples/notebooks/test_dom_with_edep_a.ipynb",
    "examples/notebooks/how_to_define_your_interaction.ipynb",
    "examples/notebooks/integration.ipynb",
    "jitrbandsdk.md",
    "**/.ipynb_checkpoints",
]
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

nb_execution_mode = "off"
nb_number_source_lines = True
suppress_warnings = ["myst.header"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
# Fail fast instead of hanging when an inventory host is unreachable.
intersphinx_timeout = 30


class _IgnoreUnreachableInventories(logging.Filter):
    """Drop the intersphinx "failed to reach any of the inventories" warning.

    GitHub runners intermittently cannot reach docs.scipy.org and friends. That
    warning is emitted without a Sphinx warning ``type``, so it cannot be
    silenced through ``suppress_warnings``; without this filter it turns
    ``sphinx-build -W`` into a hard failure. Unresolved cross-references simply
    render as plain text when an inventory is missing.
    """

    _PREFIX = "failed to reach any of the inventories"

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith(self._PREFIX)


def setup(app: Sphinx) -> None:
    """Install the intersphinx warning filter."""

    # Sphinx prefixes its logger names with "sphinx."; the intersphinx logger
    # has been "sphinx.sphinx.ext.intersphinx" through Sphinx 9.
    logging.getLogger("sphinx.sphinx.ext.intersphinx").addFilter(
        _IgnoreUnreachableInventories()
    )


html_theme = "furo"
html_title = "jitr documentation"
html_logo = str(REPO_ROOT / "assets" / "jitr_logo.png")
html_static_path: list[str] = []
