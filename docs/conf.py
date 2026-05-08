"""Sphinx configuration for the jitr documentation site."""

from __future__ import annotations

import sys
from pathlib import Path

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
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

root_doc = "docs/index"
templates_path = ["_templates"]
exclude_patterns = [
    ".git",
    ".github",
    ".venv",
    ".pytest_cache",
    "**/.pytest_cache",
    "build",
    "dist",
    "docs/_build",
    "docs/_build/**",
    "CHANGELOG.rst",
    "README.md",
    "SUPPORT.rst",
    "assets/jitr_logo.ipynb",
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

html_theme = "furo"
html_title = "jitr documentation"
html_logo = str(REPO_ROOT / "assets" / "jitr_logo.png")
html_static_path: list[str] = []
