"""Public package entry points for :mod:`jitr`."""

import importlib

from . import folding, optical_potentials, reactions, utils, xs
from .__version__ import __version__
from .data import data_dir

__all__ = [
    "__version__",
    "data_dir",
    "folding",
    "optical_potentials",
    "quadrature",
    "reactions",
    "rmatrix",
    "utils",
    "xs",
]

_DEPRECATED_MODULES = ("quadrature", "rmatrix")


def __getattr__(name: str):
    # lazy so the DeprecationWarning fires only when the legacy engine is
    # actually used, not on every `import jitr`
    if name in _DEPRECATED_MODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
