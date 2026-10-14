"""Public package entry points for :mod:`jitr`."""

import jax

from . import (
    folding,
    optical_potentials,
    reactions,
    rmatrix,
    utils,
    xs,
)
from .__version__ import __version__
from .data import data_dir

# R-matrix solves need double precision; this sets it for the whole process,
# before any array exists (importing the submodules creates none).
jax.config.update("jax_enable_x64", True)

__all__ = [
    "__version__",
    "data_dir",
    "folding",
    "optical_potentials",
    "reactions",
    "rmatrix",
    "utils",
    "xs",
]
