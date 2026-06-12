"""R-matrix solvers and low-level linear-algebra routines."""

import warnings

from . import core
from .rmatrix import Solver

__all__ = ["Solver", "core"]

warnings.warn(
    "jitr.rmatrix is deprecated: the lax-backed jitr.xs workspaces replace the "
    "internal R-matrix engine, and jitr.rmatrix will be removed in the next major "
    "release (see the lax-core rewrite CHANGELOG entry)",
    DeprecationWarning,
    stacklevel=2,
)
