"""Public package entry points for :mod:`jitr`.

Importing :mod:`jitr` enables JAX double precision (x64) process-wide: the
lax-backed solvers and dispersion kernels require float64, so it is set once
here, before any submodule builds a JAX array.
"""

import jax

jax.config.update("jax_enable_x64", True)

from . import folding, optical_potentials, reactions, utils, xs  # noqa: E402
from .__version__ import __version__  # noqa: E402
from .data import data_dir  # noqa: E402

__all__ = [
    "__version__",
    "data_dir",
    "folding",
    "optical_potentials",
    "reactions",
    "utils",
    "xs",
]
