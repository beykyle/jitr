"""Quadrature rules and kernels used by the R-matrix solver."""

import warnings

from .kernel import Kernel
from .quadrature import (
    LagrangeLaguerreQuadrature,
    LagrangeLegendreQuadrature,
    generate_laguerre_quadrature,
    generate_legendre_quadrature,
    laguerre,
    legendre,
)

__all__ = [
    "Kernel",
    "LagrangeLaguerreQuadrature",
    "LagrangeLegendreQuadrature",
    "generate_laguerre_quadrature",
    "generate_legendre_quadrature",
    "laguerre",
    "legendre",
]

warnings.warn(
    "jitr.quadrature is deprecated: the lax-backed jitr.xs workspaces replace the "
    "internal R-matrix engine, and jitr.quadrature will be removed in the next major "
    "release (see the lax-core rewrite CHANGELOG entry)",
    DeprecationWarning,
    stacklevel=2,
)
