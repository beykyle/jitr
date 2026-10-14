"""General-purpose numerical helpers used across :mod:`jitr`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .constants import HBARC


def radial_equation(r: float, y, l: int, k: float, mu: float, potential: Callable):
    r"""Right-hand side of ``u'' = [l(l+1)/r^2 + 2μ V(r)/(ħc)^2 - k^2] u`` as a
    first-order system ``y = (u, u')``, for integrators like
    :func:`scipy.integrate.solve_ivp` (``k`` in fm^-1, ``μ`` in MeV, ``V``
    in MeV)."""
    u, du = y
    return [du, (l * (l + 1) / r**2 + 2 * mu * potential(r) / HBARC**2 - k**2) * u]


def delta(Sl: complex) -> tuple[np.float64, np.float64]:
    """Return the phase shift and attenuation in degrees."""
    phase = np.log(Sl) / 2.0j
    return np.rad2deg(np.real(phase)), np.rad2deg(np.imag(phase))


def interaction_range(A: int, rA: float = 1.2, r0: float = 0.0) -> float:
    """Estimate an interaction radius in femtometers."""
    return r0 + rA * A ** (1 / 3)


def suggested_dimensionless_channel_radius(interaction_range: float, k: float) -> float:
    """Suggest a dimensionless channel radius for a given interaction range.

    The estimate leaves roughly one asymptotic wavelength beyond the interaction
    region.
    """
    return interaction_range * k + 2 * np.pi


def suggested_basis_size(a: float, zeros_per_node: int = 5) -> int:
    """Suggest an R-matrix basis size for a dimensionless channel radius."""
    return zeros_per_node * int(np.ceil(a / np.pi))


def suggest_lmax(k, channel_radius_fm: float, margin: int = 5) -> int:
    """Suggest the maximum partial wave for wavenumbers ``k`` (fm^-1).

    Partial waves with ``l`` well beyond ``k a`` do not reach the interaction
    region inside the channel radius ``a``; ``margin`` adds a few more. Use
    the largest ``k`` of an energy grid (the default takes ``max(k)``).
    """
    return int(np.ceil(np.max(k) * channel_radius_fm)) + margin
