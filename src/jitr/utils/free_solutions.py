"""Free-particle and Coulomb asymptotic solutions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.special as sc
from mpmath import coulombf, coulombg


def Gamow_factor(l: int, eta: float) -> float:
    """Return the Coulomb Gamow factor for angular momentum ``l``.

    Iterative form of the textbook recursion (setup-time helper; called
    O(lmax) times, so compilation buys nothing).
    """
    if eta == 0.0:
        result = 1.0
        for m in range(1, l + 1):
            result /= 2 * m + 1
        return result
    result = float(np.sqrt(2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1)))
    for m in range(1, l + 1):
        result *= np.sqrt(m**2 + eta**2) / (m * (2 * m + 1))
    return result


class FreeAsymptotics:
    """Spherical-Bessel asymptotics for neutral-particle scattering."""

    @staticmethod
    def F(s: float, l: int, _eta: float | None = None) -> np.float64:
        """Return the regular free solution."""
        return s * sc.spherical_jn(l, s)

    @staticmethod
    def G(s: float, l: int, _eta: float | None = None) -> np.float64:
        """Return the irregular free solution."""
        return -s * sc.spherical_yn(l, s)


class CoulombAsymptotics:
    """Coulomb asymptotic functions evaluated through :mod:`mpmath`."""

    @staticmethod
    def F(s: float, l: int, eta: float) -> np.complex128:
        """Return the regular Coulomb function."""
        return np.complex128(coulombf(l, eta, s))

    @staticmethod
    def G(s: float, l: int, eta: float) -> np.complex128:
        """Return the irregular Coulomb function."""
        return np.complex128(coulombg(l, eta, s))


def H_plus(
    s: float,
    l: int,
    eta: float,
    asym: Any = CoulombAsymptotics,
) -> complex:
    """Return the outgoing Coulomb-Hankel function."""
    return asym.G(s, l, eta) + 1j * asym.F(s, l, eta)


def H_minus(
    s: float,
    l: int,
    eta: float,
    asym: Any = CoulombAsymptotics,
) -> complex:
    """Return the incoming Coulomb-Hankel function."""
    return asym.G(s, l, eta) - 1j * asym.F(s, l, eta)


def coulomb_func_deriv(
    func: Callable[[float, int, float], complex],
    s: float,
    l: int,
    eta: float,
) -> complex:
    """Differentiate Coulomb or Coulomb-Hankel functions using recurrence relations."""
    recurrence_factor = np.sqrt(1 + eta**2 / (l + 1) ** 2)
    shift_term = (l + 1) / s + eta / (l + 1)
    Xl = func(s, l, eta)
    Xlp = func(s, l + 1, eta)
    return shift_term * Xl - recurrence_factor * Xlp


def H_plus_prime(
    s: float,
    l: int,
    eta: float,
    asym: Any = CoulombAsymptotics,
) -> complex:
    """Return the derivative of the outgoing Coulomb-Hankel function."""
    return coulomb_func_deriv(
        lambda ss, ll, ee: H_plus(ss, ll, ee, asym=asym), s, l, eta
    )


def H_minus_prime(
    s: float,
    l: int,
    eta: float,
    dx: float = 1e-6,
    asym: Any = CoulombAsymptotics,
) -> complex:
    """Return the derivative of the incoming Coulomb-Hankel function."""
    return coulomb_func_deriv(
        lambda ss, ll, ee: H_minus(ss, ll, ee, asym=asym), s, l, eta
    )
