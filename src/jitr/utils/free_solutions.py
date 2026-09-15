"""Free-particle and Coulomb asymptotic solutions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.special as sc
from mpmath import coulombf, coulombg
from numba import njit
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@njit
def Gamow_factor(l: int, eta: float) -> float:
    """Return the Coulomb Gamow factor for angular momentum ``l``."""
    if eta == 0.0:
        if l == 0:
            return 1.0
        return 1.0 / (2 * l + 1) * Gamow_factor(l - 1, 0.0)
    if l == 0:
        return np.sqrt(2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1))
    return np.sqrt(l**2 + eta**2) / (l * (2 * l + 1)) * Gamow_factor(l - 1, eta)


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
    asym: type = CoulombAsymptotics,
) -> complex:
    """Return the outgoing Coulomb-Hankel function."""
    return asym.G(s, l, eta) + 1j * asym.F(s, l, eta)


def H_minus(
    s: float,
    l: int,
    eta: float,
    asym: type = CoulombAsymptotics,
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
    asym: type = CoulombAsymptotics,
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
    asym: type = CoulombAsymptotics,
) -> complex:
    """Return the derivative of the incoming Coulomb-Hankel function."""
    return coulomb_func_deriv(
        lambda ss, ll, ee: H_minus(ss, ll, ee, asym=asym), s, l, eta
    )


def _riccati_bessel_table(rho: float, lmax: int) -> tuple[FloatArray, FloatArray]:
    """``F_l = rho j_l(rho)`` and ``G_l = -rho y_l(rho)`` for ``l = 0..lmax`` (eta = 0)."""
    ls = np.arange(lmax + 1)
    return rho * sc.spherical_jn(ls, rho), -rho * sc.spherical_yn(ls, rho)


def _coulomb_recurrence_table(
    rho: float, eta: float, lmax: int
) -> tuple[FloatArray, FloatArray]:
    """``F_l`` and ``G_l`` for ``l = 0..lmax`` from the three-term recurrence
    (Abramowitz & Stegun 14.2.3), anchored on ``mpmath`` at ``l = 0, 1``.

    ``G`` is recurred upward (stable: it is the dominant solution above the turning
    point).  ``F`` is recurred downward from ``l_top > max(lmax, rho)`` with an
    arbitrary start and normalised to the ``mpmath`` value at ``l = 0`` (Miller's
    algorithm; stable because ``F`` is the minimal solution).
    """
    ls = np.arange(0, lmax + 2, dtype=np.float64)  # one extra l for the derivatives
    n = ls.size

    def a(L):  # coefficient of u_{L+1}
        return L * np.sqrt((L + 1) ** 2 + eta**2)

    def b(L):  # coefficient of u_L
        return (2 * L + 1) * (eta + L * (L + 1) / rho)

    def c(L):  # coefficient of u_{L-1}
        return (L + 1) * np.sqrt(L**2 + eta**2)

    # G upward
    G = np.empty(n)
    G[0] = float(coulombg(0, eta, rho))
    if n > 1:
        G[1] = float(coulombg(1, eta, rho))
    for L in range(1, n - 1):
        G[L + 1] = (b(L) * G[L] - c(L) * G[L - 1]) / a(L)

    # F downward (Miller), with rescaling against overflow
    l_top = int(max(n + 1, rho + 10.0 * np.sqrt(rho) + 20))
    u_next, u = 0.0, 1e-30
    F = np.zeros(l_top + 1)
    F[l_top] = u
    for L in range(l_top, 0, -1):
        u_prev = (b(L) * u - a(L) * u_next) / c(L)
        u_next, u = u, u_prev
        F[L - 1] = u
        if abs(u) > 1e200:
            F[L - 1 :] /= 1e200
            u_next /= 1e200
            u /= 1e200
    F = F[:n] * (float(coulombf(0, eta, rho)) / F[0])
    return F, G


def _derivative_table(u: np.ndarray, rho: float, eta: float) -> np.ndarray:
    """``u'_l`` for ``l = 0..len(u)-2`` from ``u_l`` and ``u_{l+1}``
    (the relation :func:`coulomb_func_deriv` uses)."""
    L = np.arange(u.size - 1, dtype=np.float64)
    return ((L + 1) / rho + eta / (L + 1)) * u[:-1] - np.sqrt(
        1 + eta**2 / (L + 1) ** 2
    ) * u[1:]


def coulomb_hankel_table(
    rho: float, eta: float, lmax: int, wronskian_tol: float = 1e-8
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """``(H+, H-, H+', H-')`` at ``rho`` for every ``l = 0..lmax`` at once.

    Equivalent to :func:`H_plus`, :func:`H_minus`, :func:`H_plus_prime` and
    :func:`H_minus_prime` per ``l``, but ``F`` and ``G`` come from the three-term
    recurrence (two ``mpmath`` evaluations per table instead of ~six per ``l``) and
    the derivatives from the exact recurrence.  Every ``l`` is verified against the
    Wronskian ``F' G - F G' = 1``; any ``l`` failing ``wronskian_tol`` is recomputed
    with ``mpmath`` directly.
    """
    if eta == 0.0:
        F, G = _riccati_bessel_table(rho, lmax + 1)
    else:
        F, G = _coulomb_recurrence_table(rho, eta, lmax)
    Fp, Gp = _derivative_table(F, rho, eta), _derivative_table(G, rho, eta)
    F, G = F[: lmax + 1], G[: lmax + 1]
    bad = np.flatnonzero(np.abs(Fp * G - F * Gp - 1.0) > wronskian_tol)
    for l in bad:
        l = int(l)
        F[l] = CoulombAsymptotics.F(rho, l, eta).real
        G[l] = CoulombAsymptotics.G(rho, l, eta).real
        Fp[l] = coulomb_func_deriv(CoulombAsymptotics.F, rho, l, eta).real
        Gp[l] = coulomb_func_deriv(CoulombAsymptotics.G, rho, l, eta).real
    Hp = (G + 1j * F).astype(np.complex128)
    Hm = (G - 1j * F).astype(np.complex128)
    Hpp = (Gp + 1j * Fp).astype(np.complex128)
    Hmp = (Gp - 1j * Fp).astype(np.complex128)
    return Hp, Hm, Hpp, Hmp
