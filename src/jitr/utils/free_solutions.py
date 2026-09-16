"""Free-particle and Coulomb asymptotic solutions."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.special as sc
from mpmath import coulombf, coulombg
from numba import njit

from .._types import ComplexArray, FloatArray


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


class CoulombHankelTable(NamedTuple):
    """Coulomb-Hankel functions ``H± = G ± iF`` and derivatives for ``l = 0..lmax``."""

    Hp: ComplexArray
    Hm: ComplexArray
    Hpp: ComplexArray
    Hmp: ComplexArray


def _coulomb_FG(rho: float, eta: float, l: int) -> tuple[float, float]:
    """Return the regular and irregular Coulomb functions from :mod:`mpmath`.

    This is the only place the Coulomb functions are evaluated directly; the
    tables below anchor their recurrences on it and fall back to it.
    """
    return float(coulombf(l, eta, rho)), float(coulombg(l, eta, rho))


def _riccati_bessel_table(rho: float, lmax: int) -> tuple[FloatArray, FloatArray]:
    """Return ``F_l = rho j_l(rho)`` and ``G_l = -rho y_l(rho)`` for ``l = 0..lmax``."""
    ls = np.arange(lmax + 1)
    return rho * sc.spherical_jn(ls, rho), -rho * sc.spherical_yn(ls, rho)


def _coulomb_recurrence_table(
    rho: float, eta: float, lmax: int
) -> tuple[FloatArray, FloatArray]:
    """Return ``F_l`` and ``G_l`` for ``l = 0..lmax`` from the three-term recurrence.

    The recurrence is Abramowitz & Stegun 14.2.3, anchored on :func:`_coulomb_FG`
    at ``l = 0, 1`` (so ``lmax >= 1``).  ``G`` is recurred upward (stable: it is
    the dominant solution above the turning point).  ``F`` is recurred downward
    from ``l_top > max(lmax, rho)`` with an arbitrary start and normalised to the
    anchor at ``l = 0`` (Miller's algorithm; stable because ``F`` is the minimal
    solution).
    """
    assert lmax >= 1
    n = lmax + 1

    def a(L):  # coefficient of u_{L+1}
        return L * np.sqrt((L + 1) ** 2 + eta**2)

    def b(L):  # coefficient of u_L
        return (2 * L + 1) * (eta + L * (L + 1) / rho)

    def c(L):  # coefficient of u_{L-1}
        return (L + 1) * np.sqrt(L**2 + eta**2)

    F0, G0 = _coulomb_FG(rho, eta, 0)
    _, G1 = _coulomb_FG(rho, eta, 1)

    # G upward from the anchors
    G = np.empty(n)
    G[0], G[1] = G0, G1
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
    F = F[:n] * (F0 / F[0])
    return F, G


def _derivative_table(u: FloatArray, rho: float, eta: float) -> FloatArray:
    """Return ``u'_l`` for ``l = 0..len(u) - 2`` from ``u_l`` and ``u_{l+1}``.

    Abramowitz & Stegun 14.2.1 applied to a table of ``F`` or ``G``.
    """
    L = np.arange(u.size - 1, dtype=np.float64)
    return ((L + 1) / rho + eta / (L + 1)) * u[:-1] - np.sqrt(
        1 + eta**2 / (L + 1) ** 2
    ) * u[1:]


def coulomb_hankel_table(
    rho: float, eta: float, lmax: int, wronskian_tol: float = 1e-8
) -> CoulombHankelTable:
    """Return ``H+``, ``H-``, ``H+'`` and ``H-'`` at ``rho`` for ``l = 0..lmax``.

    For ``eta == 0`` the functions are Riccati-Bessel functions from
    :mod:`scipy`; otherwise ``F`` and ``G`` come from the three-term recurrence
    anchored on two :mod:`mpmath` evaluations.  The derivatives follow from the
    exact recurrence.  Every ``l`` is verified against the Wronskian
    ``F' G - F G' = 1``; any ``l`` failing ``wronskian_tol`` is recomputed with
    :mod:`mpmath` directly.
    """
    # one extra l for the derivatives
    if eta == 0.0:
        F, G = _riccati_bessel_table(rho, lmax + 1)
    else:
        F, G = _coulomb_recurrence_table(rho, eta, lmax + 1)
    Fp, Gp = _derivative_table(F, rho, eta), _derivative_table(G, rho, eta)
    F, G = F[: lmax + 1], G[: lmax + 1]
    bad = np.flatnonzero(np.abs(Fp * G - F * Gp - 1.0) > wronskian_tol)
    for l in bad.tolist():
        pair = np.array([_coulomb_FG(rho, eta, l), _coulomb_FG(rho, eta, l + 1)])
        F[l], G[l] = pair[0]
        Fp[l], Gp[l] = (
            _derivative_table(pair[:, 0], rho, eta)[0],
            _derivative_table(pair[:, 1], rho, eta)[0],
        )
    return CoulombHankelTable(
        Hp=(G + 1j * F).astype(np.complex128),
        Hm=(G - 1j * F).astype(np.complex128),
        Hpp=(Gp + 1j * Fp).astype(np.complex128),
        Hmp=(Gp - 1j * Fp).astype(np.complex128),
    )


def H_plus(s: float, l: int, eta: float) -> complex:
    """Return the outgoing Coulomb-Hankel function ``G + iF`` at ``s``."""
    return complex(coulomb_hankel_table(s, eta, l).Hp[l])


def H_minus(s: float, l: int, eta: float) -> complex:
    """Return the incoming Coulomb-Hankel function ``G - iF`` at ``s``."""
    return complex(coulomb_hankel_table(s, eta, l).Hm[l])


def H_plus_prime(s: float, l: int, eta: float) -> complex:
    """Return the derivative of the outgoing Coulomb-Hankel function at ``s``."""
    return complex(coulomb_hankel_table(s, eta, l).Hpp[l])


def H_minus_prime(s: float, l: int, eta: float) -> complex:
    """Return the derivative of the incoming Coulomb-Hankel function at ``s``."""
    return complex(coulomb_hankel_table(s, eta, l).Hmp[l])
