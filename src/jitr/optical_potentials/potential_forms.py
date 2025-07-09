import numpy as np
from scipy import special as sc
from ..utils.constants import ALPHA, HBARC

MAX_ARG = np.log(1 / 1e-16)


def perey_buck_nonlocal(r, rp, *params):
    """Eq. A.2 in Perey  & Buck, 1962. Just the non-local factor H(r,rp)."""
    beta, l = params
    z = 2 * np.pi * r * rp / beta**2
    Kl = 2 * 1j**l * z * sc.spherical_jn(l, -1j * z)
    return np.exp(-(r**2 + rp**2) / beta**2) * Kl / (beta * np.sqrt(np.pi))


def woods_saxon_potential(r, *params):
    V, W, R, a = params
    return (V + 1j * W) * woods_saxon_safe(r, R, a)


def woods_saxon_prime(r, *params):
    """derivative of the Woods-Saxon potential w.r.t. $r$"""
    V, W, R, a = params
    return (V + 1j * W) * woods_saxon_prime_safe(r, R, a)


def woods_saxon_safe(r, R, a):
    """Woods-Saxon potential. avoids `exp` overflows"""
    x = (r - R) / a
    if isinstance(x, float):
        return 1.0 / (1.0 + np.exp(x)) if x < MAX_ARG else 0
    else:
        mask = x <= MAX_ARG
        V = np.zeros_like(r)
        V[mask] = 1.0 / (1.0 + np.exp(x[mask]))
        return V


def woods_saxon_prime_safe(r, R, a):
    """derivative of the Woods-Saxon potential w.r.t. $r$ avoids `exp` overflows"""
    x = (r - R) / a
    if isinstance(x, float):
        return -1 / a * np.exp(x) / (1 + np.exp(x)) ** 2 if x < MAX_ARG else 0
    else:
        mask = x <= MAX_ARG
        V = np.zeros_like(r)
        V[mask] = -1 / a * np.exp(x[mask]) / (1 + np.exp(x[mask])) ** 2
        return V


def thomas_safe(r, R, a):
    """1/r * derivative of the Woods-Saxon potential w.r.t. $r$, avoids
    `exp` overflows, while correctly handeling 1/r term
    """
    x = (r - R) / a
    y = 1.0 / r
    if isinstance(x, float):
        return y * -1 / a * np.exp(x) / (1 + np.exp(x)) ** 2 if x < MAX_ARG else 0
    else:
        mask = x <= MAX_ARG
        V = np.zeros_like(r)
        V[mask] = y[mask] * -1 / a * np.exp(x[mask]) / (1 + np.exp(x[mask])) ** 2
        return V


def surface_peaked_gaussian_potential(r, *params):
    V, W, R, a = params
    return (V + 1j * W) * np.exp(-((r - R) ** 2) / (2 * np.pi * a) ** 2)


def woods_saxon_volume_integral(V, R, a):
    return 4 * np.pi / 3 * (V * R**3) * (1 + (np.pi * a / R) ** 2)


def woods_saxon_mean_square_radius(R, a):
    return 3.0 / 5 * R**2 * (1 + 7.0 / 3 * (np.pi * a / R) ** 2)


def woods_saxon_prime_volume_integral(V, R, a):
    return (
        (4 * np.pi / 3) * V * R**3 * 12 * a / R * (1 + 1.0 / 3 * (np.pi * a / R) ** 2)
    )


def woods_saxon_prime_mean_square_radius(R, a):
    return R**2 * (1 + 5.0 / 3 * (np.pi * a / R) ** 2)


def thomas_volume_integral(V, R, a):
    return 8 * np.pi * R**3 * V * (1 + (np.pi * a / R) ** 2)


def thomas_mean_square_radius(R, a):
    return R**2 * (1 + 7.0 / 3.0 * (np.pi * a / R) ** 2)


def coulomb_charged_sphere(r, zz, r_c):
    return zz * ALPHA * HBARC * regular_inverse_r(r, r_c)


def regular_inverse_r(r, r_c):
    if isinstance(r, float):
        return 1 / (2 * r_c) * (3 - (r / r_c) ** 2) if r < r_c else 1 / r
    else:
        mask = r <= r_c
        not_mask = np.logical_not(mask)
        V = np.zeros_like(r)
        V[mask] = 1.0 / (2.0 * r_c) * (3.0 - (r[mask] / r_c) ** 2)
        V[not_mask] = 1.0 / r[not_mask]
        return V


def yamaguchi_potential(r, rp, *params):
    """
    non-local potential with analytic s-wave phase shift; Eq. 6.14 in [Baye, 2015]
    """
    W0, beta, ALPHA = params
    return -W0 * 2 * beta * (beta + ALPHA) ** 2 * np.exp(-beta * (r + rp))


def yamaguchi_swave_delta(k, *params):
    """
    analytic k * cot(phase shift) for yamaguchi potential; Eq. 6.15 in [Baye, 2015]
    """
    _, b, a = params
    d = 2 * (a + b) ** 2

    kcotdelta = (
        a * b * (a + 2 * b) / d
        + (a**2 + 2 * a * b + 3 * b**2) * k**2 / (b * d)
        + k**4 / (b * d)
    )

    delta = np.rad2deg(np.arctan(k / kcotdelta))
    return delta
