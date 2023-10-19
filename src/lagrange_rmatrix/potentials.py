import numpy as np
from numba import njit
from .utils import alpha, hbarc


@njit
def woods_saxon_potential(r, params):
    V, W, R, a = params
    return -(V + 1j * W) / (1 + np.exp((r - R) / a))


@njit
def surface_peaked_gaussian_potential(r, params):
    V, W, R, a = params
    return -(V + 1j * W) * np.exp(-((r - R) ** 2) / (2 * np.pi * a) ** 2)


@njit
def coulomb_charged_sphere(r, zz, r_c):
    return zz * alpha * hbarc * regular_inverse_r(r, r_c)


@njit
def yamaguchi_potential(r, rp, params):
    """
    non-local potential with analytic s-wave phase shift; Eq. 6.14 in [Baye, 2015]
    """
    W0, beta, alpha = params
    return W0 * 2 * beta * (beta + alpha) ** 2 * np.exp(-beta * (r + rp))


@njit
def regular_inverse_r(r, r_c):
    if isinstance(r, float):
        return 1 / (2 * r_c) * (3 - (r / r_c) ** 2) if r < r_c else 1 / r
    else:
        ii = np.where(r <= r_c)[0]
        jj = np.where(r > r_c)[0]
        return np.hstack([1 / (2 * s_c) * (3 - (s[ii] / s_c) ** 2), 1 / s[jj]])


@njit
def yamaguchi_swave_delta(k, params):
    """
    analytic k * cot(phase shift) for yamaguchi potential; Eq. 6.15 in [Baye, 2015]
    """
    _, a, b = params
    d = 2 * (a + b) ** 2

    kcotdelta = (
        a * b * (a + 2 * b) / d
        + (a**2 + 2 * a * b + 3 * b**2) * k**2 / (b * d)
        + k**4 / (b * d)
    )

    delta = np.rad2deg(np.arctan(k / kcotdelta))
    return delta
