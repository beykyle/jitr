import numpy as np
from .utils import alpha, hbarc


def woods_saxon_potential(r, params):
    V, W, R, a = params
    return -(V + 1j * W) / (1 + np.exp((r - R) / a))


def surface_peaked_gaussian_potential(r, params):
    V, W, R, a = params
    return -(V + 1j * W) * np.exp(-((r - R) ** 2) / (2 * np.pi * a) ** 2)


def coulomb_potential(zz, r, R):
    if r > R:
        return zz * alpha * hbarc / r
    else:
        return zz * alpha * hbarc / (2 * R) * (3 - (r / R) ** 2)


def yamaguchi_potential(r, rp, params):
    """
    non-local potential with analytic s-wave phase shift; Eq. 6.14 in [Baye, 2015]
    """
    W0, beta, alpha = params
    return W0 * 2 * beta * (beta + alpha) ** 2 * np.exp(-beta * (r + rp))


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
