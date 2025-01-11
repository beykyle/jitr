from numba import njit
import scipy.special as sc
from mpmath import coulombf, coulombg
import numpy as np


@njit
def Gamow_factor(l, eta):
    r"""This returns the... Gamow factor.
    See [Wikipedia](https://en.wikipedia.org/wiki/Gamow_factor).

    Parameters:
        l (int): angular momentum
        eta (float): Sommerfeld parameter (see
            [Wikipedia](https://en.wikipedia.org/wiki/Sommerfeld_parameter))

    Returns:
        C_l (float): Gamow factor

    """
    if eta == 0.0:
        if l == 0:
            return 1
        else:
            return 1 / (2 * l + 1) * Gamow_factor(l - 1, 0)
    elif l == 0:
        return np.sqrt(2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1))
    else:
        return np.sqrt(l**2 + eta**2) / (l * (2 * l + 1)) * Gamow_factor(l - 1, eta)


class FreeAsymptotics:
    r"""For neutral particles, one may desired to explicitly pass in the type FreeAsymptotics
    into H_plus, H_minus, etc., for speed, as, while Coulomb functions reduce to the spherical
    Bessels for neutral particles, the arbitrary precision implementation in mpmath will be much
    slower than this
    """

    def __init__():
        pass

    @staticmethod
    def F(s, l, _=None):
        """
        Bessel function of the first kind.
        """
        return s * sc.spherical_jn(l, s)

    @staticmethod
    def G(s, l, _=None):
        """
        Bessel function of the second kind.
        """
        return -s * sc.spherical_yn(l, s)


class CoulombAsymptotics:
    @staticmethod
    def F(s, l, eta):
        """
        Coulomb function of the first kind.
        """
        return np.complex128(coulombf(l, eta, s))

    @staticmethod
    def G(s, l, eta):
        """
        Coulomb function of the second kind.
        """
        return np.complex128(coulombg(l, eta, s))


def H_plus(s, l, eta, asym=CoulombAsymptotics):
    """
    Hankel/Coulomb-Hankel function of the first kind (outgoing).
    """
    return asym.G(s, l, eta) + 1j * asym.F(s, l, eta)


def H_minus(s, l, eta, asym=CoulombAsymptotics):
    """
    Hankel/Coulomb-Hankel function of the second kind (incoming).
    """
    return asym.G(s, l, eta) - 1j * asym.F(s, l, eta)


def coulomb_func_deriv(func, s, l, eta):
    """
    Derivative of Coulomb functions F, G, and Coulomb Hankel functions H+ and H-
    """
    # recurrance relations from https://dlmf.nist.gov/33.4
    # dlmf Eq. 33.4.4
    R = np.sqrt(1 + eta**2 / (l + 1) ** 2)
    S = (l + 1) / s + eta / (l + 1)
    Xl = func(s, l, eta)
    Xlp = func(s, l + 1, eta)
    return S * Xl - R * Xlp


def H_plus_prime(s, l, eta, asym=CoulombAsymptotics):
    """
    Derivative of the Hankel function (first kind) with respect to s
    """
    return coulomb_func_deriv(H_plus, s, l, eta)


def H_minus_prime(s, l, eta, dx=1e-6, asym=CoulombAsymptotics):
    """
    Derivative of the Hankel function (second kind) with respect to s.
    """
    return coulomb_func_deriv(H_minus, s, l, eta)
