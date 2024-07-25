import numpy as np
from numba import njit
from mpmath import coulombf, coulombg
import scipy.special as sc
from scipy.misc import derivative

alpha = 1.0 / 137.0359991  # dimensionless fine structure constant
hbarc = 197.3269804  # hbar*c in [MeV femtometers]
c = 2.99792458e23  # fm/s


@njit
def classical_kinematics(mass_target, mass_projectile, E_lab, Q, Zz):
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    E_com = mass_target / (mass_target + mass_projectile) * E_lab
    k = np.sqrt(2 * (E_com + Q) * mu) / hbarc
    eta = (alpha * Zz) * mu / (hbarc * k)
    return mu, E_com, k, eta


@njit
def complex_det(matrix: np.array):
    d = np.linalg.det(matrix @ np.conj(matrix).T)
    return np.sqrt(d)


@njit
def block(matrix: np.array, block, block_size):
    """
    get submatrix with coordinates block from matrix, where
    each block is defined by block_size elements along each dimension
    """
    i, j = block
    n, m = block_size
    return matrix[i * n : i * n + n, j * m : j * m + m]


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


@njit
def second_derivative_op(s, channel, interaction, args=()):
    r"""second derivative operator of reduced, scaled radial Schrodinger equation"""
    return (
        eval_scaled_interaction(s, interaction, channel, args)
        + channel.l * (channel.l + 1) / s**2
        - 1.0
    )


@njit
def schrodinger_eqn_ivp_order1(s, y, channel, interaction, args=()):
    r"""
    callable for scipy.integrate.solve_ivp; converts SE to
    2 coupled 1st order ODEs
    """
    u, uprime = y
    return [uprime, second_derivative_op(s, channel, interaction, args) * u]


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


def H_plus_prime(s, l, eta, dx=1e-6, asym=CoulombAsymptotics):
    """
    Derivative of the Hankel function (first kind) with respect to s.
    """
    return derivative(lambda z: H_plus(z, l, eta, asym), s, dx=dx)


def H_minus_prime(s, l, eta, dx=1e-6, asym=CoulombAsymptotics):
    """
    Derivative of the Hankel function (second kind) with respect to s.
    """
    return derivative(lambda z: H_minus(z, l, eta, asym), s, dx=dx)


def smatrix(Rl, a, l, eta, asym=CoulombAsymptotics):
    """
    Calculates channel S-Matrix from channel R-matrix (logarithmic
    derivative of channel wavefunction at channel radius)
    """
    return (
        H_minus(a, l, eta, asym=asym) - a * Rl * H_minus_prime(a, l, eta, asym=asym)
    ) / (H_plus(a, l, eta, asym=asym) - a * Rl * H_plus_prime(a, l, eta, asym=asym))


@njit
def delta(Sl):
    """
    returns the phase shift and attentuation factor in degrees
    """
    delta = np.log(Sl) / 2.0j  # complex phase shift in radians
    return np.rad2deg(np.real(delta)), np.rad2deg(np.imag(delta))


@njit
def null(*args):
    return 0.0


@njit
def eval_scaled_interaction(s, interaction, ch, args):
    return interaction(s / ch.k, *args) / ch.E


@njit
def eval_scaled_nonlocal_interaction(s, sp, interaction, ch, args):
    return interaction(s / ch.k, sp / ch.k, *args) / ch.E


@njit
def njit_eval_legendre(n, x):
    return sc.eval_legendre(n, x)
