import numpy as np
from mpmath import coulombf, coulombg
from scipy.special import spherical_jn, spherical_yn, eval_legendre, roots_legendre
from scipy.interpolate import interp1d
from scipy.misc import derivative

alpha = 1.0 / 137.0359991  # dimensionless fine structure constant
hbarc = 197.3  # MeV fm


def complex_det(matrix: np.array):
    d = np.linalg.det(matrix @ np.conj(matrix).T)
    return np.sqrt(d)


def block(matrix: np.array, block, block_size):
    """
    get submatrix with coordinates block from matrix, where
    each block is defined by block_size elements along each dimension
    """
    i, j = block
    n, m = block_size
    return matrix[i * n : i * n + n, j * m : j * m + m]


def Gamow_factor(l, eta):
    if eta == 0.0:
        if l == 0:
            return 1
        else:
            return 1 / (2 * l + 1) * Gamow_factor(l - 1, 0)
    elif l == 0:
        return np.sqrt(2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1))
    else:
        return np.sqrt(l**2 + eta**2) / (l * (2 * l + 1)) * Gamow_factor(l - 1, eta)


def F(s, ell, eta):
    """
    Bessel function of the first kind.
    """
    # return s*spherical_jn(ell, s)
    return np.complex128(coulombf(ell, eta, s))


def G(s, ell, eta):
    """
    Bessel function of the second kind.
    """
    # return -s*spherical_yn(ell, s)
    return np.complex128(coulombg(ell, eta, s))


def H_plus(s, ell, eta):
    """
    Hankel function of the first kind.
    """
    return G(s, ell, eta) + 1j * F(s, ell, eta)


def H_minus(s, ell, eta):
    """
    Hankel function of the second kind.
    """
    return G(s, ell, eta) - 1j * F(s, ell, eta)


def H_plus_prime(s, ell, eta, dx=1e-6):
    """
    Derivative of the Hankel function (first kind) with respect to s.
    """
    return derivative(lambda z: H_plus(z, ell, eta), s, dx=dx)


def H_minus_prime(s, ell, eta, dx=1e-6):
    """
    Derivative of the Hankel function (second kind) with respect to s.
    """
    return derivative(lambda z: H_minus(z, ell, eta), s, dx=dx)


# vectorized versions of arbitrary precision Coulomb funcs form mpmath
VH_minus = np.frompyfunc(H_minus, 3, 1)
VH_plus = np.frompyfunc(H_minus, 3, 1)


def smatrix(Rl, a, l, eta):
    """
    Calculates channel S-Matrix from channel R-matrix (logarithmic
    derivative of channel wavefunction at channel radius)
    """
    return (H_minus(a, l, eta) - a * Rl * H_minus_prime(a, l, eta)) / (
        H_plus(a, l, eta) - a * Rl * H_plus_prime(a, l, eta)
    )


def delta(Sl):
    """
    returns the phase shift and attentuation factor in degrees
    """
    delta = np.log(Sl) / 2.0j  # complex phase shift in radians
    return np.rad2deg(np.real(delta)), np.rad2deg(np.imag(delta))
