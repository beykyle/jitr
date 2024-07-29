import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, njit
import scipy.special as sc


@njit
def rmatrix_with_inverse(A, b, nchannels, nbasis, a):
    r"""Eqn 15 in Descouvemont, 2016"""
    R = np.zeros((nchannels, nchannels), dtype=np.complex128)
    C = np.linalg.inv(A)
    for i in range(nchannels):
        for j in range(nchannels):
            R[i, j] = (
                b[i * nbasis : (i + 1) * nbasis].T
                @ C[i * nbasis : (i + 1) * nbasis, j * nbasis : (j + 1) * nbasis]
                @ b[j * nbasis : (j + 1) * nbasis]
            )
    return R / np.outer(a, a), C


@njit
def solve_smatrix_with_inverse(
    A: np.array,
    b: np.array,
    Hp: np.array,
    Hm: np.array,
    Hpp: np.array,
    Hmp: np.array,
    incoming_weights: np.array,
    a: np.array,
    nchannels: int32,
    nbasis: int32,
):
    r"""
    @returns the multichannel R-Matrix, S-matrix, and wavefunction coefficients,
    all in Lagrange-Legendre coordinates, as well as the derivative of
    asymptotic channel Wavefunctions evaluated at the channel radius. Everything
    returned as block-matrices and block vectors in channel space.

    This follows: Descouvemont, P. (2016).  An R-matrix package for
    coupled-channel problems in nuclear physics.  Computer physics
    communications, 200, 199-219.
    """

    R, Ainv = rmatrix_with_inverse(A, b, nchannels, nbasis, a)

    # Eqn 17 in Descouvemont, 2016
    Zp = np.diag(Hp) - R * Hpp[:, np.newaxis] * a[:, np.newaxis]
    Zm = np.diag(Hm) - R * Hmp[:, np.newaxis] * a[:, np.newaxis]

    # Eqn 16 in Descouvemont, 2016
    S = np.linalg.solve(Zp, Zm)

    uext_prime_boundary = Hmp * incoming_weights - S @ np.copy(Hpp)

    return R, S, Ainv, uext_prime_boundary


@njit
def solve_smatrix_without_inverse(
    A: np.array,
    b: np.array,
    Hp: np.array,
    Hm: np.array,
    Hpp: np.array,
    Hmp: np.array,
    incoming_weights: np.array,
    a: np.array,
    nchannels: int32,
    nbasis: int32,
):
    r"""
    @returns the multichannel R-Matrix, S-matrix, and wavefunction coefficients,
    all in Lagrange-Legendre coordinates, as well as the derivative of
    asymptotic channel Wavefunctions evaluated at the channel radius. Everything
    returned as block-matrices and block vectors in channel space.

    This follows: Descouvemont, P. (2016).  An R-matrix package for
    coupled-channel problems in nuclear physics.  Computer physics
    communications, 200, 199-219.
    """

    # Eqn 15 in Descouvemont, 2016
    x = np.linalg.solve(A, b).reshape(nchannels, nbasis)
    R = x @ b.reshape(nchannels, nbasis).T / np.outer(a, a)

    # Eqn 17 in Descouvemont, 2016
    Zp = np.diag(Hp) - R * Hpp[:, np.newaxis] * a[:, np.newaxis]
    Zm = np.diag(Hm) - R * Hmp[:, np.newaxis] * a[:, np.newaxis]

    # Eqn 16 in Descouvemont, 2016
    S = np.linalg.solve(Zp, Zm)

    uext_prime_boundary = Hmp * incoming_weights - S @ np.copy(Hpp)

    return R, S, uext_prime_boundary


@njit
def solution_coeffs_with_inverse(
    Ainv: np.array,
    b: np.array,
    S: np.array,
    uext_prime_boundary: np.array,
    nchannels: int32,
    nbasis: int32,
):
    r"""
    @returns the multichannel wavefunction coefficients, in Lagrange- Legendre
    coordinates.

    This follows: Descouvemont, P. (2016).  An R-matrix package for
    coupled-channel problems in nuclear physics.  Computer physics
    communications, 200, 199-219.  and P. Descouvemont and D. Baye 2010 Rep.
    Prog. Phys. 73 036301
    """
    b2 = b.reshape(nchannels, nbasis) * uext_prime_boundary[:, np.newaxis]
    b2 = b2.reshape(nchannels * nbasis)
    return (Ainv @ b2).reshape(nchannels, nbasis)


@njit
def solution_coeffs(
    A: np.array,
    b: np.array,
    S: np.array,
    uext_prime_boundary: np.array,
    nchannels: int32,
    nbasis: int32,
):
    r"""
    @returns the multichannel wavefunction coefficients, in Lagrange- Legendre
    coordinates.

    This follows: Descouvemont, P. (2016).  An R-matrix package for
    coupled-channel problems in nuclear physics.  Computer physics
    communications, 200, 199-219.  and P. Descouvemont and D. Baye 2010 Rep.
    Prog. Phys. 73 036301
    """

    # Eqn 3.92 in Descouvemont & Baye, 2010
    b2 = b.reshape(nchannels, nbasis) * uext_prime_boundary[:, np.newaxis]
    b2 = b2.reshape(nchannels * nbasis)
    x = np.linalg.solve(A, b2).reshape(nchannels, nbasis)

    return x
