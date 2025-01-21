import numpy as np
from numba import int32, float64, njit


@njit
def rmatrix_with_inverse(
    A: float64[:, :], b: float64[:], nchannels: int32, nbasis: int32, a: float64
):
    r"""Eqn 15 in Descouvemont, 2016"""
    R = np.zeros((nchannels, nchannels), dtype=np.complex128)
    C = np.linalg.inv(A)

    for i in range(nchannels):
        for j in range(nchannels):
            # TODO  benchmark if this is faster, can we do without inverting?
            Cblock = np.ascontiguousarray(
                C[i * nbasis : (i + 1) * nbasis, j * nbasis : (j + 1) * nbasis]
            )
            R[i, j] = b.T @ Cblock @ b
    return R / a**2, C


@njit
def solve_smatrix_with_inverse(
    A: float64[:, :],
    b: float64[:],
    Hp: float64[:],
    Hm: float64[:],
    Hpp: float64[:],
    Hmp: float64[:],
    incoming_weights: float64[:],
    a: float64,
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
    Zp = np.diag(Hp) - R @ np.diag(Hpp) * a
    Zm = np.diag(Hm) - R @ np.diag(Hmp) * a

    # Eqn 16 in Descouvemont, 2016
    S = np.linalg.solve(Zp, Zm)

    uext_prime_boundary = 1j / 2 * (Hmp * incoming_weights - S @ np.copy(Hpp))

    return R, S, Ainv, uext_prime_boundary


@njit
def solution_coeffs_with_inverse(
    Ainv: float64[:, :],
    b: float64[:],
    S: float64[:],
    uext_prime_boundary: float64[:],
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
    x = (b * uext_prime_boundary[:, np.newaxis]).reshape(nchannels * nbasis)
    return (Ainv @ x).reshape(nchannels, nbasis)
