import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, njit


def laguerre_quadrature(nbasis: int):
    r"""
    @returns zeros and weights for Gauss quadrature using the Lagrange-Laguerre
    basis. See Ch. 3.3 of Baye, 2015
    """
    return np.polynomial.laguerre.laggauss(nbasis)


def legendre_quadrature(nbasis: int):
    r"""
    @returns zeros and weights for Gauss quadrature using the Lagrange-Legendre
    basis shifted and scaled onto [0,a]. See Ch. 3.4 of Baye, 2015
    """
    x, w = np.polynomial.legendre.leggauss(nbasis)
    x = 0.5 * (x + 1)
    w = 0.5 * w
    return x, w


kernel_static_typing = [
    ("nbasis", int32),
    ("nchannels", int32),
    ("abscissa", float64[:]),
    ("weights", float64[:]),
    ("overlap", float64[:, :]),
]


@jitclass(kernel_static_typing)
class LagrangeLaguerreRMatrixKernel:
    r"""
    Lagrange Laguerre mesh for the Schrödinger equation following ch. 3.3 of
    Baye, D.  (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel; e.g.  r
    -> s_i = r * k_i, and each channel's equation is then divided by it's
    asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(
        self,
        nbasis: int32,
        nchannels: int32,
        abscissa: np.array,
        weights: np.array,
        overlap: np.array = None,
    ):
        """
        Constructs the Schrödinger equation in a basis of Lagrange Laguerre
        functions
        """
        self.nbasis = nbasis
        self.nchannels = nchannels
        self.abscissa = abscissa
        self.weights = weights

        if overlap is None:
            # Eq. 3.71 in Baye, 2015
            imj = np.arange(nbasis) - np.arange(nbasis)[:, np.newaxis]
            self.overlap += (-1) ** imj / np.sqrt(np.outer(abscissa, abscissa))
        else:
            self.overlap = overlap

    def kinetic_operator_element(self, n: int32, m: int32, a: float64, l: int32):
        """
        @returns the (n,m)th matrix element for the kinetic energy operator at
        channel radius a = k*r with orbital angular momentum l
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        N = self.nbasis

        # Eq. 3.77 in Baye, 2015
        correction = (-1) ** (n - m) / 4 / np.sqrt(xn * xm)

        if n == m:
            # Eq. 3.75 in [Baye, 2015], scaled by 1/E and with r->s=kr
            centrifugal = l * (l + 1) / (a * xn) ** 2
            radial = (
                -1.0 / (12 * xn**2) * (xn**2 - 2 * (2 * N + 1) * xn - 4) / a**2
            )
            return radial - correction + centrifugal
        else:
            # Eq. 3.76 in [Baye, 2015], scaled by 1/E and with r->s=kr
            return (-1) ** (n - m) * (xn + xm) / np.sqrt(xn * xm) / (
                xn - xm
            ) ** 2 / a**2 - correction

    def kinetic_matrix(self, a: float64, l: int32):
        r"""
        @returns the kinetic operator matrix in the Lagrange Laguerre basis
        """
        F = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)
        for n in range(1, self.nbasis + 1):
            for m in range(n, self.nbasis + 1):
                F[n - 1, m - 1] = self.kinetic_operator_element(n, m, a, l)
        F = F + np.triu(F, k=1).T
        return F

    def free_matrix(
        self,
        a: np.array,
        l: np.array,
    ):
        r"""
        @returns the full (Nxn)x(Nxn) fulll free Schrödinger equation 1/E (H-E)
        in the Lagrange Laguerre basis, where each channel is an NxN block (N being the
        basis size), and there are nxn such blocks.
        """
        nb = self.nbasis
        sz = nb * self.nchannels
        F = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(self.nchannels):
            Fij = self.kinetic_matrix(a[i], l[i]) - self.overlap
            F[i * nb : i * nb + nb, i * nb : i * nb + nb] += Fij
        return F


@jitclass(kernel_static_typing)
class LagrangeLegendreRMatrixKernel:
    r"""
    Lagrange Legendre mesh for the Schrödinger equation following ch. 3.4 of
    Baye, D.  (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel; e.g.  r
    -> s_i = r * k_i, and each channel's equation is then divided by it's
    asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(
        self,
        nbasis: int32,
        nchannels: int32,
        abscissa: np.array,
        weights: np.array,
        overlap: np.array = None,
    ):
        """
        Constructs the Schrödinger equation in a basis of Lagrange Legendre
        functions
        """
        self.nbasis = nbasis
        self.nchannels = nchannels
        self.abscissa = abscissa
        self.weights = weights

        if overlap is None:
            self.overlap = np.diag(np.ones(nbasis))
        else:
            self.overlap = overlap

    def kinetic_operator_element(self, n: int32, m: int32, a: float64, l: int32):
        """
        @returns the (n,m)th matrix element for the kinetic energy + Bloch
        operator at channel radius a = k*r with orbital angular momentum l
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        N = self.nbasis

        if n == m:
            # Eq. 3.128 in [Baye, 2015], scaled by 1/E and with r->s=kr
            centrifugal = l * (l + 1) / (a * xn) ** 2
            radial = (
                ((4 * N**2 + 4 * N + 3) * xn * (1 - xn) - 6 * xn + 1)
                / (3 * xn**2 * (1 - xn) ** 2)
                / a**2
            )
            return radial + centrifugal
        else:
            # Eq. 3.129 in [Baye, 2015], scaled by 1/E and with r->s=kr
            return (
                (-1.0) ** (n + m)
                * (
                    (N**2 + N + 1.0)
                    + (xn + xm - 2 * xn * xm) / (xn - xm) ** 2
                    - 1.0 / (1.0 - xn)
                    - 1.0 / (1.0 - xm)
                )
                / np.sqrt(xn * xm * (1.0 - xn) * (1.0 - xm))
                / a**2
            )

    def kinetic_matrix(self, a: float64, l: int32):
        r"""
        @returns the kinetic operator matrix in the Lagrange Lagrange basis
        """
        F = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)
        for n in range(1, self.nbasis + 1):
            for m in range(n, self.nbasis + 1):
                F[n - 1, m - 1] = self.kinetic_operator_element(n, m, a, l)
        F = F + np.triu(F, k=1).T
        return F

    def free_matrix(
        self,
        a: np.array,
        l: np.array,
    ):
        r"""
        @returns the full (Nxn)x(Nxn) fulll free Schrödinger equation 1/E (H-E)
        in the Lagrange basis, where each channel is an NxN block (N being the
        basis size), and there are nxn such blocks.
        """
        nb = self.nbasis
        sz = nb * self.nchannels
        F = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(self.nchannels):
            Fij = self.kinetic_matrix(a[i], l[i]) - self.overlap
            F[i * nb : i * nb + nb, i * nb : i * nb + nb] += Fij
        return F


@njit
def rmsolve_smatrix(
    A: np.array,
    b: np.array,
    asymptotics: tuple,
    incoming_weights: np.array,
    a: np.array,
    nchannels: int32,
    nbasis: int32,
):
    r"""
    @returns the multichannel R-Matrix, S-matrix, and wavefunction coefficients,
    all in Lagrange- Legendre coordinates, as well as the derivative of
    asymptotic channel Wavefunctions evaluated at the channel radius. Everything
    returned as block-matrices and block vectors in channel space.

    This follows: Descouvemont, P. (2016).  An R-matrix package for
    coupled-channel problems in nuclear physics.  Computer physics
    communications, 200, 199-219.
    """
    (Hp, Hm, Hpp, Hmp) = asymptotics

    # Eqn 15 in Descouvemont, 2016
    x = np.linalg.solve(A, b).reshape(nchannels, nbasis)
    R = x @ b.reshape(nchannels, nbasis).T / np.outer(a, a)

    # Eqn 17 in Descouvemont, 2016
    Zp = np.diag(Hp) - R * Hpp[:, np.newaxis] * a[:, np.newaxis]
    Zm = np.diag(Hm) - R * Hmp[:, np.newaxis] * a[:, np.newaxis]

    # Eqn 16 in Descouvemont, 2016
    S = np.linalg.solve(Zp, Zm)

    uext_prime_boundary = Hmp * incoming_weights - S @ Hpp

    return R, S, uext_prime_boundary


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
    b2 = (
        b.reshape(nchannels, nbasis) * uext_prime_boundary[:, np.newaxis]
    ).reshape(nchannels * nbasis)
    x = np.linalg.solve(A, b2).reshape(nchannels, nbasis)

    return x
