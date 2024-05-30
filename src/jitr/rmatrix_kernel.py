import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, njit

from .system import InteractionMatrix
from .channel import ChannelData
from .utils import eval_scaled_interaction, eval_scaled_nonlocal_interaction, block


spec = [
    ("nbasis", int32),
    ("nchannels", int32),
    ("abscissa", float64[:]),
    ("weights", float64[:]),
]


@jitclass(spec)
class LagrangeRMatrixKernel:
    r"""
    Lagrange-Legendre mesh for the Bloch-Schrödinger equation following:
    Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel; e.g.
    r -> s_i = r * k_i, and each channel's equation is then divided by it's
    asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(self, nbasis, nchannels, abscissa, weights):
        """
        Constructs the Bloch-Schrödinger equation in a basis of nbasis
        Lagrange-Legendre functions shifted and scaled onto [0,k*a] and regulated by 1/k*r,
        and solved by direct matrix inversion.

        """
        self.nbasis = nbasis
        self.nchannels = nchannels
        self.abscissa = abscissa
        self.weights = weights

    def local_interaction_matrix_element(
        self, n: np.int32, m: np.int32, interaction, ch: ChannelData, args=()
    ):
        """
        evaluates the (n,m)th matrix element for the given local interaction
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        if n != m:
            return 0  # local interactions are diagonal

        xn = self.abscissa[n - 1]

        a = ch.domain[1]
        s = xn * a

        return eval_scaled_interaction(s, interaction, ch, args)

    def nonlocal_interaction_matrix_element(
        self, n: np.int32, m: np.int32, interaction, ch: ChannelData, args=()
    ):
        """
        evaluates the (n,m)th matrix element for the given non-local interaction
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn = self.abscissa[n - 1]
        xm = self.abscissa[m - 1]
        wn = self.weights[n - 1]
        wm = self.weights[m - 1]

        a = ch.domain[1]

        s = xn * a
        sp = xm * a

        utilde = eval_scaled_nonlocal_interaction(s, sp, interaction, ch, args)

        return utilde * np.sqrt(wm * wn) * a

    def single_channel_nonlocal_interaction_matrix(
        self,
        interaction,
        ch: ChannelData,
        is_symmetric: bool = True,
        args=None,
    ):
        r"""Implements Eq. 6.10 in [Baye, 2015], scaled by 1/E and with r->s=kr, for
        just the interaction provided
        """
        V = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)

        # diagonal and upper triangle
        for n in range(1, self.nbasis + 1):
            for m in range(n, self.nbasis + 1):
                V[n - 1, m - 1] = self.nonlocal_interaction_matrix_element(
                    n, m, interaction
                )

        if is_symmetric:
            # transpose upper tri to lower
            V = V + np.triu(V, k=1).T
        else:
            # calculate lower triangle
            for n in range(1, self.nbasis + 1):
                for m in range(n + 1, self.nbasis + 1):
                    V[m - 1, n - 1] = self.nonlocal_interaction_matrix_element(
                        m, n, interaction
                    )

        return V

    def single_channel_local_interaction_matrix(
        self,
        interaction,
        ch: ChannelData,
        args=None,
    ):
        r"""Implements Eq. 6.10 in [Baye, 2015], scaled by 1/E and with r->s=kr, for
        just the interaction provided
        """
        V = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)

        # just diagonal
        for n in range(1, self.nbasis + 1):
            V[n - 1, n - 1] = self.local_interaction_matrix_element(
                n, m, interaction, ch, args
            )

        return V

    def free_matrix_element(
        self, n: np.int32, m: np.int32, a: np.float64, l: np.int32
    ):
        """
        evaluates the (n,m)th matrix element for the kinetic energy + Bloch operator
        at channel radius a = k*r with orbital angular momentum l
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        N = self.nbasis

        if n == m:
            centrifugal = l * (l + 1) / (a * xn) ** 2
            # Eq. 3.128 in [Baye, 2015], scaled by 1/E and with r->s=kr
            return ((4 * N**2 + 4 * N + 3) * xn * (1 - xn) - 6 * xn + 1) / (
                3 * xn**2 * (1 - xn) ** 2
            ) / a**2 + centrifugal
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

    def single_channel_free_matrix(self, a: np.float64, l: np.int32):
        r"""
        Returns the free Bloch-Schrödinger equation in a single channel in Lagrange-Legendre coordinates
        """
        F = np.empty((self.nbasis, self.nbasis), dtype=np.complex128)
        for n in range(1, self.nbasis + 1):
            for m in range(n, self.nbasis + 1):
                F[n - 1, m - 1] = self.free_matrix_element(n, m, a, l)

        F -= np.diag(np.ones(self.nbasis))
        F = F + np.triu(F, k=1).T

        return F

    def free_matrix(
        self,
        a: np.array,
        l: np.array,
    ):
        r"""
        Returns the full (Nxn)x(Nxn) free Bloch-Schrödinger equation in the Lagrange basis,
        where each channel is annxn block (n being the basis size), and there are NxN such blocks
        """
        nb = self.nbasis
        sz = nb * self.nchannels
        C = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(self.kernel.nchannels):
            Fij = self.kernel.single_channel_free_matrix(a[i], l[i])
            C[i * nb : i * nb + nb, i * nb : i * nb + nb] += Fij

        return C


@njit
def rmsolve_smatrix(
    A: np.array,
    b: np.array,
    asymptotics: tuple,
    incoming_weights: np.array,
    a: np.array,
    nchannels: np.int32,
    nbasis: np.int32,
):
    """
    Returns the multichannel R-Matrix, S-matrix, and wavefunction coefficients, all in Lagrange-
    Legendre coordinates, as well as the derivative of asymptotic channel Wavefunctions evaluated
    at the channel radius. Everything returned as block-matrices and block vectors in channel space.

    This follows:
    Descouvemont, P. (2016).
    An R-matrix package for coupled-channel problems in nuclear physics.
    Computer physics communications, 200, 199-219.
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
def rmsolve_wavefunction(
    A: np.array,
    b: np.array,
    asymptotics: tuple,
    incoming_weights: np.array,
    a: np.array,
    nchannels: np.int32,
    nbasis: np.int32,
):
    """
    Returns the multichannel R-Matrix, S-matrix, and wavefunction coefficients, all in Lagrange-
    Legendre coordinates, as well as the derivative of asymptotic channel Wavefunctions evaluated
    at the channel radius. Everything returned as block-matrices and block vectors in channel space.

    This follows:
    Descouvemont, P. (2016).
    An R-matrix package for coupled-channel problems in nuclear physics.
    Computer physics communications, 200, 199-219.
    and
    P. Descouvemont and D. Baye 2010 Rep. Prog. Phys. 73 036301
    """
    # should we factorize A so we don't have to solve twice?
    R, S, uext_prime_boundary = rmsolve_smatrix(
        A, b, asymptotics, incoming_weights, a, nchannels, nbasis
    )

    # Eqn 3.92 in Descouvemont & Baye, 2010
    b2 = (b.reshape(nchannels, nbasis) * uext_prime_boundary[:, np.newaxis]).reshape(
        nchannels * nbasis
    )
    x = np.linalg.solve(A, b2).reshape(nchannels, nbasis)

    return R, S, x, uext_prime_boundary
