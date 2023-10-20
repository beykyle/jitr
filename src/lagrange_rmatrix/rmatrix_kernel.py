import numpy as np
from numba.experimental import jitclass
from numba import int32, float32

from .system import InteractionMatrix
from .channel import ChannelData
from .utils import eval_scaled_interaction, eval_scaled_nolocal_interaction, block


spec = [
    ("nbasis", int32),
    ("nchannels", int32),
    ("abscissa", float64[:]),
    ("weights", float64[:]),
]


@jitclass(spec)
class LagrangeRMatrixKernel:
    r"""
    Lagrange-Legendre mesh for the Bloch-Schroedinger equation following:
    Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel; e.g.
    r -> s_i = r * k_i, and each channel's equation is then divided by it's
    asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(self, nbasis, nchannels, abscissa, weights):
        """
        Constructs the Bloch-Schroedinger equation in a basis of nbasis
        Lagrange-Legendre functions shifted and scaled onto [0,k*a] and regulated by 1/k*r,
        and solved by direct matrix inversion.

        """
        self.nbasis = nbasis
        self.nchannels = nchannels
        self.abscissa = abscissa
        self.weights = weights
        self.I = np.identity(self.nbasis * self.nchannels)

    def local_potential(self, n, m, interaction, ch: ChannelData, args=()):
        """
        evaluates the (n,m)th matrix element for the given local interaction
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        if n != m:
            return 0  # local potentials are diagonal

        xn = self.abscissa[n - 1]

        a = ch.domain[1]
        s = xn * a

        return eval_scaled_interaction(s, interaction, ch, args)

    def nonlocal_potential(self, n, m, interaction, ch, args=()):
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

        utilde = eval_scaled_nolocal_interaction(s, sp, interaction, ch, args)

        return utilde * np.sqrt(wm * wn) * a

    def kinetic_bloch(self, n, m, ch):
        """
        evaluates the (n,m)th matrix element for the kinetic energy + Bloch operator
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        N = self.nbasis

        a = ch.domain[1]
        l = ch.l

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

    def bloch_se_matrix(
        self, interaction_matrix: np.array, channel_matrix: np.array, args=()
    ):
        assert interaction_matrix.shape == channel_matrix.shape
        assert interaction_matrix.shape[0] == interaction_matrix[1]
        assert interaction_matrix.shape[0] == self.nchannels

        sz = self.nbasis * self.nchannels
        C = np.zeros((sz, sz), dtype=np.cdouble)
        for i in range(self.nchannels):
            for j in range(self.nchannels):
                C[
                    i * self.nbasis : i * self.nbasis + self.nbasis,
                    j * self.nbasis : j * self.nbasis + self.nbasis,
                ] = self.single_channel_bloch_se_matrix(
                    i,
                    j,
                    interaction_matrix.matrix[i, j],
                    channel_matrix[i, j],
                    is_local,
                    is_symmetric,
                    args,
                )
        return C

    def single_channel_bloch_se_matrix(
        self,
        i,
        j,
        interaction,
        ch: ChannelData,
        is_local: bool = True,
        is_symmetric: bool = True,
        args=(),
    ):
        # TODO handle case of local plus non-local potential in a given channel (e.g. non-local plus coulomb)
        C = np.zeros((self.nbasis, self.nbasis), dtype=np.cdouble)
        # Eq. 6.10 in [Baye, 2015], scaled by 1/E and with r->s=kr
        # diagonal submatrices in channel space
        # include full bloch-SE

        if is_local:
            potential = lambda n, m: self.local_potential(n, m, interaction, ch, args)
        else:
            potential = lambda n, m: self.nonlocal_potential(
                n, m, interaction, ch, args
            )

        if i == j:
            element = lambda n, m: (
                self.kinetic_bloch(n, m, ch, args)
                + potential(n, m)
                - (1.0 if n == m else 0.0)
            )

        # off-diagonal terms only include coupling potentials
        else:
            element = lambda n, m: self.potential(n, m, interaction, ch)

        # build matrix for channel
        if is_symmetric:
            for n in range(1, self.nbasis + 1):
                for m in range(n, self.nbasis + 1):
                    C[n - 1, m - 1] = element(n, m)

            C = C + (C - np.diag(C)).Ts

        else:
            for n in range(1, self.nbasis + 1):
                for m in range(1, self.nbasis + 1):
                    C[n - 1, m - 1] = element(n, m)

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
    Returns the R-Matrix, S-matrix wavefunction coefficients, all in Lagrange-Legendre coordinates,
    and derivative of asymptotic channel Wavefunctions evaluated at the channel radius. all as block-
    matrices and block vectors in channel space. Uses `numpy.linalg.solve`, which should be faster
    than alternatives (e.g. scipy) for mid-size (n<100) Hermitian matrices. Note: speed will be
    dependent on numpy backend for linear algebra.

    This follows:
    Descouvemont, P. (2016).
    An R-matrix package for coupled-channel problems in nuclear physics.
    Computer physics communications, 200, 199-219.
    """
    (Hp, Hm, Hpp, Hmp) = asymptotics

    # Eqn 15 in Descouvemont, 2016
    x = np.linalg.solve(A, b).reshape(nchannels, nbasis)
    R = x @ b.reshape(nchannels, nbasis).T

    # Eqn 17 in Descouvemont, 2016
    Zp = (Hp * incoming_weights - R * Hpp[:, np.newaxis] * a[:, np.newaxis]) / np.sqrt(
        a[:, np.newaxis]
    )
    Zm = (Hm * incoming_weights - R * Hmp[:, np.newaxis] * a[:, np.newaxis]) / np.sqrt(
        a[:, np.newaxis]
    )

    # Eqn 16 in Descouvemont, 2016
    S = np.linalg.solve(Zp, Zm)

    # TODO should there be a factor of k/sqrt(v) here?
    uext_prime_asym = Hmp * incoming_weights - S @ Hpp

    return R, S, x, uext_prime_asym
