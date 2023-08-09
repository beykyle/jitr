from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn, eval_legendre, roots_legendre
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.linalg import solve, ishermitian
from .utils import (
    block,
    H_plus,
    H_plus_prime,
    H_minus,
    H_minus_prime,
    VH_plus,
    VH_minus,
    smatrix,
)
from .bloch_se import (
    ProjectileTargetSystem,
    RadialSEChannel,
    NonlocalRadialSEChannel,
    Wavefunction,
)


class LagrangeRMatrix:
    """
    Lagrange-Legendre mesh for the Bloch-Schroedinger equation following:
    Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel; e.g.
    r -> s_i = r * k_i, and each channel's equation is then divided by it's
    asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(self, nbasis, system: ProjectileTargetSystem, radial_se):
        """
        Constructs the Bloch-Schroedinger equation in a basis of nbasis
        Lagrange-Legendre functions shifted and scaled onto [0,k*a] and regulated by 1/k*r,
        and solved by direct matrix inversion.

        arguments:
        nbasis -- number of Lagrange-Legendre functions to project each channel's B-S eqn onto
        system -- contains basic info about projectile target system
        radial_se -- contains information on the interaction in each channel, either a derived type from RadialSEChannel, or a square matrix of the same. If the latter, then a coupled channel calculation is done, with the diagonal terms in radial_se giving the projectile target interaction in each channel, and the off-diagonal terms giving the coupling in each channel.
        """
        self.N = nbasis
        self.system = system
        self.se = radial_se

        if self.system.Zproj * self.system.Ztarget == 0:
            self.coulomb_potential = lambda n, m, i=0, j=0: 0

        if self.system.num_channels > 1:
            # interaction should take the form of a square matrix
            assert isinstance(self.se, (np.ndarray, np.generic))
            assert radial_se.shape == (
                self.system.num_channels,
                self.system.num_channels,
            )

            self.coupled_channels = True

        else:
            if not isinstance(self.se, (np.ndarray, np.generic)):
                self.se = np.array([[self.se]])
            assert self.se.shape == (1, 1)

            self.coupled_channels = False

        # generate Lagrange-Legendre quadrature and weights shifted to [0,1] from [-1,1]
        x, w = roots_legendre(self.N)
        self.abscissa = 0.5 * (x + 1)
        self.weights = 0.5 * w

    def plot_basis(self, i=0):
        s = self.se[i, i].s_grid()
        for n in range(1, self.N + 1):
            plt.plot(s, self.f(n, s, i), label=f"$n={n}$")

        plt.legend()
        plt.xlabel(r"$s$")
        plt.xlabel(r"$f_n(s)$")
        plt.tight_layout()
        plt.show()

    def f(self, n, s, i=0):
        """
        nth basis function in channel i - Lagrange-Legendre polynomial of degree n shifted onto
        [0,a_i] and regularized by s/( a_i * xn)
        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.N and n >= 1

        a = self.se[i, i].a
        x = s / a
        xn = self.abscissa[n - 1]

        # Eqn 3.122 in [Baye, 2015], with s = kr
        return (
            (-1.0) ** (self.N - n)
            * np.sqrt((1 - xn) / xn)
            * eval_legendre(self.N, 2.0 * x - 1.0)
            * x
            / (x - xn)
        )

    def coulomb_potential(self, n, m, i=0, j=0):
        """
        evaluates the (n,m)th matrix element for the Coulomb interaction in
        the (i,j)th channel
        """
        assert n <= self.N and n >= 1
        assert m <= self.N and m >= 1

        if n != m:
            return 0  # local potentials are diagonal

        xn = self.abscissa[n - 1]
        a = self.se[i, j].a

        return self.se[i, j].VCoulombScaled(xn * a)

    def potential(self, n, m, i=0, j=0):
        """
        Evaluates the (n,m)th matrix element for the potential in the (i,j)th channel
        """
        se = self.se[i, j]

        if se.is_local:
            return self.local_potential(n, m, i, j)
        else:
            return self.nonlocal_potential(n, m, i, j)

    def local_potential(self, n, m, i=0, j=0):
        """
        evaluates the (n,m)th matrix element for the given local interaction
        in the (i,j)th channel
        """
        assert n <= self.N and n >= 1
        assert m <= self.N and m >= 1

        if n != m:
            return 0  # local potentials are diagonal

        xn = self.abscissa[n - 1]
        a = self.se[i, j].a

        return self.se[i, j].Vscaled(xn * a)

    def nonlocal_potential(self, n, m, i=0, j=0):
        """
        evaluates the (n,m)th matrix element for the given non-local interaction
        in the (i,j)th channel
        """
        assert n <= self.N and n >= 1
        assert m <= self.N and m >= 1

        xn = self.abscissa[n - 1]
        xm = self.abscissa[m - 1]
        wn = self.weights[n - 1]
        wm = self.weights[m - 1]

        a = self.se[i, j].a

        return self.se[i, j].Vscaled(xn * a, xm * a) * np.sqrt(wm * wn) * a

    def kinetic_bloch(self, n, m, i=0, j=0):
        """
        evaluates the (n,m)th matrix element for the kinetic energy + Bloch operator
        in the (i,j)th channel
        """
        assert n <= self.N and n >= 1
        assert m <= self.N and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        k = self.se[i, j].k
        l = self.se[i, j].l
        a = self.se[i, j].a
        N = self.N

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

    def bloch_se_matrix(self):
        sz = self.N * self.system.num_channels
        C = np.zeros((sz, sz), dtype=complex)
        for i in range(self.system.num_channels):
            for j in range(self.system.num_channels):
                C[
                    i * self.N : i * self.N + self.N, j * self.N : j * self.N + self.N
                ] = self.single_channel_bloch_se_matrix(i, j)
        return C

    def single_channel_bloch_se_matrix(self, i=None, j=None):
        C = np.zeros((self.N, self.N), dtype=complex)
        # TODO  use symmetry to calculate more efficiently
        # Eq. 6.10 in [Baye, 2015], scaled by 1/E and with r->s=kr

        # diagonal submatrices in channel space
        # include full bloch-SE
        if i == j:
            element = lambda n, m: (
                self.kinetic_bloch(n, m, i, j)
                + self.potential(n, m, i, j)
                + self.coulomb_potential(n, m, i, j)
                - (1.0 if n == m else 0.0)
            )
        # off-diagonal terms only include coupling potentials
        else:
            element = lambda n, m: self.potential(n, m, i, j)

        for n in range(1, self.N + 1):
            for m in range(1, self.N + 1):
                C[n - 1, m - 1] = element(n, m)

        return C

    def solve(self):
        """
        Returns the R-Matrix and the S-matrix, as well as the Green's function in Lagrange-Legendre
        coordinates

        For the coupled-channels case this follows:
        Descouvemont, P. (2016).
        An R-matrix package for coupled-channel problems in nuclear physics.
        Computer physics communications, 200, 199-219.
        """
        A = self.bloch_se_matrix()

        a = self.se[0, 0].a
        l = self.se[0, 0].l
        eta = self.se[0, 0].eta

        if not self.coupled_channels:
            # Eq. 6.11 in [Baye, 2015]
            b = np.array([self.f(n, a) for n in range(1, self.N + 1)])
            G = np.linalg.inv(A)
            x = np.dot(G, b)
            R = np.dot(x, b) / (a * a)
            S = smatrix(R, a, l, eta)
            return R, S, G
        else:
            # TODO formulate all of this as solving a linear system
            # rather than an inversion
            ach = [se.a for se in np.diag(self.se)]

            # source term - basis functions evaluated at each channel radius
            b = np.concatenate(
                [np.array([self.f(n, a) for n in range(1, self.N + 1)]) for a in ach]
            )

            # find Green's function explicitly in Lagrange-Legendre coords
            G = np.linalg.inv(A)

            # calculate R-matrix
            # Eq. 15 in [Descouvemont, 2016]
            R = np.zeros(
                (self.system.num_channels, self.system.num_channels), dtype=complex
            )
            for i in range(self.system.num_channels):
                for j in range(self.system.num_channels):
                    a = self.se[i, i].a
                    submatrix = block(G, (i, j), (self.N, self.N))
                    b1 = b[i * self.N : i * self.N + self.N]
                    b2 = b[j * self.N : j * self.N + self.N]
                    R[i, j] = np.dot(b1, np.dot(submatrix, b2)) / (a * a)

            # calculate collision matrix (S-matrix)
            # Eqns. 16 and 17 in [Descouvemont, 2016]
            Hm = np.diag(
                np.ones(self.system.num_channels) * H_minus(a, l, eta),
            )
            Hp = np.diag(
                np.ones(self.system.num_channels) * H_plus(a, l, eta),
            )

            Z_minus = Hm - a * R * H_minus_prime(a, l, eta)
            Z_plus = Hp - a * R * H_plus_prime(a, l, eta)
            S = solve(Z_plus, Z_minus)

            return R, S, G

    def uext_prime_boundary(self, S, i=0):
        out = S[i, 0] * H_plus_prime(
            self.se[i, i].a, self.se[i, i].l, self.se[i, i].eta
        )
        if i == 0:
            return (
                H_minus_prime(self.se[i, i].a, self.se[i, i].l, self.se[i, i].eta) + out
            )
        else:
            return out

    def solve_wavefunction(self):
        """
        Solves the system and returns a callable for the reduced radial wavefunctions in each channel.
        If there is just one channel, simply returns the callable, otherwise returns a list of
        callables.
        """
        R, S, G = self.solve()

        if not self.coupled_channels:
            S = np.array([[S]])
            R = np.array([[R]])

        b = np.concatenate(
            [
                np.array(
                    [
                        (self.f(n, self.se[j, j].a) * self.uext_prime_boundary(S, j))
                        for n in range(1, self.N + 1)
                    ]
                )
                for j in range(self.system.num_channels)
            ]
        )
        coeffs = np.split(np.dot(G, b), self.system.num_channels)

        wavefunctions = []
        for i in range(self.system.num_channels):
            norm_factor = self.uext_prime_boundary(S, i) * self.se[i, i].a * R[i, i]
            wavefunctions.append(
                Wavefunction(
                    self,
                    coeffs[i],
                    S[i, 0],
                    self.se[i, i],
                    norm_factor,
                    is_entrance_channel=(i == 0),
                )
            )

        if not self.coupled_channels:
            wavefunctions = wavefunctions[0]
            S = S[0, 0]
            R = R[0, 0]

        return R, S, wavefunctions
