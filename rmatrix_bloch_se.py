from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from mpmath import coulombf, coulombg
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn, eval_legendre, roots_legendre
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.linalg import solve, ishermitian

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
    return (H_minus(a, l, eta) - a * Rl * H_minus_prime(a, l, eta)) / (
        H_plus(a, l, eta) - a * Rl * H_plus_prime(a, l, eta)
    )


def delta(Sl):
    """
    returns the phase shift and attentuation factor in degrees
    """
    delta = np.log(Sl) / 2.0j  # complex phase shift in radians
    return np.rad2deg(np.real(delta)), np.rad2deg(np.imag(delta))


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


@dataclass
class ProjectileTargetSystem:
    """
    Channel agnostic data for a projectile target system
    """

    incident_energy: float  # [MeV]
    reduced_mass: float  # [MeV]
    channel_radius: float  # [dimensionless]
    Ztarget: float = 0
    Zproj: float = 0
    num_channels: int = 1
    level_energies: list[float] = field(default_factory=list)


class RadialSEChannel:
    """
    Implements a single-channel radial schrodinger eqn for a local interaction in the
    (scaled) coordinate (r) basis with s = k * r for wavenumber k in [fm^-1] and r in [fm]
    """

    def __init__(
        self,
        l: int,
        system: ProjectileTargetSystem,
        interaction,
        coulomb_interaction=None,
        threshold_energy: float = 0,
    ):
        """
        arguments:
        l -- orbital angular momentum quantum number in the channel
        threshold_energy -- energy threshold of channel in MeV
        system -- basic information about projectile target system
        interaction -- callable that takes projectile-target distance [fm] and returns energy in [MeV]
        coulomb_interaction -- same as interaction, but takes two arguments: the sommerfield parameter, projectile-target distance [fm])
        """
        self.is_local = True
        self.l = l
        self.threshold_energy = threshold_energy
        self.system = system

        self.Zzprod = system.Zproj * system.Ztarget
        self.E = system.incident_energy - threshold_energy
        self.mass = system.reduced_mass
        self.k = np.sqrt(2 * self.mass * self.E) / hbarc
        self.eta = (alpha * self.Zzprod) * self.mass / (hbarc * self.k)
        self.a = system.channel_radius

        self.domain = [1.0e-10, self.a]

        if interaction is not None:
            self.interaction = interaction
            self.Vscaled = lambda s: interaction(s / self.k) / self.E

        if coulomb_interaction is not None:
            self.coulomb_interaction = coulomb_interaction
            self.VCoulombScaled = (
                lambda s: coulomb_interaction(self.Zzprod, s / self.k) / self.E
            )
            assert self.eta > 0
        else:
            self.VCoulombScaled = lambda s: 0.0
            self.eta = 0

    def second_derivative(self, s, u):
        return (
            self.Vscaled(s)  # 2-body nuclear potential
            + self.VCoulombScaled(s)  # Coulomb interaction
            + self.l * (self.l + 1) / s**2  # orbital angular momentum
            - 1.0  # energy term
        ) * u

    def initial_conditions(self):
        """
        initial conditions for numerical integration in coordinate (s) space
        """
        s_0 = self.domain[0]
        l = self.l
        C_l = Gamow_factor(l, self.eta)
        rho_0 = (s_0 / C_l) ** (1 / (l + 1))
        u0 = C_l * rho_0 ** (l + 1)
        uprime0 = C_l * (l + 1) * rho_0**l
        return np.array([u0 * (1 + 0j), uprime0 * (1 + 0j)])

    def s_grid(self, size=200):
        return np.linspace(self.domain[0], self.domain[1], size)


class NonlocalRadialSEChannel(RadialSEChannel):
    """
    Implements a single-channel radial schrodinger eqn for a nonlocal interaction in the
    (scaled) coordinate (r,rp) basis with s = k * r for wavenumber k in [fm^-1] and r in [fm]
    """

    def __init__(
        self,
        l: int,
        system: ProjectileTargetSystem,
        interaction,
        coulomb_interaction=None,
        threshold_energy: float = 0.0,
    ):
        super().__init__(l, system, None, coulomb_interaction, threshold_energy)
        self.interaction = interaction
        self.Vscaled = lambda s, sp: interaction(s / self.k, sp / self.k) / self.E
        self.is_local = False

    def second_derivative(self, s, u, grid_size=200):
        raise NotImplementedError("Not implemented for non-local potentials")


def schrodinger_eqn_ivp_order1(s, y, radial_se):
    """
    callable for scipy.integrate.solve_ivp; converts SE to
    2 coupled 1st order ODEs
    """
    u, uprime = y
    return [uprime, radial_se.second_derivative(s, u)]


class Wavefunction:
    """
    Represents a wavefunction, expressed internally to the channel as a linear combination
    of Lagrange-Legendre functions, and externally as a linear combination of incoming and
    outgoing Coulomb scattering wavefunctions
    """

    def __init__(self, lm, coeffs, S, se, norm_factor, is_entrance_channel=False):
        self.is_entrance_channel = is_entrance_channel
        self.lm = lm
        self.se = se
        self.norm_factor = norm_factor

        self.coeffs = np.copy(coeffs)
        self.S = np.copy(S)

        self.callable = self.u()

    def __call__(self, r):
        return self.callable(r)

    def calculate_s(self, units):
        if units == "r":
            return lambda r : r * self.se.k
        elif units == "s":
            return lambda s : s

    def uext(self, units="r"):
        out = lambda r: np.array(
            self.S * VH_plus(self.calculate_s(units)(r), self.se.l, self.se.eta), dtype=complex
        )
        if self.is_entrance_channel:
            return lambda r: np.array(
                VH_minus(self.calculate_s(units)(r), self.se.l, self.se.eta) + out(r), dtype=complex
            )
        else:
            return out

    def uint(self, units="r"):
        return lambda r: np.sum(
            [
                self.coeffs[n - 1] * self.lm.f(n, self.calculate_s(units)(r))
                for n in range(1, self.lm.N + 1)
            ],
            axis=0,
        )

    def u(self, units="r"):
        ch_radius = self.se.a / self.se.k
        uint = self.uint(units)
        uext = self.uext(units)
        factor = 1.0
        return lambda r: np.where(r < ch_radius, uint(r), factor * uext(r))


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
            norm_factor = self.uext_prime_boundary(S,i) * self.se[i,i].a * R[i,i]
            wavefunctions.append(
                Wavefunction(
                    self, coeffs[i], S[i,0], self.se[i, i], norm_factor, is_entrance_channel=(i == 0)
                )
            )

        if not self.coupled_channels:
            wavefunctions = wavefunctions[0]
            S = S[0, 0]
            R = R[0, 0]

        return R, S, wavefunctions


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


def nonlocal_interaction_example():
    alpha = 0.2316053  # fm**-1
    beta = 1.3918324  # fm**-1
    W0 = 41.472  # Mev fm**2

    params = (W0, beta, alpha)
    hbarc = 197.3  # MeV fm
    mass = hbarc**2 / (2 * W0)

    sys = ProjectileTargetSystem(
        incident_energy=0.1, reduced_mass=mass, channel_radius=20
    )

    se = NonlocalRadialSEChannel(
        l=0, system=sys, interaction=lambda r, rp: yamaguchi_potential(r, rp, params)
    )

    nbasis = 20
    solver_lm = LagrangeRMatrix(nbasis, sys, se)

    R, S, _ = solver_lm.solve()
    delta = np.rad2deg(np.real(np.log(S) / 2j))

    print("\nYamaguchi potential test:\n delta:")
    print("Lagrange-Legendre Mesh: {:.6f} [degrees]".format(delta))
    print(
        "Analytic              : {:.6f} [degrees]".format(
            yamaguchi_swave_delta(se.k, params)
        )
    )


def coupled_channels_example(visualize=False):
    """
    3 level system example with local diagonal and transition potentials and neutral
    particles. Potentials are real, so S-matrix is unitary and symmetric
    """
    mass = 939  # reduced mass of scattering system MeV / c^2

    # Potential parameters
    V = 60  # real potential strength
    W = 0  # imag potential strength
    R = 4  # Woods-Saxon potential radius
    a = 0.5  # Woods-Saxon potential diffuseness
    params = (V, W, R, a)

    nodes_within_radius = 10

    system = ProjectileTargetSystem(
        incident_energy=50,
        reduced_mass=939,
        channel_radius=5 * (2 * np.pi),
        num_channels=3,
        level_energies=[0, 12, 20],
    )

    l = 0

    matrix = np.empty((3, 3), dtype=object)

    # diagonal potentials are just Woods-Saxons
    for i in range(system.num_channels):
        matrix[i, i] = RadialSEChannel(
            l=l,
            system=system,
            interaction=lambda r: woods_saxon_potential(r, params),
            threshold_energy=system.level_energies[i],
        )

    # transition potentials have depths damped by a factor compared to diagonal terms
    # and use surface peaked Gaussian form factors rather than Woods-Saxons
    transition_dampening_factor = 1
    Vt = V / transition_dampening_factor
    Wt = W / transition_dampening_factor

    # off diagonal potential terms
    for i in range(system.num_channels):
        for j in range(system.num_channels):
            if i != j:
                matrix[i, j] = RadialSEChannel(
                    l=l,
                    system=system,
                    interaction=lambda r: surface_peaked_gaussian_potential(
                        r, (Vt, Wt, R, a)
                    ),
                    threshold_energy=system.level_energies[i],
                )

    solver_lm = LagrangeRMatrix(40, system, matrix)

    H = solver_lm.bloch_se_matrix()

    if visualize:
        for i in range(3):
            for j in range(3):
                plt.imshow(np.real(solver_lm.single_channel_bloch_se_matrix(i, j)))
                plt.xlabel("n")
                plt.ylabel("m")
                plt.colorbar()
                plt.title(f"({i}, {j})")
                plt.show()

    # get R and S-matrix, and both internal and external soln
    R, S, uint = solver_lm.solve_wavefunction()

    # S must be unitary
    assert np.isclose(complex_det(S), 1.0)
    # S is symmetric iff the correct factors of momentum are applied
    # assert np.allclose(S, S.T)

    r_values = np.linspace(0.05, 40, 500)
    s_values = np.linspace(0.05, system.channel_radius, 500)

    lines = []
    for i in range(system.num_channels):
        u_values = uint[i].uint(units="s")(s_values)
        (p1,) = plt.plot(s_values, np.real(u_values), label=r"$n=%d$" % i)
        (p2,) = plt.plot(s_values, np.imag(u_values), ":", color=p1.get_color())
        lines.append([p1, p2])

    legend1 = plt.legend(
        lines[0], [r"$\mathfrak{Re}\, u_n(s) $", r"$\mathfrak{Im}\, u_n(s)$"], loc=3
    )
    plt.legend([l[0] for l in lines], [l[0].get_label() for l in lines], loc=1)
    plt.gca().add_artist(legend1)

    plt.xlabel(r"$s_n = k_n r$")
    plt.ylabel(r"$u (s) $ [a.u.]")
    plt.tight_layout()
    plt.show()


def channel_radius_dependence_test():
    # Potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    a_grid = np.linspace(10, 30, 50)
    delta_grid = np.zeros_like(a_grid, dtype=complex)

    for i, a in enumerate(a_grid):
        sys = ProjectileTargetSystem(
            incident_energy=50, reduced_mass=939, channel_radius=a, Ztarget=60, Zproj=0
        )

        se = RadialSEChannel(
            l=0,
            system=sys,
            interaction=lambda r: woods_saxon_potential(r, params),
            coulomb_interaction=lambda eta, r: np.vectorize(coulomb_potential)(
                zz, r, R0
            ),
        )

        solver_lm = LagrangeRMatrix(40, sys, se)

        R_lm, S_lm, G = solver_lm.solve()
        delta_lm, atten_lm = delta(S_lm)

        delta_grid[i] = delta_lm + 1.0j * atten_lm

    plt.plot(a_grid, np.real(delta_grid), label=r"$\mathfrak{Re}\,\delta_l$")
    plt.plot(a_grid, np.imag(delta_grid), label=r"$\mathfrak{Im}\,\delta_l$")
    plt.legend()
    plt.xlabel("channel radius [fm]")
    plt.ylabel(r"$\delta_l$ [degrees]")
    plt.show()


def local_interaction_example():
    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    nodes_within_radius = 5

    sys = ProjectileTargetSystem(
        incident_energy=20,
        reduced_mass=939,
        channel_radius=nodes_within_radius * (2 * np.pi),
        Ztarget=40,
        # Zproj=1
    )

    se = RadialSEChannel(
        l=1,
        system=sys,
        interaction=lambda r: woods_saxon_potential(r, params),
        # coulomb_interaction=lambda zz, r: np.vectorize(coulomb_potential)(zz, r, R0)
    )

    s_values = np.linspace(0.01, sys.channel_radius,200)
    r_values = s_values / se.k


    # Runge-Kutta
    sol_rk = solve_ivp(
        lambda s, y,: schrodinger_eqn_ivp_order1(s, y, se),
        se.domain,
        se.initial_conditions(),
        dense_output=True,
        atol=1.0e-12,
        rtol=1.0e-9,
    ).sol

    u_rk = sol_rk(s_values)[0]
    R_rk = sol_rk(se.a)[0] / (se.a * sol_rk(se.a)[1])
    S_rk = smatrix(R_rk, se.a, se.l, se.eta)

    # Lagrange-Mesh
    solver_lm = LagrangeRMatrix(40, sys, se)

    R_lm, S_lm, u_lm = solver_lm.solve_wavefunction()
    # R_lmp = u_lm(se.a) / (se.a * derivative(u_lm, se.a, dx=1.0e-6))

    u_lm = u_lm(r_values)

    delta_lm, atten_lm = delta(S_lm)
    delta_rk, atten_rk = delta(S_rk)

    # normalization and phase matching
    u_rk = u_rk * np.max(np.real(u_lm)) / np.max(np.real(u_rk)) * (-1j)

    print(f"k: {se.k}")
    print(f"R-Matrix RK: {R_rk:.3e}")
    print(f"R-Matrix LM: {R_lm:.3e}")
    # print(f"R-Matrix LMp: {R_lmp:.3e}")
    print(f"S-Matrix RK: {S_rk:.3e}")
    print(f"S-Matrix LM: {S_lm:.3e}")
    print(f"real phase shift RK: {delta_rk:.3e} degrees")
    print(f"real phase shift LM: {delta_lm:.3e} degrees")
    print(f"complex phase shift RK: {atten_rk:.3e} degrees")
    print(f"complex phase shift LM: {atten_lm:.3e} degrees")

    plt.plot(r_values, np.real(u_rk), "k", label="Runge-Kutta")
    plt.plot(r_values, np.imag(u_rk), ":k")

    plt.plot(r_values, np.real(u_lm), "r", label="Lagrange-Legendre")
    plt.plot(r_values, np.imag(u_lm), ":r")

    plt.legend()
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$u_{%d} (r) $ [a.u.]" % se.l)
    plt.tight_layout()
    plt.show()


def rmse_RK_LM():
    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    nodes_within_radius = 5

    lgrid = np.arange(0, 10)
    egrid = np.linspace(0.01, 100, 100)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for l in lgrid:
        for i, e in enumerate(egrid):
            sys = ProjectileTargetSystem(
                incident_energy=e,
                reduced_mass=939,
                channel_radius=nodes_within_radius * (2 * np.pi),
                Ztarget=40,
                Zproj=1,
            )

            se = RadialSEChannel(
                l=l,
                system=sys,
                interaction=lambda r: woods_saxon_potential(r, params),
                coulomb_interaction=lambda zz, r: np.vectorize(coulomb_potential)(
                    zz, r, R0
                ),
            )

            # Runge-Kutta
            sol_rk = solve_ivp(
                lambda s, y,: schrodinger_eqn_ivp_order1(s, y, se),
                se.domain,
                se.initial_conditions(),
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            R_rk = sol_rk(se.a)[0] / (se.a * sol_rk(se.a)[1])
            S_rk = smatrix(R_rk, se.a, se.l, se.eta)

            # Lagrange-Mesh
            solver_lm = LagrangeRMatrix(40, sys, se)

            R_lm, S_lm, _ = solver_lm.solve()

            delta_lm, atten_lm = delta(S_lm)
            delta_rk, atten_rk = delta(S_rk)

            err = 0 + 0j

            if np.fabs(delta_rk) > 1e-12:
                err += np.fabs(delta_lm - delta_rk)

            if np.fabs(atten_rk) > 1e-12:
                err += 1j * np.fabs(atten_lm - atten_rk)

            error_matrix[l, i] = err

    lines = []
    for l in lgrid:
        (p1,) = plt.plot(egrid, np.real(error_matrix[l, :]), label=r"$l = %d$" % l)
        (p2,) = plt.plot(egrid, np.imag(error_matrix[l, :]), ":", color=p1.get_color())
        lines.append([p1, p2])

    plt.ylabel(r"$\Delta \equiv | \delta^{\rm RK} - \delta^{\rm LM} |$ [degrees]")
    plt.xlabel(r"$E$ [MeV]")

    legend1 = plt.legend(
        lines[0], [r"$\mathfrak{Re}\, \Delta$", r"$\mathfrak{Im}\, \Delta$"], loc=0
    )
    plt.legend([l[0] for l in lines], [l[0].get_label() for l in lines], loc=1)
    plt.ylim([0, 1])
    plt.yscale("log")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # channel_radius_dependence_test()
    local_interaction_example()
    # nonlocal_interaction_example()
    coupled_channels_example()
    # rmse_RK_LM()
