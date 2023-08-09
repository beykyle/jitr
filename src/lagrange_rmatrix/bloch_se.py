from dataclasses import dataclass, field
import numpy as np
from mpmath import coulombf, coulombg

from .utils import Gamow_factor, VH_plus, VH_minus, alpha, hbarc


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
            return lambda r: r * self.se.k
        elif units == "s":
            return lambda s: s

    def uext(self, units="r"):
        out = lambda r: np.array(
            self.S * VH_plus(self.calculate_s(units)(r), self.se.l, self.se.eta),
            dtype=complex,
        )
        if self.is_entrance_channel:
            return lambda r: np.array(
                VH_minus(self.calculate_s(units)(r), self.se.l, self.se.eta) + out(r),
                dtype=complex,
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
