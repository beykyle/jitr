from numba import njit
from dataclasses import astuple, dataclass
from scipy.special import eval_legendre, lpmv, gamma, sph_harm
from sympy.physics.wigner import clebsch_gordan

import numpy as np

from ..utils import constants, kinematics
from ..reactions import (
    spin_half_orbit_coupling,
    ProjectileTargetSystem,
    ChannelKinematics,
)
from ..rmatrix import Solver


class QuasielasticPNSystem:
    r"""
    Stores physics parameters of system. Calculates useful parameters for each partial wave.
    """

    def __init__(
        self,
        channel_radius: np.float64,
        lmax: np.int64,
        target: tuple,
        analog: tuple,
        Elab: np.float64,
        Q_IAS: np.float64,
        entrance_kinematics: ChannelKinematics,
        exit_kinematics: ChannelKinematics,
        mass_target: np.float64 = 0,
        mass_analog: np.float64 = 0,
    ):
        r"""
        @params
            channel_radius (np.float64):  dimensionless channel radius k_0 * radius
        """

        self.channel_radius = channel_radius
        self.lmax = lmax
        self.target = target
        self.analog = analog
        self.mass_target = mass_target
        self.mass_analog = mass_analog
        self.mass_p = kinematics.mass(1, 1)
        self.mass_n = kinematics.mass(1, 0)
        self.Q_IAS = Q_IAS
        self.l = np.arange(0, lmax + 1, dtype=np.int64)

        self.entrance = ProjectileTargetSystem(
            channel_radius=self.channel_radius,
            lmax=self.lmax,
            mass_target=self.mass_target,
            mass_projectile=self.mass_p,
            Ztarget=self.target[1],
            Zproj=1,
            coupling=spin_half_orbit_coupling,
        )

        self.exit = ProjectileTargetSystem(
            channel_radius=self.channel_radius,
            lmax=self.lmax,
            mass_target=self.mass_analog,
            mass_projectile=self.mass_n,
            Ztarget=self.analog[1],
            Zproj=0,
            coupling=spin_half_orbit_coupling,
        )


class ChexPNWorkspace:
    r"""
    Workspace for (p,n) quasi-elastic scattering observables for local interactions
    """

    def __init__(
        self,
        sys: QuasielasticPNSystem,
        kinematrics_entrance: ChannelKinematics,
        kinematrics_exit: ChannelKinematics,
        entrance_interaction_scalar,
        entrance_interaction_spin_orbit,
        exit_interaction_scalar,
        exit_interaction_spin_orbit,
        solver: Solver,
        angles: np.array,
        smatrix_abs_tol: np.float64 = 1e-6,
    ):
        assert np.all(np.diff(angles) > 0)
        assert angles[0] >= 0.0 and angles[-1] <= np.pi
        self.sys = sys
        self.kinematrics_entrance = kinematrics_entrance
        self.kinematrics_exit = kinematrics_exit
        self.solver = solver
        self.smatrix_abs_tol = smatrix_abs_tol

        self.entrance_interaction_scalar = entrance_interaction_scalar
        self.entrance_interaction_spin_orbit = entrance_interaction_spin_orbit
        self.exit_interaction_scalar = exit_interaction_scalar
        self.exit_interaction_spin_orbit = exit_interaction_spin_orbit

        # precompute things
        self.free_matrices = self.solver.free_matrix(
            sys.channel_radius, sys.l, coupled=False
        )
        self.basis_boundary = self.solver.precompute_boundaries(sys.channel_radius)

        # get information for each channel
        channels, asymptotics = sys.entrance.get_partial_wave_channels(
            *self.kinematrics_entrance
        )
        self.p_channels = [ch.decouple() for ch in channels]
        self.p_asymptotics = [asym.decouple() for asym in asymptotics]

        channels, asymptotics = sys.exit.get_partial_wave_channels(
            *self.kinematrics_exit
        )
        self.n_channels = [ch.decouple() for ch in channels]
        self.n_asymptotics = [asym.decouple() for asym in asymptotics]

        # l . s for p-wave and up
        self.l_dot_s = np.array(
            [np.diag(coupling) for coupling in sys.entrance.couplings[1:]]
        )

        # pre-compute purely geometric factors
        self.xs_factor = (
            (self.kinematrics_exit.k / self.kinematrics_entrance.k)
            * self.kinematrics_entrance.mu
            * self.kinematrics_exit.mu
            / (4 * np.pi**2 * constants.HBARC**4 * (2 * 1.0 / 2 + 1))
        )
        self.geometric_factor = np.zeros(
            (2, 2, self.sys.l_max, 2, self.angles.shape[0]), dtype=np.complex128
        )
        self.sigma_c = np.angle(
            gamma(1 + self.sys.l + 1j * self.kinematrics_entranc.eta)
        )
        for im, mu in enumerate([-0.5, 0.5]):
            for imp, mu_pr in enumerate([-0.5, 0.5]):
                for l in range(0, self.sys.l_max):
                    for ijp, jp in enumerate([l - 0.5, l + 0.5]):
                        if abs(mu - mu_pr) <= l and jp >= 0:
                            ylm = sph_harm(mu - mu_pr, l, 0, self.angles)
                            cg0 = clebsch_gordan(l, 1 / 2, jp, mu - mu_pr, mu, mu_pr)
                            cg1 = clebsch_gordan(l, 1 / 2, jp, 0, mu, mu)
                            self.geometric_factor[im, imp, l, ijp, :] = (
                                (4 * np.pi) ** (3.0 / 2.0)
                                / (
                                    self.kinematrics_entrance.k
                                    * self.kinematrics_exit.k
                                )
                                * np.exp(1j * self.sigma_c[l])
                                * cg1
                                * cg0
                                * np.sqrt(2 * l + 1)
                                * (-1) ** (2 * jp + 1)
                                * ylm
                            )

    def tmatrix(self, args_entrance=None, args_exit=None):
        Tpn = np.zeros((2, l_max), dtype=np.complex128)
        Sn = np.zeros((2, l_max), dtype=np.complex128)
        Sp = np.zeros((2, l_max), dtype=np.complex128)

        # precomute scalar and spin-obit interaction matrices for distorted wave solns

        # precompute isovector part by subtracting entrance and exit potentials

        # for each partial wave
        # get coefficients and smatrix elements
        # get pn t matrix

        return Tpn, Sn, Sp

    def xs(self, args_entrance=None, args_exit=None):
        T = np.zeros((2, 2, self.angles.shape[0]), dtype=np.complex128)
        Tpn = self.tmatrix(args_entrance, args_exit)
        for im, mu in enumerate([-0.5, 0.5]):
            for imp, mu_pr in enumerate([-0.5, 0.5]):
                for l in range(0, self.sys.l_max):
                    for ijp, jp in enumerate([l - 0.5, l + 0.5]):
                        if abs(mu - mu_pr) <= l and jp >= 0:
                            T[im, imp, :] += (
                                self.geometric_factor[im, imp, l, ijp, :] * Tpn[ijp, l]
                            )

        return self.xs_factor * 10 * np.sum(np.absolute(T) ** 2, axis=(0, 1))
