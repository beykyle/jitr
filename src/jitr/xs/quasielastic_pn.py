from scipy.special import sph_harm, gamma
from sympy.physics.wigner import clebsch_gordan

import numpy as np

from ..utils import constants
from ..utils.kinematics import (
    ChannelKinematics,
    mass,
    get_AME_binding_energy,
    classical_kinematics,
    classical_kinematics_cm,
)
from ..reactions import (
    spin_half_orbit_coupling,
    ProjectileTargetSystem,
)
from ..rmatrix import Solver

# TODO
# test using the coefficients not the wavefunctions for the dwba t-matrix
# test doing DWBA T-matrix in s-space not r-space


def kinematics(target: tuple, analog: tuple, Elab: np.float64, Ex_IAS: np.float64):
    mass_target = mass(*target)
    mass_analog = mass(*analog)
    mn = mass(1, 0)
    mp = mass(1, 1)
    BE_target = get_AME_binding_energy(*target)
    BE_analog = get_AME_binding_energy(*analog)
    Q = BE_analog - BE_target - Ex_IAS
    CDE = 1.33 * (target[1] + analog[1]) * 0.5 / target[0] ** (1.0 / 3.0)
    kinematics_entrance = classical_kinematics(mass_target, mp, Elab, Zz=target[1])
    Ecm_exit = kinematics_entrance.Ecm + Q
    kinematics_exit = classical_kinematics_cm(mass_analog, mn, Ecm_exit)
    return kinematics_entrance, kinematics_exit, Q, CDE


class System:
    r"""
    Stores physics parameters of system. Calculates useful parameters for each partial wave.
    """

    def __init__(
        self,
        channel_radius: np.float64,
        lmax: np.int64,
        target: tuple,
        analog: tuple,
        mass_target: np.float64,
        mass_analog: np.float64,
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
        self.mass_p = mass(1, 1)
        self.mass_n = mass(1, 0)
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


class Workspace:
    r"""
    Workspace for (p,n) quasi-elastic scattering observables for local interactions
    """

    def __init__(
        self,
        sys: System,
        kinematrics_entrance: ChannelKinematics,
        kinematrics_exit: ChannelKinematics,
        entrance_interaction_scalar,
        entrance_interaction_spin_orbit,
        exit_interaction_scalar,
        exit_interaction_spin_orbit,
        solver: Solver,
        angles: np.array,
        tmatrix_abs_tol: np.float64 = 1e-6,
    ):
        assert np.all(np.diff(angles) > 0)
        assert angles[0] >= 0.0 and angles[-1] <= np.pi
        self.sys = sys
        A = self.sys.target[0]
        Z = self.sys.target[1]
        N = A - Z
        self.isovector_factor = np.sqrt(np.fabs(N - Z)) / (N - Z - 1)
        self.kinematrics_entrance = kinematrics_entrance
        self.kinematrics_exit = kinematrics_exit
        self.solver = solver
        self.tmatrix_abs_tol = tmatrix_abs_tol

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
            (2, 2, self.sys.lmax, 2, self.angles.shape[0]), dtype=np.complex128
        )
        self.sigma_c = np.angle(
            gamma(1 + self.sys.l + 1j * self.kinematrics_entranc.eta)
        )
        for im, mu in enumerate([-0.5, 0.5]):
            for imp, mu_pr in enumerate([-0.5, 0.5]):
                for l in range(0, self.sys.lmax):
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

    def tmatrix(
        self,
        args_entrance_scalar=None,
        args_entrance_spin_orbit=None,
        args_exit_scalar=None,
        args_exit_spin_orbit=None,
    ):
        Tpn = np.zeros((2, self.sys.lmax), dtype=np.complex128)
        Sn = np.zeros((2, self.sys.lmax), dtype=np.complex128)
        Sp = np.zeros((2, self.sys.lmax), dtype=np.complex128)

        # precomute scalar and spin-obit interaction matrices for distorted wave solns
        im_scalar_p = self.solver.interaction_matrix(
            self.p_channels[0][0],
            local_interaction=self.entrance_interaction_scalar,
            local_args=args_entrance_scalar,
        )
        im_spin_orbit_p = self.solver.interaction_matrix(
            self.p_channels[0][0],
            local_interaction=self.entrance_interaction_spin_orbit,
            local_args=args_entrance_spin_orbit,
        )
        im_scalar_n = self.solver.interaction_matrix(
            self.n_channels[0][0],
            local_interaction=self.exit_interaction_scalar,
            local_args=args_exit_scalar,
        )
        im_spin_orbit_n = self.solver.interaction_matrix(
            self.n_channels[0][0],
            local_interaction=self.exit_interaction_spin_orbit,
            local_args=args_exit_spin_orbit,
        )

        # precompute isovector part by subtracting entrance and exit potentials
        im_scalar_isovector = -(im_scalar_n - im_scalar_p) * self.isovector_factor
        im_spin_orbit_isovector = (
            -(im_spin_orbit_n - im_spin_orbit_p) * self.isovector_factor
        )
        w = self.solver.kernel.quadrature.weights

        def tmatrix_element(l, ji):
            nch = self.n_channels[l]
            pch = self.p_channels[l]
            nasym = self.n_asymptotics[l]
            pasym = self.p_asymptotics[l]
            lds = self.l_dot_s[l - 1]  # starts from 1 not 0

            _, snlj, xn, _ = self.solver.solve(
                nch[ji],
                nasym[ji],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_scalar_n + lds * im_spin_orbit_n,
                basis_boundary=self.basis_boundary,
                wavefunction=True,
            )
            _, splj, xp, _ = self.solver.solve(
                pch[ji],
                pasym[ji],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_scalar_p + lds * im_spin_orbit_p,
                basis_boundary=self.basis_boundary,
                wavefunction=True,
            )
            v1 = np.diag(im_scalar_isovector) + lds * np.diag(im_spin_orbit_isovector)
            tlj = np.sum(xp * v1 * xn * w) * self.sys.channel_radius
            return tlj, snlj, splj

        # S-wave
        Tpn[0, 0], Sn[0, 0], Sp[0, 0] = tmatrix_element(0, 0)

        # higher partial waves
        for l in self.sys.l[1:]:
            Tpn[l, 0], Sn[l, 0], Sp[l, 0] = tmatrix_element(l, 0)
            Tpn[l, 1], Sn[l, 1], Sp[l, 1] = tmatrix_element(l, 1)

            if Tpn[l, 1] < self.tmatrix_abs_tol and Tpn[l, 1] < self.tmatrix_abs_tol:
                break

        return Tpn, Sn, Sp

    def xs(self, args_entrance=None, args_exit=None):
        T = np.zeros((2, 2, self.angles.shape[0]), dtype=np.complex128)
        Tpn = self.tmatrix(args_entrance, args_exit)
        for im, mu in enumerate([-0.5, 0.5]):
            for imp, mu_pr in enumerate([-0.5, 0.5]):
                # sum over partial waves
                for l in range(0, self.sys.lmax):
                    for ijp, jp in enumerate([l - 0.5, l + 0.5]):
                        if abs(mu - mu_pr) <= l and jp >= 0:
                            T[im, imp, :] += (
                                self.geometric_factor[im, imp, l, ijp, :] * Tpn[ijp, l]
                            )

        return self.xs_factor * 10 * np.sum(np.absolute(T) ** 2, axis=(0, 1))
