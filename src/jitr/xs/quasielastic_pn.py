import pickle
import numpy as np

from scipy.special import sph_harm, gamma
from sympy.physics.wigner import clebsch_gordan

from ..utils import constants
from ..utils.kinematics import (
    ChannelKinematics,
    classical_kinematics,
    classical_kinematics_cm,
)
from ..utils.mass import mass, binding_energy
from ..reactions import (
    spin_half_orbit_coupling,
    ProjectileTargetSystem,
)
from ..rmatrix import Solver


def kinematics(
    target: tuple, analog: tuple, Elab: np.float64, Ex_IAS: np.float64, mass_kwargs={}
):
    mass_target = mass(*target, **mass_kwargs)[0]  # TODO use uncertainties
    mass_analog = mass(*analog, **mass_kwargs)[0]
    mn = constants.MASS_N
    mp = constants.MASS_P
    BE_target = binding_energy(*target, **mass_kwargs)[0]
    BE_analog = binding_energy(*analog, **mass_kwargs)[0]
    Q = BE_analog - BE_target - Ex_IAS
    CDE = 1.33 * (target[1] + analog[1]) * 0.5 / target[0] ** (1.0 / 3.0)
    kinematics_entrance = classical_kinematics(mass_target, mp, Elab, Zz=target[1])
    Ecm_exit = kinematics_entrance.Ecm + Q
    Elab_n, kinematics_exit = classical_kinematics_cm(mass_analog, mn, Ecm_exit)
    return kinematics_entrance, kinematics_exit, Elab_n, Q, CDE


class System:
    r"""
    Stores physics parameters of system. Calculates useful parameters for each partial wave.
    """

    def __init__(
        self,
        channel_radius_fm: np.float64,
        lmax: np.int64,
        target: tuple,
        analog: tuple,
        mass_target: np.float64,
        mass_analog: np.float64,
        kp: np.float64,
        kn: np.float64,
    ):
        r"""
        @params
            channel_radius_fm (np.float64): channel radius in fm
        """

        self.channel_radius_fm = channel_radius_fm
        self.lmax = lmax
        self.target = target
        self.analog = analog
        self.mass_target = mass_target
        self.mass_analog = mass_analog
        self.mass_p = mass(1, 1)
        self.mass_n = mass(1, 0)
        self.l = np.arange(0, lmax + 1, dtype=np.int64)

        self.entrance = ProjectileTargetSystem(
            channel_radius=self.channel_radius_fm * kp,
            lmax=self.lmax,
            mass_target=self.mass_target,
            mass_projectile=self.mass_p,
            Ztarget=self.target[1],
            Zproj=1,
            coupling=spin_half_orbit_coupling,
        )

        self.exit = ProjectileTargetSystem(
            channel_radius=self.channel_radius_fm * kn,
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

    @classmethod
    def load(obj, filename):
        with open(filename, "rb") as f:
            ws = pickle.load(f)
            ws.solver = Solver(ws.nbasis)
        return ws

    def save(self, filename):
        self.solver = None
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __init__(
        self,
        sys: System,
        kinematics_entrance: ChannelKinematics,
        kinematics_exit: ChannelKinematics,
        Elab_entrance: np.float64,
        Elab_exit: np.float64,
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
        self.angles = angles
        self.isovector_factor = np.sqrt(np.fabs(N - Z)) / (N - Z - 1)
        self.kinematics_entrance = kinematics_entrance
        self.kinematics_exit = kinematics_exit
        self.Elab_entrance = Elab_entrance
        self.Elab_exit = Elab_exit
        self.solver = solver
        self.nbasis = solver.kernel.quadrature.nbasis
        self.tmatrix_abs_tol = tmatrix_abs_tol

        # precompute things for entrance channel
        self.free_matrices_p = self.solver.free_matrix(
            sys.entrance.channel_radius, sys.l, coupled=False
        )
        self.basis_boundary_p = self.solver.precompute_boundaries(
            sys.entrance.channel_radius
        )

        # precompute things for exit channel
        self.free_matrices_n = self.solver.free_matrix(
            sys.exit.channel_radius, sys.l, coupled=False
        )
        self.basis_boundary_n = self.solver.precompute_boundaries(
            sys.exit.channel_radius
        )

        # get partial wave information for entrance channel
        channels, asymptotics = sys.entrance.get_partial_wave_channels(
            *self.kinematics_entrance
        )
        self.p_channels = [ch.decouple() for ch in channels]
        self.p_asymptotics = [asym.decouple() for asym in asymptotics]

        # get partial wave information for exit channel
        channels, asymptotics = sys.exit.get_partial_wave_channels(
            *self.kinematics_exit
        )
        self.n_channels = [ch.decouple() for ch in channels]
        self.n_asymptotics = [asym.decouple() for asym in asymptotics]

        # l . s for p-wave and up
        self.l_dot_s = np.array(
            [np.diag(coupling) for coupling in sys.entrance.couplings[1:]]
        )

        # pre-compute purely geometric factors
        self.xs_factor = (
            (self.kinematics_exit.k / self.kinematics_entrance.k)
            * self.kinematics_entrance.mu
            * self.kinematics_exit.mu
            / (4 * np.pi**2 * constants.HBARC**4 * (2 * 1.0 / 2 + 1))
        )
        self.geometric_factor = np.zeros(
            (2, 2, self.sys.lmax + 1, 2, self.angles.shape[0]), dtype=np.complex128
        )
        self.sigma_c = np.angle(
            gamma(1 + self.sys.l + 1j * self.kinematics_entrance.eta)
        )
        for im, m in enumerate([-0.5, 0.5]):
            for imp, mp in enumerate([-0.5, 0.5]):
                for l in range(0, self.sys.lmax + 1):
                    for ijp, jp in enumerate(
                        [l + 1 / 2, l - 1 / 2] if l > 0 else [l + 1 / 2]
                    ):
                        if abs(m - mp) <= l and jp >= 0:
                            ylm = sph_harm(m - mp, l, 0, self.angles)
                            cg0 = clebsch_gordan(l, 1 / 2, jp, m - mp, m, mp)
                            cg1 = clebsch_gordan(l, 1 / 2, jp, 0, m, m)

                            self.geometric_factor[im, imp, l, ijp, :] = (
                                (4 * np.pi) ** (3.0 / 2.0)
                                / (self.kinematics_entrance.k * self.kinematics_exit.k)
                                * np.exp(1j * self.sigma_c[l])
                                * cg1
                                * cg0
                                * np.sqrt(2 * l + 1)
                                * (-1) ** (2 * jp + 1)
                                * ylm
                            )

    def tmatrix(
        self,
        U_p_coulomb=None,
        U_p_scalar=None,
        U_p_spin_orbit=None,
        U_n_scalar=None,
        U_n_spin_orbit=None,
        args_p_coulomb=None,
        args_p_scalar=None,
        args_p_spin_orbit=None,
        args_n_scalar=None,
        args_n_spin_orbit=None,
    ):
        Tpn = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)
        Sn = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)
        Sp = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)

        # precomute scalar, spin-obit, and Coulomb interaction matrices
        # for entrance channel distorted waves
        im_scalar_p = self.solver.interaction_matrix(
            self.p_channels[0][0].k[0],
            self.p_channels[0][0].E[0],
            self.p_channels[0][0].a,
            self.p_channels[0][0].size,
            local_interaction=U_p_scalar,
            local_args=args_p_scalar,
        )
        im_spin_orbit_p = self.solver.interaction_matrix(
            self.p_channels[0][0].k[0],
            self.p_channels[0][0].E[0],
            self.p_channels[0][0].a,
            self.p_channels[0][0].size,
            local_interaction=U_p_spin_orbit,
            local_args=args_p_spin_orbit,
        )
        im_coulomb_p = self.solver.interaction_matrix(
            self.p_channels[0][0].k[0],
            self.p_channels[0][0].E[0],
            self.p_channels[0][0].a,
            self.p_channels[0][0].size,
            local_interaction=U_p_coulomb,
            local_args=args_p_coulomb,
        )

        # precomute scalar and spin-obit interaction matrices
        # for exit channel distorted waves
        im_scalar_n = self.solver.interaction_matrix(
            self.n_channels[0][0].k[0],
            self.n_channels[0][0].E[0],
            self.n_channels[0][0].a,
            self.n_channels[0][0].size,
            local_interaction=U_n_scalar,
            local_args=args_n_scalar,
        )
        im_spin_orbit_n = self.solver.interaction_matrix(
            self.n_channels[0][0].k[0],
            self.n_channels[0][0].E[0],
            self.n_channels[0][0].a,
            self.n_channels[0][0].size,
            local_interaction=U_n_spin_orbit,
            local_args=args_n_spin_orbit,
        )

        # evaluate the QE (p,n) transition matrix element on the quadrature
        # in r-space
        r_quadrature = (
            self.solver.kernel.quadrature.abscissa * self.sys.channel_radius_fm
        )
        U1_scalar = (
            -(
                U_n_scalar(r_quadrature, *args_n_scalar)
                - U_p_scalar(r_quadrature, *args_p_scalar)
            )
            * self.isovector_factor
        )
        U1_spin_orbit = (
            -(
                U_n_spin_orbit(r_quadrature, *args_n_spin_orbit)
                - U_p_spin_orbit(r_quadrature, *args_p_spin_orbit)
            )
            * self.isovector_factor
        )

        def tmatrix_element(l, ji, l_dot_s):
            nch = self.n_channels[l]
            pch = self.p_channels[l]
            Fn = self.free_matrices_n[l]
            Fp = self.free_matrices_p[l]
            nasym = self.n_asymptotics[l]
            pasym = self.p_asymptotics[l]

            _, snlj, xn, un = self.solver.solve(
                nch[ji],
                nasym[ji],
                free_matrix=Fn,
                interaction_matrix=im_scalar_n + l_dot_s * im_spin_orbit_n,
                basis_boundary=self.basis_boundary_n,
                wavefunction=True,
            )
            _, splj, xp, up = self.solver.solve(
                pch[ji],
                pasym[ji],
                free_matrix=Fp,
                interaction_matrix=(
                    im_scalar_p + im_coulomb_p + l_dot_s * im_spin_orbit_p
                ),
                basis_boundary=self.basis_boundary_p,
                wavefunction=True,
            )

            tlj = (
                np.sum(xp * (U1_scalar + l_dot_s * U1_spin_orbit) * xn)
                / self.sys.channel_radius_fm
                / self.kinematics_entrance.k
                / self.kinematics_exit.k
            )
            return tlj, snlj, splj

        # S-wave
        Tpn[0, 0], Sn[0, 0], Sp[0, 0] = tmatrix_element(0, 0, 0)

        # higher partial waves
        for l in self.sys.l[1:]:
            l_dot_s = self.l_dot_s[l - 1]
            Tpn[l, 0], Sn[l, 0], Sp[l, 0] = tmatrix_element(l, 0, l_dot_s[0])
            Tpn[l, 1], Sn[l, 1], Sp[l, 1] = tmatrix_element(l, 1, l_dot_s[1])

            if (
                np.absolute(Tpn[l, 0]) < self.tmatrix_abs_tol
                and np.absolute(Tpn[l, 1]) < self.tmatrix_abs_tol
            ):
                break

        return Tpn, Sn, Sp

    def xs(
        self,
        U_p_coulomb=None,
        U_p_scalar=None,
        U_p_spin_orbit=None,
        U_n_scalar=None,
        U_n_spin_orbit=None,
        args_p_coulomb=None,
        args_p_scalar=None,
        args_p_spin_orbit=None,
        args_n_scalar=None,
        args_n_spin_orbit=None,
    ):
        Tmmp = np.zeros((2, 2, self.angles.shape[0]), dtype=np.complex128)
        Tlj, Sn, Sp = self.tmatrix(
            U_p_coulomb=U_p_coulomb,
            U_p_scalar=U_p_scalar,
            U_p_spin_orbit=U_p_spin_orbit,
            U_n_scalar=U_n_scalar,
            U_n_spin_orbit=U_n_spin_orbit,
            args_p_coulomb=args_p_coulomb,
            args_p_scalar=args_p_scalar,
            args_p_spin_orbit=args_p_spin_orbit,
            args_n_scalar=args_n_scalar,
            args_n_spin_orbit=args_n_spin_orbit,
        )
        # TODO cast into a np.sum
        for im, m in enumerate([-0.5, 0.5]):
            for imp, mp in enumerate([-0.5, 0.5]):
                for l in range(0, self.sys.lmax):
                    for ijp, jp in enumerate([l + 0.5, l - 0.5]):
                        if abs(m - mp) <= l and jp >= 0:
                            Tmmp[im, imp, :] += (
                                self.geometric_factor[im, imp, l, ijp, :] * Tlj[l, ijp]
                            )
        return self.xs_factor * 10 * np.sum(np.absolute(Tmmp) ** 2, axis=(0, 1))
