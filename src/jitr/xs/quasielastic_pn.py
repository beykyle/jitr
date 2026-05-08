"""DWBA workspaces for quasi-elastic ``(p,n)`` scattering observables."""

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, sph_harm_y
from sympy.physics.wigner import clebsch_gordan

from ..reactions import ProjectileTargetSystem, Reaction, spin_half_orbit_coupling
from ..rmatrix import Solver
from ..utils import constants
from ..utils.kinematics import ChannelKinematics
from .elastic import check_angles

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


class System:
    r"""
    System for (p,n) quasi-elastic scattering observables for local interactions
    This system contains the entrance and exit channels, which are defined by the
    projectile and target masses, charges, and the channel radius.

    :ivar channel_radius_fm: float The channel radius in femtometers.
    :ivar lmax: int The maximum angular momentum quantum number.
    :ivar l: np.ndarray An array of angular momentum quantum numbers from 0 to lmax.
    :ivar entrance: ProjectileTargetSystem The entrance channel system, which
                    includes the projectile and target masses, charges, and the
                    channel radius.
    :ivar exit: ProjectileTargetSystem The exit channel system, which includes the
                product and residual masses, charges, and the channel radius."""

    def __init__(
        self,
        channel_radius_fm: float,
        lmax: int,
        reaction: Reaction,
        kinematics_entrance: ChannelKinematics,
        kinematics_exit: ChannelKinematics,
    ) -> None:
        r"""
        Initialize the System for (p,n) quasi-elastic scattering observables.
        :param channel_radius_fm: float The channel radius in femtometers.
        :param lmax: int The maximum angular momentum quantum number.
        :param reaction: Reaction The reaction object containing information
                         about the target, projectile, residual, and product.
        :param kinematics_entrance: ChannelKinematics The kinematics for the
                                    entrance channel.
        :param kinematics_exit: ChannelKinematics The kinematics for the exit
                                channel."""

        self.channel_radius_fm = channel_radius_fm
        self.lmax = lmax
        self.l = np.arange(0, lmax + 1, dtype=np.int64)

        self.entrance = ProjectileTargetSystem(
            channel_radius=self.channel_radius_fm * kinematics_entrance.k,
            lmax=self.lmax,
            mass_target=reaction.target.m0,
            mass_projectile=reaction.projectile.m0,
            Ztarget=reaction.target.Z,
            Zproj=reaction.projectile.Z,
            coupling=spin_half_orbit_coupling,
        )

        if reaction.residual is None or reaction.product is None:
            raise ValueError(
                "Reaction must define both residual and product for (p,n) scattering"
            )
        self.exit = ProjectileTargetSystem(
            channel_radius=self.channel_radius_fm * kinematics_exit.k,
            lmax=self.lmax,
            mass_target=reaction.residual.m0,
            mass_projectile=reaction.product.m0,
            Ztarget=reaction.residual.Z,
            Zproj=reaction.product.Z,
            coupling=spin_half_orbit_coupling,
        )


class Workspace:
    r"""
    Workspace for (p,n) quasi-elastic scattering observables in the DWBA.
    This class computes the transition matrix and differential cross section
    for the (p,n) reaction using the distorted wave Born approximation (DWBA).
    """

    def __init__(
        self,
        reaction: Reaction,
        kinematics_entrance: ChannelKinematics,
        kinematics_exit: ChannelKinematics,
        solver: Solver,
        angles: FloatArray,
        lmax: int,
        channel_radius_fm: float,
        tmatrix_abs_tol: float = 1e-6,
    ) -> None:
        r"""
        Initialize the Workspace for (p,n) quasi-elastic scattering observables.
        :param reaction: Reaction The reaction object containing information
                         about the target, projectile, residual, and product.
        :param kinematics_entrance: ChannelKinematics The kinematics for the
                                    entrance channel.
        :param kinematics_exit: ChannelKinematics The kinematics for the exit
                                channel.
        :param solver: Solver The solver used to compute the distorted waves and
                       interaction matrices.
        :param angles: np.array The angles in radians at which to compute the
                       differential cross section.
        :param lmax: int The maximum angular momentum quantum number.
        :param channel_radius_fm: float The channel radius in femtometers.
        :param tmatrix_abs_tol: float The absolute tolerance for the transition
                                matrix elements."""

        # params
        self.lmax = lmax
        self.channel_radius_fm = channel_radius_fm
        self.tmatrix_abs_tol = tmatrix_abs_tol

        # system
        self.reaction = reaction
        self.sys = System(
            channel_radius_fm,
            lmax,
            reaction,
            kinematics_entrance,
            kinematics_exit,
        )

        # kinematics
        self.kinematics_entrance = kinematics_entrance
        self.kinematics_exit = kinematics_exit
        self.solver = solver

        # angles
        check_angles(angles)
        self.angles = angles

        # precompute for DWBA matrix element
        A = self.reaction.target.A
        Z = self.reaction.target.Z
        N = A - Z
        self.isovector_factor = np.sqrt(np.fabs(N - Z)) / (N - Z - 1)

        # precompute things for entrance channel
        self.free_matrices_p = self.solver.free_matrix(
            self.sys.entrance.channel_radius, self.sys.l, coupled=False
        )
        self.basis_boundary_p = self.solver.precompute_boundaries(
            self.sys.entrance.channel_radius
        )

        # precompute things for exit channel
        self.free_matrices_n = self.solver.free_matrix(
            self.sys.exit.channel_radius, self.sys.l, coupled=False
        )
        self.basis_boundary_n = self.solver.precompute_boundaries(
            self.sys.exit.channel_radius
        )

        # get partial wave information for entrance channel
        channels, asymptotics = self.sys.entrance.get_partial_wave_channels(
            *self.kinematics_entrance
        )
        self.p_channels = [ch.decouple() for ch in channels]
        self.p_asymptotics = [asym.decouple() for asym in asymptotics]

        # get partial wave information for exit channel
        channels, asymptotics = self.sys.exit.get_partial_wave_channels(
            *self.kinematics_exit
        )
        self.n_channels = [ch.decouple() for ch in channels]
        self.n_asymptotics = [asym.decouple() for asym in asymptotics]

        # l . s for p-wave and up
        self.l_dot_s = np.array(
            [np.diag(coupling) for coupling in self.sys.entrance.couplings[1:]]
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
                            ylm = sph_harm_y(l, int(m - mp), self.angles, 0)
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

    def radial_grid(self) -> FloatArray:
        """Return the physical quadrature grid used for local potentials."""
        return self.solver.radial_grid(
            self.p_channels[0][0].a, self.kinematics_entrance.k
        )

    def _local_potential(self, potential: npt.ArrayLike, name: str) -> ComplexArray:
        """Validate and cast a local potential array on the quadrature grid."""
        potential_array = np.asarray(potential, dtype=np.complex128)
        expected_shape = (self.solver.kernel.quadrature.nbasis,)
        if potential_array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        return potential_array

    def tmatrix(
        self,
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike,
        U_n_central: npt.ArrayLike,
        U_n_spin_orbit: npt.ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the transition matrix for (p,n) quasi-elastic scattering
        using the distorted wave Born approximation (DWBA).
        :param U_p_coulomb: callable The Coulomb interaction for the proton.
        :param U_p_central: callable The central interaction for the proton.
        :param U_p_spin_orbit: callable The spin-orbit interaction for the
                               proton.
        :param U_n_central: callable The central interaction for the neutron.
        :param U_n_spin_orbit: callable The spin-orbit interaction for the
                               neutron.
        :param args_p_coulomb: tuple parameters for the proton Coulomb
                               interaction.
        :param args_p_central: tuple parameters for the proton central
                               interaction.
        :param args_p_spin_orbit: tuple parameters for the proton spin-orbit
                                  interaction.
        :param args_n_central: tuple parameters for the neutron central
                               interaction.
        :param args_n_spin_orbit: tuple parameters for the neutron spin-orbit
                                  interaction.
        :returns: Tpn: np.ndarray The transition matrix for the (p,n) reaction.;
                  Sn: np.ndarray The S-matrix for the neutron elastic exit
                  channel.; Sp: np.ndarray The S-matrix for the proton elastic
                  entrance channel.
        :rtype: tuple[Tpn, Sn, Sp]"""
        Tpn = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)
        Sn = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)
        Sp = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)

        # precomute central, spin-obit, and Coulomb interaction matrices
        # for entrance channel distorted waves
        proton_central = self._local_potential(U_p_central, "U_p_central")
        proton_spin_orbit = self._local_potential(U_p_spin_orbit, "U_p_spin_orbit")
        proton_coulomb = self._local_potential(U_p_coulomb, "U_p_coulomb")
        neutron_central = self._local_potential(U_n_central, "U_n_central")
        neutron_spin_orbit = self._local_potential(U_n_spin_orbit, "U_n_spin_orbit")

        im_central_p = self.solver.interaction_matrix(
            self.p_channels[0][0].k[0],
            self.p_channels[0][0].E[0],
            self.p_channels[0][0].a,
            self.p_channels[0][0].size,
            local_potential=proton_central,
        )
        im_spin_orbit_p = self.solver.interaction_matrix(
            self.p_channels[0][0].k[0],
            self.p_channels[0][0].E[0],
            self.p_channels[0][0].a,
            self.p_channels[0][0].size,
            local_potential=proton_spin_orbit,
        )
        im_coulomb_p = self.solver.interaction_matrix(
            self.p_channels[0][0].k[0],
            self.p_channels[0][0].E[0],
            self.p_channels[0][0].a,
            self.p_channels[0][0].size,
            local_potential=proton_coulomb,
        )

        # precomute central and spin-obit interaction matrices
        # for exit channel distorted waves
        im_central_n = self.solver.interaction_matrix(
            self.n_channels[0][0].k[0],
            self.n_channels[0][0].E[0],
            self.n_channels[0][0].a,
            self.n_channels[0][0].size,
            local_potential=neutron_central,
        )
        im_spin_orbit_n = self.solver.interaction_matrix(
            self.n_channels[0][0].k[0],
            self.n_channels[0][0].E[0],
            self.n_channels[0][0].a,
            self.n_channels[0][0].size,
            local_potential=neutron_spin_orbit,
        )

        U1_central = -(neutron_central - proton_central) * self.isovector_factor
        U1_spin_orbit = (
            -(neutron_spin_orbit - proton_spin_orbit) * self.isovector_factor
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
                interaction_matrix=im_central_n + l_dot_s * im_spin_orbit_n,
                basis_boundary=self.basis_boundary_n,
                wavefunction=True,
            )
            _, splj, xp, up = self.solver.solve(
                pch[ji],
                pasym[ji],
                free_matrix=Fp,
                interaction_matrix=(
                    im_central_p + im_coulomb_p + l_dot_s * im_spin_orbit_p
                ),
                basis_boundary=self.basis_boundary_p,
                wavefunction=True,
            )

            tlj = (
                np.sum(xp * (U1_central + l_dot_s * U1_spin_orbit) * xn)
                / self.sys.channel_radius_fm
                / self.kinematics_entrance.k
                / self.kinematics_exit.k
            )
            return tlj, snlj[0, 0], splj[0, 0]

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
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike,
        U_n_central: npt.ArrayLike,
        U_n_spin_orbit: npt.ArrayLike,
    ) -> np.ndarray:
        """
        Calculate the differential cross section for (p,n) quasi-elastic
        scattering
        in mb/Sr in the outgoing neutron angle using DWBA.
        :param U_p_coulomb: callable The Coulomb interaction for the proton.
        :param U_p_central: callable The central interaction for the proton.
        :param U_p_spin_orbit: callable The spin-orbit interaction for the
                               proton.
        :param U_n_central: callable The central interaction for the neutron.
        :param U_n_spin_orbit: callable The spin-orbit interaction for the
                               neutron.
        :param args_p_coulomb: tuple parameters for the proton Coulomb
                               interaction.
        :param args_p_central: tuple parameters for the proton central
                               interaction.
        :param args_p_spin_orbit: tuple parameters for the proton spin-orbit
                                  interaction.
        :param args_n_central: tuple parameters for the neutron central
                               interaction.
        :param args_n_spin_orbit: tuple parameters for the neutron spin-orbit
                                  interaction.
        :returns: np.ndarray The differential cross section for the (p,n)
                  reaction in mb/Sr.
        :rtype: xs"""

        Tmmp = np.zeros((2, 2, self.angles.shape[0]), dtype=np.complex128)
        Tlj, Sn, Sp = self.tmatrix(
            U_p_coulomb=U_p_coulomb,
            U_p_central=U_p_central,
            U_p_spin_orbit=U_p_spin_orbit,
            U_n_central=U_n_central,
            U_n_spin_orbit=U_n_spin_orbit,
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
