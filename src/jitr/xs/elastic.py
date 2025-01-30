from numba import njit
from dataclasses import dataclass
from scipy.special import eval_legendre, lpmv, gamma
import numpy as np
import pickle

from ..utils import constants
from ..utils.kinematics import ChannelKinematics
from ..reactions import ProjectileTargetSystem
from ..rmatrix import Solver


@dataclass
class ElasticXS:
    r"""
    Holds differential cross section, analyzing power, total cross section and
    reaction cross secton, all at a given energy
    """

    dsdo: np.ndarray
    Ay: np.ndarray
    t: np.float64
    rxn: np.float64
    rutherford: np.ndarray = None


class IntegralWorkspace:
    r"""
    Workspace for integral observables like S-matrix elements and total and reaction cross sections for
    local interactions with spin-orbit coupling
    """

    def __init__(
        self,
        projectile: tuple,
        target: tuple,
        sys: ProjectileTargetSystem,
        kinematics: ChannelKinematics,
        solver: Solver,
        smatrix_abs_tol: np.float64 = 1e-6,
    ):
        # system info
        self.projectile = projectile
        self.target = target
        self.sys = sys
        self.solver = solver
        self.nbasis = solver.kernel.quadrature.nbasis
        self.smatrix_abs_tol = smatrix_abs_tol

        # kinematic info
        self.mu = kinematics.mu
        self.Ecm = kinematics.Ecm
        self.k = kinematics.k
        self.eta = kinematics.eta

        # precompute things
        self.free_matrices = self.solver.free_matrix(
            sys.channel_radius, sys.l, coupled=False
        )
        self.basis_boundary = self.solver.precompute_boundaries(sys.channel_radius)

        # get information for each channel
        channels, asymptotics = sys.get_partial_wave_channels(
            self.Ecm, self.mu, self.k, self.eta
        )

        # de couple into two independent systems per partial wave
        self.channels = [ch.decouple() for ch in channels]
        self.asymptotics = [asym.decouple() for asym in asymptotics]
        self.l_dot_s = np.array([np.diag(coupling) for coupling in sys.couplings[1:]])

        # precompute things related to Coulomb interaction
        self.ls = self.sys.l[:, np.newaxis]
        self.Zz = self.projectile[1] * self.target[1]
        if self.Zz > 0:
            self.k_c = constants.ALPHA * self.Zz * self.mu / constants.HBARC
            self.eta = self.k_c / self.k
            self.sigma_l = np.angle(gamma(1 + self.ls + 1j * self.eta))
        else:
            self.sigma_l = np.angle(gamma(1 + self.ls + 1j * 0))
            self.k_c = 0
            self.eta = 0

    def smatrix(
        self,
        interaction_central,
        interaction_spin_orbit,
        args_central=None,
        args_spin_orbit=None,
    ):
        r"""
        returns the partial wave S-matrix elements as two arrays over partial
        wave l, one for for the l+1/2 and a ssecond for the l-1/2 partial waves
        """
        splus = np.zeros(self.sys.lmax + 1, dtype=np.complex128)
        sminus = np.zeros(self.sys.lmax + 1, dtype=np.complex128)

        # precompute the interaction matrix
        im_central = self.solver.interaction_matrix(
            self.channels[0][0].k[0],
            self.channels[0][0].E[0],
            self.channels[0][0].a,
            self.channels[0][0].size,
            local_interaction=interaction_central,
            local_args=args_central,
        )
        im_spin_orbit = self.solver.interaction_matrix(
            self.channels[0][0].k[0],
            self.channels[0][0].E[0],
            self.channels[0][0].a,
            self.channels[0][0].size,
            local_interaction=interaction_spin_orbit,
            local_args=args_spin_orbit,
        )

        # s-wave, l = 0, j = 1/2
        _, splus[0], _ = self.solver.solve(
            self.channels[0][0],
            self.asymptotics[0][0],
            free_matrix=self.free_matrices[0],
            interaction_matrix=im_central,
            basis_boundary=self.basis_boundary,
        )

        # higher partial waves
        for l in self.sys.l[1:]:
            ch = self.channels[l]
            asym = self.asymptotics[l]
            lds = self.l_dot_s[l - 1]  # starts from 1 not 0
            # j = l + 1/2
            _, splus[l], _ = self.solver.solve(
                ch[0],
                asym[0],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_central + lds[0] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )

            # j = l - 1/2
            _, sminus[l], _ = self.solver.solve(
                ch[1],
                asym[1],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_central + lds[1] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )

            if (np.absolute(1 - splus[l])) < self.smatrix_abs_tol and (
                np.absolute(1 - sminus[l])
            ) < self.smatrix_abs_tol:
                break

        return splus[:l], sminus[:l]

    def xs(
        self,
        interaction_central,
        interaction_spin_orbit,
        args_central=None,
        args_spin_orbit=None,
        angles=None,
    ):
        r"""
        returns the angle-integrated total, elastic and reaction cross sections
        """
        splus, sminus = self.smatrix(
            interaction_central,
            interaction_spin_orbit,
            args_central,
            args_spin_orbit,
        )
        return integral_elastic_xs(self.k, splus, sminus, self.ls, self.sigma_l)

    def transmission_coefficients(
        self,
        interaction_central,
        interaction_spin_orbit,
        args_central=None,
        args_spin_orbit=None,
        angles=None,
    ):
        r"""
        returns the partial wave tranmission coefficients as two arrays over
        partial wave l, one for for the l+1/2 and a second for the l-1/2
        partial waves
        """
        splus, sminus = self.smatrix(
            interaction_central,
            interaction_spin_orbit,
            args_central,
            args_spin_orbit,
        )
        return 1.0 - np.absolute(splus) ** 2, 1.0 - np.absolute(sminus) ** 2


class DifferentialWorkspace:
    r"""
    Workspace for differential elastic scattering observables for local
    interactions with spin-orbit coupling
    """

    @classmethod
    def build_from_system(
        cls,
        projectile: tuple,
        target: tuple,
        sys: ProjectileTargetSystem,
        kinematics: ChannelKinematics,
        solver: Solver,
        angles: np.array,
        smatrix_abs_tol: np.float64 = 1e-6,
    ):
        integral_workspace = IntegralWorkspace(
            projectile,
            target,
            sys,
            kinematics,
            solver,
            smatrix_abs_tol,
        )
        return cls(integral_workspace, angles, smatrix_abs_tol)

    def __init__(
        self,
        integral_workspace: IntegralWorkspace,
        angles: np.array,
        smatrix_abs_tol: np.float64 = 1e-6,
    ):

        # system info
        self.integral_workspace = integral_workspace
        self.projectile = integral_workspace.projectile
        self.target = integral_workspace.target
        self.sys = integral_workspace.sys
        self.solver = integral_workspace.solver

        # kinematic info
        self.mu = integral_workspace.mu
        self.Ecm = integral_workspace.Ecm
        self.k = integral_workspace.k
        self.eta = integral_workspace.eta

        # precompute angular distributions in each partial wave
        self.angles = angles
        self.ls = self.integral_workspace.ls
        self.P_l_costheta = eval_legendre(self.ls, np.cos(self.angles))
        self.P_1_l_costheta = lpmv(1, self.ls, np.cos(self.angles))

        # precompute things related to Coulomb interaction
        self.sigma_l = self.integral_workspace.sigma_l
        self.Zz = self.integral_workspace.Zz
        self.k_c = self.integral_workspace.k_c
        self.eta = self.integral_workspace.eta
        if self.Zz > 0:
            sin2 = np.sin(self.angles / 2.0) ** 2
            self.f_c = (
                -self.eta
                / (2 * self.k * sin2)
                * np.exp(
                    2j * self.sigma_l[0]
                    - 2j * self.eta * np.log(np.sin(self.angles / 2))
                )
            )
            self.rutherford = 10 * self.eta**2 / (4 * self.k**2 * sin2**2)
        else:
            self.f_c = np.zeros_like(angles)
            self.rutherford = None

    def xs(
        self,
        interaction_central,
        interaction_spin_orbit,
        args_central=None,
        args_spin_orbit=None,
        angles=None,
    ):
        if angles is None:
            angles = self.angles
            P_l_costheta = self.P_l_costheta
            P_1_l_costheta = self.P_1_l_costheta
            rutherford = self.rutherford
            f_c = self.f_c
        else:
            P_l_costheta = eval_legendre(self.ls, np.cos(angles))
            P_1_l_costheta = lpmv(1, self.ls, np.cos(angles))
            if self.Zz > 0:
                sin2 = np.sin(angles / 2) ** 2
                rutherford = 10 * self.eta**2 / (4 * self.k**2 * sin2**2)
                f_c = (
                    -self.eta
                    / (2 * self.k * sin2)
                    * np.exp(-1j * self.eta * np.log(sin2) + 2j * self.sigma_l[0])
                )
            else:
                rutherford = None
                f_c = np.zeros_like(angles)

        splus, sminus = self.integral_workspace.smatrix(
            interaction_central, interaction_spin_orbit, args_central, args_spin_orbit
        )
        return ElasticXS(
            *differential_elastic_xs(
                self.k,
                angles,
                splus,
                sminus,
                self.ls,
                P_l_costheta,
                P_1_l_costheta,
                f_c,
                self.sigma_l,
            ),
            rutherford,
        )


@njit
def integral_elastic_xs(
    k: float,
    Splus: np.array,
    Sminus: np.array,
    ls: np.array,
    sigma_l: np.array = 0,
):
    xsrxn = 0.0
    xst = 0.0

    for l in range(Splus.shape[0]):
        xsrxn += (l + 1) * (1 - np.absolute(Splus[l]) ** 2) + l * (
            1 - np.absolute(Sminus[l]) ** 2
        )
        xst += (l + 1) * (1 - np.real(Splus[l])) + l * (1 - np.real(Sminus[l]))

    xsrxn *= 10 * np.pi / k**2
    xst *= 10 * 2 * np.pi / k**2

    return xst, xsrxn


@njit
def differential_elastic_xs(
    k: float,
    angles: np.array,
    splus: np.array,
    sminus: np.array,
    ls: np.array,
    P_l_costheta: np.array,
    P_1_l_costheta: np.array,
    f_c: np.array = 0,
    sigma_l: np.array = 0,
):
    r"""
    Calculates differential, total and reaction cross section for spin-1/2 spin-0 scattering
    following Herman, et al., 2007, https://doi.org/10.1016/j.nds.2007.11.003
    """
    a = np.zeros_like(angles, dtype=np.complex128) + f_c
    b = np.zeros_like(angles, dtype=np.complex128)
    xsrxn = 0.0
    xst = 0.0

    for l in range(splus.shape[0]):
        a += (
            P_l_costheta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2j * k)
            * ((l + 1) * (splus[l] - 1) + l * (sminus[l] - 1))
        )
        b += (
            P_1_l_costheta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2j * k)
            * (splus[l] - sminus[l])
        )

        xsrxn += (l + 1) * (1 - np.absolute(splus[l]) ** 2) + l * (
            1 - np.absolute(sminus[l]) ** 2
        )
        xst += (l + 1) * (1 - np.real(splus[l])) + l * (1 - np.real(sminus[l]))

    dsdo = (np.absolute(a) ** 2 + np.absolute(b) ** 2) * 10
    Ay = np.imag(a.conj() * b) * 10 / dsdo
    xsrxn *= 10 * np.pi / k**2
    xst *= 10 * 2 * np.pi / k**2

    return dsdo, Ay, xst, xsrxn
