from numba import njit
import pickle
from dataclasses import dataclass
from scipy.special import eval_legendre, lpmv, gamma
import numpy as np

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


class Workspace:
    r"""
    Workspace for elastic scattering observables for local interactions with spin-orbit coupling
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
        projectile: tuple,
        target: tuple,
        sys: ProjectileTargetSystem,
        kinematics: ChannelKinematics,
        solver: Solver,
        angles: np.array,
        smatrix_abs_tol: np.float64 = 1e-6,
    ):

        self.projectile = projectile
        self.target = target
        self.sys = sys
        self.solver = solver
        self.nbasis = solver.kernel.quadrature.nbasis
        self.mu = kinematics.mu
        self.Ecm = kinematics.Ecm
        self.k = kinematics.k
        self.eta = kinematics.eta
        self.smatrix_abs_tol = smatrix_abs_tol

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

        # preocmpute angular distributions in each partial wave
        self.angles = angles
        ls = self.sys.l[:, np.newaxis]
        self.P_l_costheta = eval_legendre(ls, np.cos(self.angles))
        self.P_1_l_costheta = lpmv(1, ls, np.cos(self.angles))

        # precompute things related to Coulomb interaction
        self.Zz = self.projectile[1] * self.target[1]
        if self.Zz > 0:
            self.k_c = constants.ALPHA * self.Zz * self.mu / constants.HBARC
            self.eta = self.k_c / self.k
            self.sigma_l = np.angle(gamma(1 + ls + 1j * self.eta))
            sin2 = np.sin(self.angles / 2.0) ** 2
            self.f_c = (
                -self.eta
                / (2 * self.k * sin2)
                * np.exp(
                    2j * self.sigma_l[0]
                    - 2j * self.eta * np.log(np.sin(self.angles / 2))
                )
            )
            self.rutherford = (
                10 * self.eta**2 / (4 * self.k**2 * np.sin(self.angles / 2) ** 4)
            )
        else:
            self.k_c = 0
            self.eta = 0
            self.sigma_l = np.angle(gamma(1 + ls + 1j * 0))
            self.f_c = np.zeros_like(angles)
            self.rutherford = np.zeros_like(angles)

    def smatrix(
        self,
        interaction_scalar,
        interaction_spin_orbit,
        args_scalar=None,
        args_spin_orbit=None,
    ):
        splus = np.zeros(self.sys.lmax + 1, dtype=np.complex128)
        sminus = np.zeros(self.sys.lmax + 1, dtype=np.complex128)

        # precompute the interaction matrix
        im_scalar = self.solver.interaction_matrix(
            self.channels[0][0],
            local_interaction=interaction_scalar,
            local_args=args_scalar,
        )
        im_spin_orbit = self.solver.interaction_matrix(
            self.channels[0][0],
            local_interaction=interaction_spin_orbit,
            local_args=args_spin_orbit,
        )

        # s-wave
        _, splus[0], _ = self.solver.solve(
            self.channels[0][0],
            self.asymptotics[0][0],
            free_matrix=self.free_matrices[0],
            interaction_matrix=im_scalar,
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
                interaction_matrix=im_scalar + lds[0] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )

            # j = l - 1/2
            _, sminus[l], _ = self.solver.solve(
                ch[1],
                asym[1],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_scalar + lds[1] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )

            if (np.absolute(1 - splus[l])) < self.smatrix_abs_tol and (
                np.absolute(1 - sminus[l])
            ) < self.smatrix_abs_tol:
                break

        return splus[:l], sminus[:l]

    def xs(
        self,
        interaction_scalar,
        interaction_spin_orbit,
        args_scalar=None,
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
            P_l_costheta = eval_legendre(self.sys.l[:, np.newaxis], np.cos(angles))
            P_1_l_costheta = lpmv(1, self.sys.l[:, np.newaxis], np.cos(angles))
            sin2 = np.sin(angles / 2) ** 2
            rutherford = 10 * self.eta**2 / (4 * self.k**2 * sin2**2)
            f_c = (
                -self.eta
                / (2 * self.k * sin2)
                * np.exp(-1j * self.eta * np.log(sin2) + 2j * self.sigma_l[0])
            )

        splus, sminus = self.smatrix(
            interaction_scalar, interaction_spin_orbit, args_scalar, args_spin_orbit
        )
        return ElasticXS(
            *elastic_xs(
                self.k,
                angles,
                splus,
                sminus,
                P_l_costheta,
                P_1_l_costheta,
                f_c,
                self.sigma_l,
            ),
            rutherford,
        )


@njit
def elastic_xs(
    k: float,
    angles: np.array,
    Splus: np.array,
    Sminus: np.array,
    P_l_theta: np.array,
    P_1_l_theta: np.array,
    f_c: np.array = 0,
    sigma_l: np.array = 0,
):
    a = np.zeros_like(angles, dtype=np.complex128) + f_c
    b = np.zeros_like(angles, dtype=np.complex128)
    xsrxn = 0.0
    xst = 0.0

    for l in range(Splus.shape[0]):
        # scattering amplitudes
        a += 1j * (
            (2 * l + 1 - (l + 1) * Splus[l] - l * Sminus[l])
            * P_l_theta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2 * k)
        )
        b += 1j * (
            (Sminus[l] - Splus[l])
            * P_1_l_theta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2 * k)
        )
        xsrxn += (l + 1) * (1 - np.absolute(Splus[l]) ** 2) + l * (
            1 - np.absolute(Sminus[l]) ** 2
        )
        xst += (l + 1) * (1 - np.real(Splus[l])) + l * (1 - np.real(Sminus[l]))

    dsdo = (np.absolute(a) ** 2 + np.absolute(b) ** 2) * 10
    Ay = np.real(a * np.conj(b) + b * np.conj(a)) * 10 / dsdo
    xsrxn *= 10 * np.pi / k**2
    xst *= 10 * 2 * np.pi / k**2

    return dsdo, Ay, xst, xsrxn
