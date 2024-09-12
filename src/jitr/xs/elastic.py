from numba import njit
from dataclasses import dataclass
from scipy.special import eval_legendre, gamma
import numpy as np

from ..utils import eval_assoc_legendre, constants, kinematics
from ..reactions import ProjectileTargetSystem
from ..rmatrix import Solver


@dataclass
class ElasticXS:
    r"""
    Holds differential cross section, analyzing power, total cross section and
    reaction cross secton, all at a given energy
    """

    dsdo: np.array
    Ay: np.array
    t: np.float64
    rxn: np.float64


class ElasticXSWorkspace:
    r"""
    Workspace for elastic scattering observables for a parametric,
    local and l-independent interaction
    """

    def __init__(
        self,
        projectile: tuple,
        target: tuple,
        sys: ProjectileTargetSystem,
        Ecm: np.float64,
        k: np.float64,
        mu: np.float64,
        eta: np.float64,
        local_interaction_scalar,
        local_interaction_spin_orbit,
        solver: Solver,
        angles: np.array,
        smatrix_abs_tol: np.float64 = 1e-6,
    ):
        assert np.all(np.diff(angles) > 0)
        assert angles[0] >= 0.0 and angles[-1] <= np.pi

        self.projectile = projectile
        self.target = target
        self.sys = sys
        self.solver = solver
        self.mu = mu
        self.Ecm = Ecm
        self.k = k
        self.eta = eta
        self.local_interaction_scalar = local_interaction_scalar
        self.local_interaction_spin_orbit = local_interaction_spin_orbit
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
        self.l_dot_s = np.array( [np.diag(coupling) for coupling in  sys.couplings[1:]])

        # preocmpute angular distributions in each partial wave
        self.angles = angles
        ls = self.sys.l[:, np.newaxis]
        self.P_l_costheta = eval_legendre(ls, np.cos(self.angles))
        self.P_1_l_costheta = np.array(
            [eval_assoc_legendre(l, np.cos(self.angles)) for l in self.sys.l]
        )

        # precompute things related to Coulomb interaction
        self.Zz = self.projectile[1] * self.target[1]
        if self.Zz > 0:
            self.k_c = constants.ALPHA * self.Zz * self.mu / constants.HBARC
            self.eta = self.k_c / self.k
            self.sigma_l = np.angle(gamma(1 + ls + 1j * self.eta))
            sin2 = np.sin(self.angles / 2.) ** 2
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

    def xs(self, args_scalar=None, args_spin_orbit=None):
        splus = np.zeros(self.sys.lmax+1, dtype=np.complex128)
        sminus = np.zeros(self.sys.lmax, dtype=np.complex128)

        # precompute the interaction matrix
        im_scalar = self.solver.interaction_matrix(
            self.channels[0][0],
            local_interaction=self.local_interaction_scalar,
            local_args=args_scalar,
        )
        im_spin_orbit = self.solver.interaction_matrix(
            self.channels[0][0],
            local_interaction=self.local_interaction_spin_orbit,
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
            # j = l + 1/2
            _, splus[l], _ = self.solver.solve(
                ch[0],
                asym[0],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_scalar + self.l_dot_s[l-1, 0] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )

            # j = l - 1/2
            _, sminus[l-1], _ = self.solver.solve(
                ch[1],
                asym[1],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_scalar + self.l_dot_s[l-1, 1] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )

            if (1.0 - np.absolute(splus[l])) < self.smatrix_abs_tol and (
                1.0 - np.absolute(sminus[l-1])
            ) < self.smatrix_abs_tol:
                break

        #return splus, sminus
        return ElasticXS(
            *elastic_xs(
                self.k,
                self.angles,
                splus,
                sminus,
                self.P_l_costheta,
                self.P_1_l_costheta,
                self.f_c,
                self.sigma_l,
            )
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

    # l = 0 only has j=1/2 contribution
    a += (1 - Splus[0]) * P_l_theta[0, :] * np.exp(2j * sigma_l[0]) / (2j * k)
    xsrxn = 1 - np.absolute(Splus[0])
    xst = 1 - np.real(Splus[0])  # only valid for neutral projectiles

    for l in range(1, Splus.shape[0]):
        a += (
            (2 * l + 1 - (l + 1) * Splus[l] - l * Sminus[l])
            * P_l_theta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2j * k)
        )
        b += (
            (Sminus[l] - Splus[l])
            * P_1_l_theta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2j * k)
        )
        xsrxn += (l + 1) * (1 - np.absolute(Splus[l])) + l * (
            1 - np.absolute(Sminus[l])
        )
        xst += (l + 1) * (1 - np.real(Splus[l])) + l * (1 - np.real(Sminus[l]))

    dsdo = np.real(a * np.conj(a) + b * np.conj(b)) * 10
    Ay = np.real(a * np.conj(b) + b * np.conj(a)) * 10 / dsdo
    xsrxn *= 10 * np.pi / k**2
    xst *= 10 * 2 * np.pi / k**2

    return dsdo, Ay, xst, xsrxn
