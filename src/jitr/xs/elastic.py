"""Elastic-scattering observables built from the R-matrix solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.special import eval_legendre, gamma, lpmv

from ..reactions import ProjectileTargetSystem, Reaction, spin_half_orbit_coupling
from ..rmatrix import Solver
from ..utils.kinematics import ChannelKinematics

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass
class ElasticXS:
    """Container for elastic-scattering observables.

    Attributes:
        dsdo: Differential cross section in mb/sr.
        Ay: Analyzing power.
        Q: Spin-rotation function.
        t: Total cross section in mb.
        rxn: Reaction cross section in mb.
    """

    dsdo: np.ndarray
    Ay: np.ndarray
    Q: np.ndarray
    t: np.float64
    rxn: np.float64


class IntegralWorkspace:
    """Workspace for integral elastic observables with spin-orbit coupling."""

    def __init__(
        self,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        channel_radius_fm: float,
        solver: Solver,
        lmax: int,
        smatrix_abs_tol: float = 1e-6,
    ) -> None:
        """Build the workspace from reaction and kinematic information."""
        if reaction.process is None or reaction.process.lower() != "el":
            raise ValueError("Reaction must be elastic!")

        self.smatrix_abs_tol = smatrix_abs_tol
        self.lmax = lmax
        self.channel_radius_fm = channel_radius_fm
        self.a = channel_radius_fm * kinematics.k

        self.kinematics = kinematics
        self.reaction = reaction
        self.sys = ProjectileTargetSystem(
            self.a,
            lmax,
            mass_target=self.reaction.target.m0,
            mass_projectile=self.reaction.projectile.m0,
            Ztarget=self.reaction.target.Z,
            Zproj=self.reaction.projectile.Z,
            coupling=spin_half_orbit_coupling,
        )
        self.solver = solver

        self.free_matrices = self.solver.free_matrix(self.a, self.sys.l, coupled=False)
        self.basis_boundary = self.solver.precompute_boundaries(self.a)

        channels, asymptotics = self.sys.get_partial_wave_channels(*self.kinematics)
        self.channels = [channel.decouple() for channel in channels]
        self.asymptotics = [asym.decouple() for asym in asymptotics]
        self.l_dot_s = np.array(
            [np.diag(coupling) for coupling in self.sys.couplings[1:]]
        )
        self.ls = self.sys.l[:, np.newaxis]

    def radial_grid(self) -> FloatArray:
        """Return the physical quadrature grid used for local potentials."""
        return self.solver.radial_grid(self.a, self.kinematics.k)

    def _local_potential(self, potential: npt.ArrayLike, name: str) -> ComplexArray:
        """Validate and cast a local potential array on the quadrature grid."""
        potential_array = np.asarray(potential, dtype=np.complex128)
        expected_shape = (self.solver.kernel.quadrature.nbasis,)
        if potential_array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        return potential_array

    def _optional_local_potential(
        self, potential: npt.ArrayLike | None, name: str
    ) -> ComplexArray:
        """Return a validated local potential or a zero array when omitted."""
        if potential is None:
            return np.zeros(self.solver.kernel.quadrature.nbasis, dtype=np.complex128)
        return self._local_potential(potential, name)

    def smatrix(
        self,
        central_potential: npt.ArrayLike,
        spin_orbit_potential: npt.ArrayLike | None = None,
        coulomb_potential: npt.ArrayLike | None = None,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Compute the elastic S-matrix for ``j=l±1/2`` channels."""
        splus = np.zeros(self.sys.lmax + 1, dtype=np.complex128)
        sminus = np.zeros(self.sys.lmax + 1, dtype=np.complex128)
        central_array = self._local_potential(central_potential, "central_potential")
        spin_orbit_array = self._optional_local_potential(
            spin_orbit_potential, "spin_orbit_potential"
        )

        im_central = self.solver.interaction_matrix(
            self.channels[0][0].k[0],
            self.channels[0][0].E[0],
            self.channels[0][0].a,
            self.channels[0][0].size,
            local_potential=central_array,
        )
        im_spin_orbit = self.solver.interaction_matrix(
            self.channels[0][0].k[0],
            self.channels[0][0].E[0],
            self.channels[0][0].a,
            self.channels[0][0].size,
            local_potential=spin_orbit_array,
        )
        if coulomb_potential is not None:
            coulomb_array = self._local_potential(
                coulomb_potential, "coulomb_potential"
            )
            im_coulomb = self.solver.interaction_matrix(
                self.channels[0][0].k[0],
                self.channels[0][0].E[0],
                self.channels[0][0].a,
                self.channels[0][0].size,
                local_potential=coulomb_array,
            )
            im_central += im_coulomb

        _, s0, _ = self.solver.solve(
            self.channels[0][0],
            self.asymptotics[0][0],
            free_matrix=self.free_matrices[0],
            interaction_matrix=im_central,
            basis_boundary=self.basis_boundary,
        )
        splus[0] = s0[0, 0]
        last_l = 0

        for l in self.sys.l[1:]:
            channel = self.channels[l]
            asymptotic = self.asymptotics[l]
            lds = self.l_dot_s[l - 1]
            _, sp, _ = self.solver.solve(
                channel[0],
                asymptotic[0],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_central + lds[0] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )
            splus[l] = sp[0, 0]

            _, sm, _ = self.solver.solve(
                channel[1],
                asymptotic[1],
                free_matrix=self.free_matrices[l],
                interaction_matrix=im_central + lds[1] * im_spin_orbit,
                basis_boundary=self.basis_boundary,
            )
            sminus[l] = sm[0, 0]

            last_l = int(l)
            if (np.absolute(1 - splus[l])) < self.smatrix_abs_tol and (
                np.absolute(1 - sminus[l])
            ) < self.smatrix_abs_tol:
                break

        return splus[: last_l + 1], sminus[: last_l + 1]

    def xs(
        self,
        central_potential: npt.ArrayLike,
        spin_orbit_potential: npt.ArrayLike | None = None,
        coulomb_potential: npt.ArrayLike | None = None,
    ) -> tuple[float, float]:
        """Return total and reaction cross sections in mb."""
        splus, sminus = self.smatrix(
            central_potential,
            spin_orbit_potential,
            coulomb_potential,
        )
        return integral_elastic_xs(self.kinematics.k, splus, sminus, self.ls)

    def transmission_coefficients(
        self,
        central_potential: npt.ArrayLike,
        spin_orbit_potential: npt.ArrayLike | None = None,
        coulomb_potential: npt.ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return transmission coefficients for ``j=l±1/2`` channels."""
        splus, sminus = self.smatrix(
            central_potential,
            spin_orbit_potential,
            coulomb_potential,
        )
        return 1.0 - np.absolute(splus) ** 2, 1.0 - np.absolute(sminus) ** 2


class DifferentialWorkspace:
    """Workspace for angular elastic-scattering observables."""

    @classmethod
    def build_from_system(
        cls: type[DifferentialWorkspace],
        reaction: Reaction,
        kinematics: ChannelKinematics,
        channel_radius_fm: float,
        solver: Solver,
        lmax: int,
        angles: FloatArray,
        smatrix_abs_tol: float = 1e-6,
    ) -> DifferentialWorkspace:
        """Construct a differential workspace from the raw system inputs."""
        integral_workspace = IntegralWorkspace(
            reaction, kinematics, channel_radius_fm, solver, lmax, smatrix_abs_tol
        )
        return cls(integral_workspace, angles)

    def __init__(
        self, integral_workspace: IntegralWorkspace, angles: FloatArray
    ) -> None:
        """Precompute angular factors for differential observables."""
        self.integral_workspace = integral_workspace
        self.reaction = self.integral_workspace.reaction
        self.kinematics = self.integral_workspace.kinematics

        check_angles(angles)
        self.angles = angles
        self.ls = self.integral_workspace.ls
        self.P_l_costheta = eval_legendre(self.ls, np.cos(self.angles))
        self.P_1_l_costheta = lpmv(1, self.ls, np.cos(self.angles))

        self.Zz = self.reaction.projectile.Z * self.reaction.target.Z
        self.sigma_l = self.coulomb_phase_shift(self.ls.astype(np.float64))
        if self.Zz > 0:
            self.rutherford: FloatArray | None = self.rutherford_xs(self.angles)
            self.f_c: FloatArray | ComplexArray = self.coulomb_amplitude(
                self.angles, self.sigma_l[0]
            )
        else:
            self.f_c = np.zeros_like(angles)
            self.rutherford = None

    def radial_grid(self) -> FloatArray:
        """Return the physical quadrature grid used for local potentials."""
        return self.integral_workspace.radial_grid()

    def rutherford_xs(self, angles: FloatArray) -> FloatArray:
        """Return the Rutherford cross section in mb/sr."""
        check_angles(angles)
        sin2 = np.sin(angles / 2.0) ** 2
        return 10 * self.kinematics.eta**2 / (4 * self.kinematics.k**2 * sin2**2)

    def coulomb_amplitude(self, angles: FloatArray, sigma_0: float) -> ComplexArray:
        """Return the Coulomb scattering amplitude."""
        sin2 = np.sin(angles / 2.0)
        return np.asarray(
            -self.kinematics.eta
            / (2 * self.kinematics.k * sin2**2)
            * np.exp(2j * sigma_0 - 2j * self.kinematics.eta * np.log(sin2)),
            dtype=np.complex128,
        )

    def coulomb_phase_shift(self, ls: FloatArray) -> FloatArray:
        """Return Coulomb phase shifts for the supplied partial waves."""
        return np.angle(gamma(1 + ls + 1j * self.kinematics.eta))

    def xs(
        self,
        central_potential: npt.ArrayLike,
        spin_orbit_potential: npt.ArrayLike | None = None,
        coulomb_potential: npt.ArrayLike | None = None,
    ) -> ElasticXS:
        """Return differential and integral elastic observables."""
        splus, sminus = self.integral_workspace.smatrix(
            central_potential,
            spin_orbit_potential,
            coulomb_potential,
        )
        return ElasticXS(
            *differential_elastic_xs(
                self.kinematics.k,
                self.angles,
                splus,
                sminus,
                self.ls,
                self.P_l_costheta,
                self.P_1_l_costheta,
                self.f_c,
                self.sigma_l,
            )
        )


@njit
def integral_elastic_xs(
    k: float,
    Splus: np.ndarray,
    Sminus: np.ndarray,
    ls: np.ndarray,
) -> tuple[float, float]:
    """Return total and reaction cross sections for spin-1/2 on spin-0 scattering."""
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
    angles: np.ndarray,
    splus: np.ndarray,
    sminus: np.ndarray,
    ls: np.ndarray,
    P_l_costheta: np.ndarray,
    P_1_l_costheta: np.ndarray,
    f_c: npt.ArrayLike = 0,
    sigma_l: npt.ArrayLike = 0,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return differential and integral elastic observables.

    The returned tuple contains ``(dσ/dΩ, A_y, Q, σ_total, σ_reaction)``.
    """
    _sigma_l = np.asarray(sigma_l, dtype=np.float64)
    _f_c = np.asarray(f_c, dtype=np.complex128)
    a = np.zeros_like(angles, dtype=np.complex128) + _f_c
    b = np.zeros_like(angles, dtype=np.complex128)

    xsrxn = 0.0
    xst = 0.0

    for l in range(splus.shape[0]):
        phase = np.exp(2j * _sigma_l[l]) / (2j * k)

        a += (
            P_l_costheta[l, :]
            * phase
            * ((l + 1) * (splus[l] - 1) + l * (sminus[l] - 1))
        )
        b += P_1_l_costheta[l, :] * phase * (splus[l] - sminus[l])

        xsrxn += (l + 1) * (1 - np.abs(splus[l]) ** 2) + l * (
            1 - np.abs(sminus[l]) ** 2
        )
        xst += (l + 1) * (1 - np.real(splus[l])) + l * (1 - np.real(sminus[l]))

    dsdo0 = np.abs(a) ** 2 + np.abs(b) ** 2
    dsdo = dsdo0 * 10.0
    denom = np.maximum(dsdo0, eps)

    Ay = 2.0 * np.imag(np.conjugate(a) * b) / denom
    Q = 2.0 * np.real(np.conjugate(a) * b) / denom

    xsrxn *= 10.0 * np.pi / k**2
    xst *= 10.0 * 2.0 * np.pi / k**2
    return dsdo, Ay, Q, xst, xsrxn


def check_angles(angles: FloatArray) -> None:
    """Validate that the angle grid is one-dimensional and lies on ``[0, π)``."""
    if angles.ndim != 1:
        raise ValueError("angles must be a 1D array")
    if angles[0] < 0 or angles[-1] > np.pi:
        raise ValueError("angles must a grid in radians on [0,pi)")
