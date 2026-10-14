r"""Coupled-channels Lane workspace for quasi-elastic ``(p,n)`` to the IAS.

The proton (entrance) and neutron (isobaric analog) channels are coupled by the
isovector transition potential :math:`U_1`. The Lane coupling is a scalar plus
a spin-orbit term, so it conserves :math:`l` and :math:`j`, and for each
:math:`(l, j)` the radial problem is the 2x2 system

.. math::
    \left[T_l + U_{pp} - E_p\right] u_p + U_{1} u_n = 0, \qquad
    \left[T_l + U_{nn} - E_n\right] u_n + U_{1} u_p = 0,

with an incoming wave in the proton channel only. It is solved exactly with the
R-matrix method, which gives the full 2x2 R- and S-matrices. The first-order
(Born) approximation to the off-diagonal S-matrix element is the DWBA of
:mod:`jitr.xs.quasielastic_pn`.

This is the minimal reference example of a coupled-channels calculation in
jitR: channels with different wavenumbers, reduced masses and Sommerfeld
parameters sharing one physical channel radius.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gamma

from ..reactions import ProjectileTargetSystem, Reaction, spin_half_orbit_coupling
from ..rmatrix import Solver
from ..utils.kinematics import ChannelKinematics
from .elastic import check_angles
from .quasielastic_pn import (
    QuasielasticPnXS,
    isovector_factor,
    pn_observables,
    pn_potentials,
    spin_half_transition_geometry,
)

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]

PROTON = 0
NEUTRON = 1


class Workspace:
    r"""
    Workspace for coupled-channels Lane (p,n) scattering to the isobaric
    analog state.

    Channel 0 is the proton (entrance) channel and channel 1 the neutron
    (exit) channel. Both share the channel radius ``channel_radius_fm``.
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
    ) -> None:
        r"""
        Initialize the coupled-channels (p,n) workspace.

        Args:
            reaction: Reaction object containing information about the target,
                projectile, residual, and product.
            kinematics_entrance: Kinematics for the proton channel.
            kinematics_exit: Kinematics for the neutron channel.
            solver: R-matrix solver.
            angles: Angles in radians at which to compute the differential
                cross section.
            lmax: The maximum orbital angular momentum.
            channel_radius_fm: The channel radius in femtometers.
        """
        if reaction.residual is None or reaction.product is None:
            raise ValueError(
                "Reaction must define both residual and product for (p,n) scattering"
            )
        check_angles(angles)

        self.reaction = reaction
        self.kinematics_entrance = kinematics_entrance
        self.kinematics_exit = kinematics_exit
        self.solver = solver
        self.angles = angles
        self.lmax = lmax
        self.channel_radius_fm = channel_radius_fm
        self.isovector_factor = isovector_factor(reaction)

        k = np.array([kinematics_entrance.k, kinematics_exit.k], dtype=np.float64)
        mu = np.array([kinematics_entrance.mu, kinematics_exit.mu], dtype=np.float64)
        eta = np.array([kinematics_entrance.eta, kinematics_exit.eta], dtype=np.float64)

        # one 2-channel (p, n) system per l; j enters only through l . sigma
        self.sys = ProjectileTargetSystem(
            channel_radius=channel_radius_fm * k[PROTON],
            lmax=lmax,
            mass_target=reaction.target.m0,
            mass_projectile=reaction.projectile.m0,
            Ztarget=reaction.target.Z,
            Zproj=reaction.projectile.Z,
            coupling=lambda l: np.eye(2),
        )
        # Elab and Ecm are unused; the per-channel energy is hbar^2 k^2 / (2 mu)
        self.channels, self.asymptotics = self.sys.get_partial_wave_channels(
            kinematics_entrance.Elab, kinematics_entrance.Ecm, mu, k, eta
        )
        self.free_matrices = [
            np.asarray(self.solver.free_matrix(ch.a, ch.l, ch.E, ch.mu, coupled=True))
            for ch in self.channels
        ]
        self.basis_boundary = self.solver.precompute_boundaries(self.sys.channel_radius)

        # l . sigma for j = l + 1/2, l - 1/2
        self.l_dot_s = [np.diag(spin_half_orbit_coupling(l)) for l in range(lmax + 1)]

        # Coulomb phases; the neutron channel has eta = 0
        l = np.arange(lmax + 1)
        self.sigma_c = np.angle(gamma(1 + l + 1j * eta[PROTON])) + np.angle(
            gamma(1 + l + 1j * eta[NEUTRON])
        )
        self.geometric_factor = (
            np.sqrt(4 * np.pi)
            / (2j * k[PROTON])
            * np.exp(1j * self.sigma_c)[:, np.newaxis, np.newaxis]
            * spin_half_transition_geometry(lmax, angles)
        )

    def radial_grid(self) -> FloatArray:
        """Return the physical quadrature grid used for local potentials."""
        return self.solver.radial_grid(
            self.sys.channel_radius, self.kinematics_entrance.k
        )

    def _coupled_interaction_matrix(
        self, Vpp: ComplexArray, Vnn: ComplexArray, Vpn: ComplexArray
    ) -> ComplexArray:
        """Interaction matrix for the symmetric 2x2 local potential."""
        ch = self.channels[0]
        return self.solver.interaction_matrix(
            ch.k[PROTON],
            ch.E[PROTON],
            ch.a,
            ch.size,
            local_potential=np.array([[Vpp, Vpn], [Vpn, Vnn]]),
        )

    def rsmatrix(
        self,
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike | None = None,
        U_n_central: npt.ArrayLike | None = None,
        U_n_spin_orbit: npt.ArrayLike | None = None,
        U1_central: npt.ArrayLike | None = None,
        U1_spin_orbit: npt.ArrayLike | None = None,
    ) -> tuple[ComplexArray, ComplexArray]:
        """
        Solve the coupled (p,n) channels for every partial wave.

        Args:
            U_p_coulomb: Coulomb interaction for the proton.
            U_p_central: Central interaction for the proton.
            U_p_spin_orbit: Spin-orbit interaction for the proton.
            U_n_central: Central interaction for the neutron.
            U_n_spin_orbit: Spin-orbit interaction for the neutron.
            U1_central: Central (p,n) coupling potential on the quadrature
                grid, used as-is. If None, defaults to
                ``-(U_n_central - U_p_central) * isovector_factor``.
            U1_spin_orbit: Spin-orbit (p,n) coupling potential on the
                quadrature grid, used as-is. If None, defaults to
                ``-(U_n_spin_orbit - U_p_spin_orbit) * isovector_factor``.

        Returns:
            Tuple ``(R, S)`` of complex arrays with shape
            ``(lmax + 1, 2, 2, 2)`` indexed by ``[l, j, out, in]``, where
            ``j`` indexes ``(l + 1/2, l - 1/2)`` and channels are
            ``(p, n)``. ``S`` is flux-normalized, so ``S[l, j, 1, 0]`` is the
            (p,n) element. Entries for ``l = 0, j = l - 1/2`` are zero.
        """
        potentials = pn_potentials(
            self.solver.kernel.quadrature.nbasis,
            self.isovector_factor,
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
            U1_central,
            U1_spin_orbit,
        )
        # local interactions do not depend on l, and the spin-orbit part enters
        # linearly with strength l . sigma, so build each piece once
        im_central = self._coupled_interaction_matrix(
            potentials["U_p_central"] + potentials["U_p_coulomb"],
            potentials["U_n_central"],
            potentials["U1_central"],
        )
        im_spin_orbit = self._coupled_interaction_matrix(
            potentials["U_p_spin_orbit"],
            potentials["U_n_spin_orbit"],
            potentials["U1_spin_orbit"],
        )

        R = np.zeros((self.lmax + 1, 2, 2, 2), dtype=np.complex128)
        S = np.zeros((self.lmax + 1, 2, 2, 2), dtype=np.complex128)
        for l in range(self.lmax + 1):
            for ij, l_dot_s in enumerate(self.l_dot_s[l]):
                R[l, ij], S[l, ij], _ = self.solver.solve(
                    self.channels[l],
                    self.asymptotics[l],
                    interaction_matrix=im_central + l_dot_s * im_spin_orbit,
                    free_matrix=self.free_matrices[l],
                    basis_boundary=self.basis_boundary,
                )
        return R, S

    def xs(
        self,
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike | None = None,
        U_n_central: npt.ArrayLike | None = None,
        U_n_spin_orbit: npt.ArrayLike | None = None,
        U1_central: npt.ArrayLike | None = None,
        U1_spin_orbit: npt.ArrayLike | None = None,
    ) -> FloatArray:
        """
        Differential (p,n) cross section in mb/sr in the outgoing neutron
        angle, from the coupled-channels S-matrix.

        Args are as for :meth:`rsmatrix`.

        Returns:
            Differential cross section at ``self.angles`` in mb/sr.
        """
        _, S = self.rsmatrix(
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
            U1_central,
            U1_spin_orbit,
        )
        return self.xs_from_smatrix(S)

    def observables(
        self,
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike | None = None,
        U_n_central: npt.ArrayLike | None = None,
        U_n_spin_orbit: npt.ArrayLike | None = None,
        U1_central: npt.ArrayLike | None = None,
        U1_spin_orbit: npt.ArrayLike | None = None,
    ) -> QuasielasticPnXS:
        """
        Differential cross section, analyzing power and spin-rotation function
        in the outgoing neutron angle, from the coupled-channels S-matrix.

        Args are as for :meth:`rsmatrix`.

        Returns:
            Observables at ``self.angles``; the cross section is in mb/sr.
        """
        _, S = self.rsmatrix(
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
            U1_central,
            U1_spin_orbit,
        )
        return self.observables_from_smatrix(S)

    def integrated_xs(
        self,
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike | None = None,
        U_n_central: npt.ArrayLike | None = None,
        U_n_spin_orbit: npt.ArrayLike | None = None,
        U1_central: npt.ArrayLike | None = None,
        U1_spin_orbit: npt.ArrayLike | None = None,
    ) -> float:
        """
        Angle-integrated (p,n) cross section in mb, from partial waves.

        Args are as for :meth:`rsmatrix`.

        Returns:
            Integrated cross section in mb.
        """
        _, S = self.rsmatrix(
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
            U1_central,
            U1_spin_orbit,
        )
        return self.integrated_xs_from_smatrix(S)

    def xs_from_smatrix(self, S: ComplexArray) -> FloatArray:
        r"""
        Differential (p,n) cross section in mb/sr from the coupled S-matrix.

        .. math::
            f_{m m'}(\theta) = \frac{\sqrt{4\pi}}{2 i k_p} \sum_{lj}
            \sqrt{2l+1}\langle l 0 \tfrac{1}{2} m | j m \rangle
            \langle l, m-m'; \tfrac{1}{2} m' | j m \rangle
            e^{i(\sigma_l^p + \sigma_l^n)} S^{lj}_{np} Y_l^{m-m'}(\theta, 0),
            \qquad
            \frac{d\sigma}{d\Omega} = \frac{1}{2}\sum_{m m'} |f_{m m'}|^2.

        Args:
            S: Flux-normalized S-matrix from :meth:`rsmatrix`.

        Returns:
            Differential cross section at ``self.angles`` in mb/sr.
        """
        return self.observables_from_smatrix(S).dsdo

    def amplitudes_from_smatrix(self, S: ComplexArray) -> ComplexArray:
        r"""
        Spin-1/2 transition amplitude matrix from the coupled S-matrix.

        .. math::
            f_{m m'}(\theta) = \frac{\sqrt{4\pi}}{2 i k_p} \sum_{lj}
            \sqrt{2l+1}\langle l 0 \tfrac{1}{2} m | j m \rangle
            \langle l, m-m'; \tfrac{1}{2} m' | j m \rangle
            e^{i(\sigma_l^p + \sigma_l^n)} S^{lj}_{np} Y_l^{m-m'}(\theta, 0)

        Args:
            S: Flux-normalized S-matrix from :meth:`rsmatrix`.

        Returns:
            Amplitudes :math:`f_{mm'}(\theta)` with shape
            ``(2, 2, len(self.angles))``.
        """
        return np.einsum(
            "abljt,lj->abt", self.geometric_factor, S[:, :, NEUTRON, PROTON]
        )

    def observables_from_smatrix(self, S: ComplexArray) -> QuasielasticPnXS:
        r"""
        Observables from the coupled S-matrix,
        :math:`\frac{d\sigma}{d\Omega} = \frac{1}{2}\sum_{m m'} |f_{m m'}|^2`
        and the analyzing power and spin-rotation function of
        :func:`jitr.xs.quasielastic_pn.pn_observables`.

        Args:
            S: Flux-normalized S-matrix from :meth:`rsmatrix`.

        Returns:
            Observables at ``self.angles``; the cross section is in mb/sr.
        """
        return pn_observables(self.amplitudes_from_smatrix(S))

    def integrated_xs_from_smatrix(self, S: ComplexArray) -> float:
        r"""
        Angle-integrated (p,n) cross section in mb from the coupled S-matrix,
        :math:`\sigma = \frac{\pi}{k_p^2}\sum_{lj} \frac{2j+1}{2}|S^{lj}_{np}|^2`.

        Args:
            S: Flux-normalized S-matrix from :meth:`rsmatrix`.

        Returns:
            Integrated cross section in mb.
        """
        l = np.arange(self.lmax + 1)[:, np.newaxis]
        two_j_plus_1 = np.hstack([2 * l + 2, 2 * l])
        return float(
            10
            * np.pi
            / self.kinematics_entrance.k**2
            * np.sum(two_j_plus_1 / 2 * np.abs(S[:, :, NEUTRON, PROTON]) ** 2)
        )
