"""DWBA workspaces for quasi-elastic ``(p,n)`` scattering observables."""

from dataclasses import dataclass

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


@dataclass
class QuasielasticPnXS:
    """Container for quasi-elastic ``(p,n)`` observables.

    Attributes:
        dsdo: Differential cross section in mb/sr.
        Ay: Analyzing power.
        Q: Spin-rotation function.
    """

    dsdo: FloatArray
    Ay: FloatArray
    Q: FloatArray


def isovector_factor(reaction: Reaction) -> float:
    r"""Return :math:`\sqrt{|N-Z|}/(N-Z-1)` for the target of ``reaction``.

    This scales the difference of the neutron and proton optical potentials
    into the default (p,n) transition potential.
    """
    A = reaction.target.A
    Z = reaction.target.Z
    N = A - Z
    if N - Z == 1:
        raise ValueError(
            f"the (p,n) isovector factor diverges for N - Z = 1 targets like "
            f"{reaction.target}; supply U1_central and U1_spin_orbit explicitly"
        )
    return float(np.sqrt(np.fabs(N - Z)) / (N - Z - 1))


def as_local_potential(
    potential: npt.ArrayLike, nbasis: int, name: str
) -> ComplexArray:
    """Validate and cast a local potential array on the quadrature grid."""
    potential_array = np.asarray(potential, dtype=np.complex128)
    if potential_array.shape != (nbasis,):
        raise ValueError(f"{name} must have shape {(nbasis,)}")
    return potential_array


def as_optional_local_potential(
    potential: npt.ArrayLike | None, nbasis: int, name: str
) -> ComplexArray:
    """Return a validated local potential or a zero array when omitted."""
    if potential is None:
        return np.zeros(nbasis, dtype=np.complex128)
    return as_local_potential(potential, nbasis, name)


def pn_potentials(
    nbasis: int,
    isovector_factor: float,
    U_p_coulomb: npt.ArrayLike,
    U_p_central: npt.ArrayLike,
    U_p_spin_orbit: npt.ArrayLike | None = None,
    U_n_central: npt.ArrayLike | None = None,
    U_n_spin_orbit: npt.ArrayLike | None = None,
    U1_central: npt.ArrayLike | None = None,
    U1_spin_orbit: npt.ArrayLike | None = None,
) -> dict[str, ComplexArray]:
    """Validate the (p,n) potentials and fill in the default transition terms.

    Args:
        nbasis: Size of the quadrature grid.
        isovector_factor: Scale applied to ``U_n - U_p`` in the default
            transition potentials (see :func:`isovector_factor`).
        U_p_coulomb: Coulomb interaction for the proton.
        U_p_central: Central interaction for the proton.
        U_p_spin_orbit: Spin-orbit interaction for the proton.
        U_n_central: Central interaction for the neutron (required).
        U_n_spin_orbit: Spin-orbit interaction for the neutron.
        U1_central: Central transition potential, used as-is. Defaults to
            ``-(U_n_central - U_p_central) * isovector_factor``.
        U1_spin_orbit: Spin-orbit transition potential, used as-is. Defaults
            to ``-(U_n_spin_orbit - U_p_spin_orbit) * isovector_factor``.

    Returns:
        Validated complex arrays keyed by argument name; omitted spin-orbit
        terms are zero.
    """
    if U_n_central is None:
        raise TypeError("U_n_central is required")
    potentials = {
        "U_p_coulomb": as_local_potential(U_p_coulomb, nbasis, "U_p_coulomb"),
        "U_p_central": as_local_potential(U_p_central, nbasis, "U_p_central"),
        "U_p_spin_orbit": as_optional_local_potential(
            U_p_spin_orbit, nbasis, "U_p_spin_orbit"
        ),
        "U_n_central": as_local_potential(U_n_central, nbasis, "U_n_central"),
        "U_n_spin_orbit": as_optional_local_potential(
            U_n_spin_orbit, nbasis, "U_n_spin_orbit"
        ),
    }
    if U1_central is None:
        potentials["U1_central"] = (
            -(potentials["U_n_central"] - potentials["U_p_central"]) * isovector_factor
        )
    else:
        potentials["U1_central"] = as_local_potential(U1_central, nbasis, "U1_central")
    if U1_spin_orbit is None:
        potentials["U1_spin_orbit"] = (
            -(potentials["U_n_spin_orbit"] - potentials["U_p_spin_orbit"])
            * isovector_factor
        )
    else:
        potentials["U1_spin_orbit"] = as_local_potential(
            U1_spin_orbit, nbasis, "U1_spin_orbit"
        )
    return potentials


def spin_half_transition_geometry(lmax: int, angles: FloatArray) -> ComplexArray:
    r"""Angular factors for a spin-1/2 transition on a spin-0 target.

    For a transition that conserves :math:`l` and :math:`j`, the amplitude
    for projectile spin projection :math:`m \to m'` is a sum over partial
    waves of

    .. math::
        \sqrt{2l+1} \langle l 0 \tfrac{1}{2} m | j m \rangle
        \langle l, m-m'; \tfrac{1}{2} m' | j m \rangle Y_l^{m-m'}(\theta, 0)

    times a partial-wave amplitude.

    Args:
        lmax: Maximum orbital angular momentum.
        angles: Scattering angles in radians.

    Returns:
        Array of shape ``(2, 2, lmax + 1, 2, len(angles))`` indexed by
        ``[m, m', l, j]``, with ``m, m'`` in ``(-1/2, +1/2)`` and ``j`` in
        ``(l + 1/2, l - 1/2)``. Entries with no allowed ``j`` are zero.
    """
    geometry = np.zeros((2, 2, lmax + 1, 2, angles.shape[0]), dtype=np.complex128)
    for im, m in enumerate([-0.5, 0.5]):
        for imp, mp in enumerate([-0.5, 0.5]):
            for l in range(0, lmax + 1):
                if abs(m - mp) > l:
                    continue
                ylm = sph_harm_y(l, int(m - mp), angles, 0)
                for ijp, jp in enumerate(
                    [l + 1 / 2, l - 1 / 2] if l > 0 else [l + 1 / 2]
                ):
                    cg0 = float(clebsch_gordan(l, 1 / 2, jp, m - mp, mp, m))
                    cg1 = float(clebsch_gordan(l, 1 / 2, jp, 0, m, m))
                    geometry[im, imp, l, ijp, :] = cg1 * cg0 * np.sqrt(2 * l + 1) * ylm
    return geometry


def pn_observables(
    f: ComplexArray, xs_factor: float = 0.5, eps: float = 1e-30
) -> QuasielasticPnXS:
    r"""Observables from the spin-1/2 transition amplitude matrix.

    The scattering plane is taken at :math:`\phi = 0`, so the normal is
    :math:`\hat n = \hat k_{in} \times \hat k_{out} = \hat y` and, for a
    transition with :math:`l = s = j = 0` transfer, the amplitude matrix is

    .. math::
        M = A + B\, \sigma \cdot \hat n ,

    with :math:`A` the non-spin-flip and :math:`B` the spin-flip amplitude.
    In the ``f[m, m']`` basis of :func:`spin_half_transition_geometry` this is
    :math:`A = f[1, 1] = f[0, 0]` and
    :math:`\langle -|M|+\rangle = f[1, 0] = i B`, which gives

    .. math::
        \frac{d\sigma}{d\Omega} = \frac{1}{2}\sum_{mm'}|f_{mm'}|^2
        = |A|^2 + |B|^2, \qquad
        A_y = \frac{2\,\mathrm{Im}(A^* f[1,0])}{|A|^2 + |B|^2}, \qquad
        Q = \frac{2\,\mathrm{Re}(A^* f[1,0])}{|A|^2 + |B|^2}.

    This is the same convention as :func:`jitr.xs.elastic.differential_elastic_xs`
    and as Eq. (12) of Gosset, Mayer and Escudie, Phys. Rev. C 14, 878 (1976).

    The cross section is the full sum over ``m, m'``. The analyzing power and
    spin-rotation function, on the other hand, are only meaningful when ``f``
    really does reduce to two amplitudes, i.e. when the transition conserves
    ``l`` and ``j`` on a spin-0 target so that ``f[0, 0] == f[1, 1]`` and
    ``f[0, 1] == -f[1, 0]``.

    Args:
        f: Amplitude matrix with shape ``(2, 2, len(angles))`` indexed by
            ``[m, m', theta]``, with ``m, m'`` in ``(-1/2, +1/2)``.
        xs_factor: Overall factor multiplying :math:`\sum_{mm'}|f_{mm'}|^2` to
            give the cross section in fm^2/sr. Defaults to the ``1/(2s+1)``
            spin average of a flux-normalized S-matrix amplitude.
        eps: Floor on the cross section used to regularize the ratios.

    Returns:
        The differential cross section in mb/sr, the analyzing power and the
        spin-rotation function at each angle.
    """
    total = np.sum(np.absolute(f) ** 2, axis=(0, 1))
    denom = np.maximum(0.5 * total, eps)
    return QuasielasticPnXS(
        dsdo=10.0 * xs_factor * total,
        Ay=2.0 * np.imag(np.conjugate(f[1, 1]) * f[1, 0]) / denom,
        Q=2.0 * np.real(np.conjugate(f[1, 1]) * f[1, 0]) / denom,
    )


class System:
    r"""
    System for (p,n) quasi-elastic scattering observables for local interactions
    This system contains the entrance and exit channels, which are defined by the
    projectile and target masses, charges, and the channel radius.

    Attributes:
        channel_radius_fm: The channel radius in femtometers.
        lmax: The maximum angular momentum quantum number.
        l: An array of angular momentum quantum numbers from 0 to lmax.
        entrance: The entrance channel system, including projectile/target
            masses, charges, and the channel radius.
        exit: The exit channel system, including product/residual masses,
            charges, and the channel radius.
    """

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

        Args:
            channel_radius_fm: The channel radius in femtometers.
            lmax: The maximum angular momentum quantum number.
            reaction: Reaction object containing information about the target,
                projectile, residual, and product.
            kinematics_entrance: Kinematics for the entrance channel.
            kinematics_exit: Kinematics for the exit channel.
        """

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

        Args:
            reaction: Reaction object containing information about the target,
                projectile, residual, and product.
            kinematics_entrance: Kinematics for the entrance channel.
            kinematics_exit: Kinematics for the exit channel.
            solver: Solver used to compute the distorted waves and interaction
                matrices.
            angles: Angles in radians at which to compute the differential
                cross section.
            lmax: The maximum angular momentum quantum number.
            channel_radius_fm: The channel radius in femtometers.
            tmatrix_abs_tol: The absolute tolerance for the transition matrix
                elements.
        """

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
        self.isovector_factor = isovector_factor(self.reaction)

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
        self.sigma_c = np.angle(
            gamma(1 + self.sys.l + 1j * self.kinematics_entrance.eta)
        )
        # (-1)^(2j+1) = 1 for half-integer j
        self.geometric_factor = (
            (4 * np.pi) ** (3.0 / 2.0)
            / (self.kinematics_entrance.k * self.kinematics_exit.k)
            * np.exp(1j * self.sigma_c)[:, np.newaxis, np.newaxis]
            * spin_half_transition_geometry(self.sys.lmax, self.angles)
        )

    def radial_grid(self) -> FloatArray:
        """Return the physical quadrature grid used for local potentials."""
        return self.solver.radial_grid(
            self.p_channels[0][0].a, self.kinematics_entrance.k
        )

    def tmatrix(
        self,
        U_p_coulomb: npt.ArrayLike,
        U_p_central: npt.ArrayLike,
        U_p_spin_orbit: npt.ArrayLike | None = None,
        U_n_central: npt.ArrayLike | None = None,
        U_n_spin_orbit: npt.ArrayLike | None = None,
        U1_central: npt.ArrayLike | None = None,
        U1_spin_orbit: npt.ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the transition matrix for (p,n) quasi-elastic scattering
        using the distorted wave Born approximation (DWBA).

        Args:
            U_p_coulomb: Coulomb interaction for the proton.
            U_p_central: Central interaction for the proton.
            U_p_spin_orbit: Spin-orbit interaction for the proton.
            U_n_central: Central interaction for the neutron.
            U_n_spin_orbit: Spin-orbit interaction for the neutron.
            U1_central: Central (p,n) transition potential on the quadrature
                grid, used as-is in the radial integral. If None, defaults to
                ``-(U_n_central - U_p_central) * isovector_factor``.
            U1_spin_orbit: Spin-orbit (p,n) transition potential on the
                quadrature grid, used as-is in the radial integral. If None,
                defaults to
                ``-(U_n_spin_orbit - U_p_spin_orbit) * isovector_factor``.

        Returns:
            Tuple (Tpn, Sn, Sp) where Tpn is the transition matrix for the
            (p,n) reaction, Sn is the S-matrix for the neutron elastic exit
            channel, and Sp is the S-matrix for the proton elastic entrance
            channel.
        """
        Tpn = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)
        Sn = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)
        Sp = np.zeros((self.sys.lmax + 1, 2), dtype=np.complex128)

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
        proton_central = potentials["U_p_central"]
        proton_spin_orbit = potentials["U_p_spin_orbit"]
        proton_coulomb = potentials["U_p_coulomb"]
        neutron_central = potentials["U_n_central"]
        neutron_spin_orbit = potentials["U_n_spin_orbit"]
        transition_central = potentials["U1_central"]
        transition_spin_orbit = potentials["U1_spin_orbit"]

        # precomute central, spin-obit, and Coulomb interaction matrices
        # for entrance channel distorted waves

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
                np.sum(xp * (transition_central + l_dot_s * transition_spin_orbit) * xn)
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
        U_p_spin_orbit: npt.ArrayLike | None = None,
        U_n_central: npt.ArrayLike | None = None,
        U_n_spin_orbit: npt.ArrayLike | None = None,
        U1_central: npt.ArrayLike | None = None,
        U1_spin_orbit: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Calculate the differential cross section for (p,n) quasi-elastic
        scattering in mb/Sr in the outgoing neutron angle using DWBA.

        Args:
            U_p_coulomb: Coulomb interaction for the proton.
            U_p_central: Central interaction for the proton.
            U_p_spin_orbit: Spin-orbit interaction for the proton.
            U_n_central: Central interaction for the neutron.
            U_n_spin_orbit: Spin-orbit interaction for the neutron.
            U1_central: Central (p,n) transition potential on the quadrature
                grid, used as-is in the radial integral. If None, defaults to
                ``-(U_n_central - U_p_central) * isovector_factor``.
            U1_spin_orbit: Spin-orbit (p,n) transition potential on the
                quadrature grid, used as-is in the radial integral. If None,
                defaults to
                ``-(U_n_spin_orbit - U_p_spin_orbit) * isovector_factor``.

        Returns:
            Differential cross section for the (p,n) reaction in mb/Sr.
        """
        return self.observables(
            U_p_coulomb=U_p_coulomb,
            U_p_central=U_p_central,
            U_p_spin_orbit=U_p_spin_orbit,
            U_n_central=U_n_central,
            U_n_spin_orbit=U_n_spin_orbit,
            U1_central=U1_central,
            U1_spin_orbit=U1_spin_orbit,
        ).dsdo

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
        for (p,n) quasi-elastic scattering in DWBA.

        Args are as for :meth:`xs`.

        Returns:
            Observables at ``self.angles``; the cross section is in mb/Sr.
        """
        Tlj, _, _ = self.tmatrix(
            U_p_coulomb=U_p_coulomb,
            U_p_central=U_p_central,
            U_p_spin_orbit=U_p_spin_orbit,
            U_n_central=U_n_central,
            U_n_spin_orbit=U_n_spin_orbit,
            U1_central=U1_central,
            U1_spin_orbit=U1_spin_orbit,
        )
        return self.observables_from_tmatrix(Tlj)

    def amplitudes_from_tmatrix(self, Tlj: ComplexArray) -> ComplexArray:
        r"""
        Spin-1/2 transition amplitude matrix from the DWBA T-matrix.

        Args:
            Tlj: Partial-wave T-matrix from :meth:`tmatrix`, with shape
                ``(lmax + 1, 2)`` indexed by ``[l, j]``.

        Returns:
            Amplitudes :math:`T_{mm'}(\theta)` with shape
            ``(2, 2, len(self.angles))``.
        """
        # geometric_factor is zero wherever the (l, j, m, m') combination is
        # not allowed, so the sum needs no further selection rules
        return np.einsum("abljt,lj->abt", self.geometric_factor, Tlj)

    def observables_from_tmatrix(self, Tlj: ComplexArray) -> QuasielasticPnXS:
        """
        Observables from the DWBA T-matrix.

        Args:
            Tlj: Partial-wave T-matrix from :meth:`tmatrix`.

        Returns:
            Observables at ``self.angles``; the cross section is in mb/Sr.
        """
        return pn_observables(
            self.amplitudes_from_tmatrix(Tlj), xs_factor=self.xs_factor
        )
