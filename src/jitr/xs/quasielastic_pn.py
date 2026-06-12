"""DWBA workspaces for quasi-elastic ``(p,n)`` scattering on the lax engine.

The workspace owns two blocked solvers (proton entrance, neutron exit) on
the same mesh and energy grid (design doc §3.6). Distorted waves come from
``wavefunction_grid`` (or ``wavefunction_direct_grid`` under
``method="linear_solve"``); the isovector transition element is the
non-conjugated bilinear ``matrix_element(χp, χn, U₁, conjugate=False)``.

T-matrix normalization: lax's interior solution is driven by the boundary
*value* ``H⁻(a)`` while the legacy engine was driven by the matched exterior
*derivative*. The per-(l, j, E) conversion is the closed form
``(i/2)(H⁻′ − S·H⁺′)/H⁻`` from the solver's boundary cache (lax DESIGN.md
Appendix C.12, machine-verified in lax ``tests/acceptance``), giving

    T_lj(E) = conv_p · conv_n · matrix_element(χp, χn, U₁) / R²

identical to the legacy node sum. The conversion is per-(l, j, E), so the
relative phases entering the coherent ``xs()`` sum are preserved exactly and
the angular reduction is unchanged from the validated legacy form.

Shapes: ``tmatrix()`` returns ``(Tpn, Sn, Sp)``, each ``(lmax+1, 2, N_E)``
with the ``[l=0, j=l−½]`` entries zero (no such channel); ``xs()`` returns
``(N_E, N_θ)`` in mb/sr.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, sph_harm_y
from sympy.physics.wigner import clebsch_gordan

from ..reactions import Reaction
from ..utils import constants
from ..utils.kinematics import ChannelKinematics
from ._lax_engine import BlockedEngine, ldots
from .elastic import check_angles

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


class Workspace:
    r"""Workspace for (p,n) quasi-elastic DWBA observables.

    Energy-vectorized: both kinematics objects may carry scalar or ``(N_E,)``
    fields (the grids must be index-aligned: entry ``e`` of the exit grid is
    the exit channel for entry ``e`` of the entrance grid).
    """

    def __init__(
        self,
        reaction: Reaction,
        kinematics_entrance: ChannelKinematics,
        kinematics_exit: ChannelKinematics,
        angles: FloatArray,
        lmax: int,
        channel_radius_fm: float,
        nbasis: int = 40,
        *,
        V_is_complex: bool = True,
        method: str | None = None,
        dps: int = 40,
        dtype: Any = None,
        device: Any = None,
    ) -> None:
        if reaction.residual is None or reaction.product is None:
            raise ValueError(
                "Reaction must define both residual and product for (p,n) scattering"
            )
        self.reaction = reaction
        self.lmax = int(lmax)
        self.nbasis = int(nbasis)
        self.channel_radius_fm = float(channel_radius_fm)
        self.kinematics_entrance = kinematics_entrance
        self.kinematics_exit = kinematics_exit
        self.method = method

        check_angles(angles)
        self.angles = angles

        solvers = (
            ("rmatrix_direct", "wavefunction")
            if method == "linear_solve"
            else ("spectrum", "smatrix", "wavefunction")
        )
        engine_kwargs = dict(
            V_is_complex=V_is_complex,
            method=method,
            solvers=solvers,
            dps=dps,
            dtype=dtype,
            device=device,
        )
        self.engine_p = BlockedEngine(
            kinematics_entrance,
            channel_radius_fm,
            lmax,
            nbasis,
            (reaction.projectile.Z, reaction.target.Z),
            **engine_kwargs,
        )
        self.engine_n = BlockedEngine(
            kinematics_exit,
            channel_radius_fm,
            lmax,
            nbasis,
            (reaction.product.Z, reaction.residual.Z),
            **engine_kwargs,
        )
        if self.engine_p.grid.n_energies != self.engine_n.grid.n_energies:
            raise ValueError(
                "entrance and exit kinematics must have the same number of energies"
            )

        # isovector factor of the Lane-consistent transition operator
        A = reaction.target.A
        Z = reaction.target.Z
        N = A - Z
        self.isovector_factor = np.sqrt(np.fabs(N - Z)) / (N - Z - 1)

        self._ldots_plus, self._ldots_minus = ldots(self.lmax)
        self._precompute_geometry()

    def _precompute_geometry(self) -> None:
        """Purely geometric/Coulomb factors of the angular reduction."""
        k_p = self.engine_p.grid.k
        k_n = self.engine_n.grid.k
        mu_p = self.engine_p.grid.mu
        mu_n = self.engine_n.grid.mu
        eta_p = self.engine_p.grid.eta
        n_e = self.engine_p.grid.n_energies
        ls = np.arange(self.lmax + 1)

        self.xs_factor = (
            (k_n / k_p) * mu_p * mu_n / (4 * np.pi**2 * constants.HBARC**4 * 2.0)
        )
        # σ_c(E, l) with the entrance-channel η(E)
        self.sigma_c = np.angle(gamma(1 + ls[None, :] + 1j * eta_p[:, None]))

        self.geometric_factor = np.zeros(
            (2, 2, self.lmax + 1, 2, n_e, self.angles.shape[0]),
            dtype=np.complex128,
        )
        kinematic = (4 * np.pi) ** 1.5 / (k_p * k_n)  # (N_E,)
        for im, m in enumerate([-0.5, 0.5]):
            for imp, mp in enumerate([-0.5, 0.5]):
                for ell in range(self.lmax + 1):
                    j_values = [ell + 1 / 2, ell - 1 / 2] if ell > 0 else [ell + 1 / 2]
                    for ijp, jp in enumerate(j_values):
                        if abs(m - mp) <= ell and jp >= 0:
                            ylm = sph_harm_y(ell, int(m - mp), self.angles, 0)
                            cg0 = clebsch_gordan(ell, 1 / 2, jp, m - mp, m, mp)
                            cg1 = clebsch_gordan(ell, 1 / 2, jp, 0, m, m)
                            self.geometric_factor[im, imp, ell, ijp] = (
                                kinematic[:, None]
                                * np.exp(1j * self.sigma_c[:, ell])[:, None]
                                * float(cg1)
                                * float(cg0)
                                * np.sqrt(2 * ell + 1)
                                * (-1) ** (2 * jp + 1)
                                * ylm[None, :]
                            )

    def radial_grid(self) -> FloatArray:
        """Physical quadrature grid in fm (energy-independent, shared)."""
        return self.engine_p.radial_grid()

    # -- internals -----------------------------------------------------------

    def _raw_term(
        self,
        engine: BlockedEngine,
        term: Any,
        name: str,
        *,
        energy_dependent: bool | None,
        optional: bool = False,
    ):
        """Normalize one *unscaled* potential term to (values, interp).

        The interior rescale is deliberately NOT applied: these raw values
        feed the U₁ transition operator, which enters the bilinear matrix
        element (physical, not interior-scaled).
        """
        if term is None:
            if not optional:
                raise TypeError(f"{name} is required")
            return None
        return engine._term_arrays(
            term, energy_dependent=energy_dependent, l_dependent=False, name=name
        )

    def _u1_interactions(self, terms_p, terms_n):
        """Build the per-j isovector operator U₁ = −(U_n − U_p)·factor.

        ``terms_*`` are dicts with optional ``central``/``spin_orbit``
        (values, interp) pairs. Returns ``(U1_plus, U1_minus)`` Interactions
        on the proton solver (both solvers share the mesh).
        """
        import jax.numpy as jnp

        solver = self.engine_p.solver
        factor = self.isovector_factor
        signed_terms = []
        for sign, terms in ((-1.0, terms_n), (+1.0, terms_p)):
            if terms.get("central") is not None:
                signed_terms.append((*terms["central"], sign, False))
            if terms.get("spin_orbit") is not None:
                signed_terms.append((*terms["spin_orbit"], sign, True))

        out = []
        for couplings in (self._ldots_plus, self._ldots_minus):
            local_terms: list[Any] = []
            nonlocal_terms: list[Any] = []
            e_dep = {"local": False, "nonlocal": False}
            for values, interp, sign, l_scaled in signed_terms:
                kind = "nonlocal" if interp.is_nonlocal else "local"
                scaled = sign * factor * values
                if l_scaled:
                    expand = (slice(None),) + (None,) * np.ndim(values)
                    scaled = couplings[expand] * scaled[None]
                target = nonlocal_terms if interp.is_nonlocal else local_terms
                target.append((scaled, interp.energy_dependent, l_scaled))
                e_dep[kind] = e_dep[kind] or interp.energy_dependent

            interactions = []
            n_e = self.engine_p.grid.n_energies
            for kind, terms_list in (
                ("local", local_terms),
                ("nonlocal", nonlocal_terms),
            ):
                mesh_axes = 2 if kind == "nonlocal" else 1
                for values, term_e_dep, l_scaled in terms_list:
                    promote_e = e_dep[kind] and not term_e_dep
                    if promote_e:
                        shape = tuple(values.shape)
                        cut = len(shape) - mesh_axes
                        values = jnp.broadcast_to(
                            values.reshape(shape[:cut] + (1,) + shape[cut:]),
                            shape[:cut] + (n_e,) + shape[cut:],
                        )
                    kwargs = {
                        "energy_dependent": e_dep[kind],
                        "block_dependent": l_scaled,
                    }
                    term_kw = (
                        {"nonlocal_": [jnp.asarray(values)]}
                        if kind == "nonlocal"
                        else {"local": [jnp.asarray(values)]}
                    )
                    interactions.append(
                        solver.interaction_from_array(**term_kw, **kwargs)
                    )
            total = interactions[0]
            for interaction in interactions[1:]:
                total = total + interaction
            out.append(total)
        return out[0], out[1]

    def _distorted_waves(self, engine: BlockedEngine, interaction):
        """Return (χ (N_b, N_E, M), S (N_b, N_E), conv (N_b, N_E))."""
        solver = engine.solver
        if self.method == "linear_solve":
            s = solver.smatrix_direct(interaction)[:, :, 0, 0]
            chi = solver.wavefunction_direct_grid(interaction)
        else:
            spectrum = solver.spectrum(interaction)
            use_grid = (
                interaction.energy_dependent or not engine.grid.uniform_mass_factor
            )
            if use_grid:
                s = solver.smatrix_grid(spectrum)[:, :, 0, 0]
            else:
                s = solver.smatrix(spectrum)[:, :, 0, 0]
            chi = solver.wavefunction_grid(spectrum)
        boundary = solver.boundary
        h_minus = np.asarray(boundary.H_minus)[:, :, 0]
        h_minus_p = np.asarray(boundary.H_minus_p)[:, :, 0]
        h_plus_p = np.asarray(boundary.H_plus_p)[:, :, 0]
        conv = 0.5j * (h_minus_p - np.asarray(s) * h_plus_p) / h_minus
        return chi, np.asarray(s), conv

    def tmatrix(
        self,
        U_p_coulomb: Any,
        U_p_central: Any,
        U_p_spin_orbit: Any = None,
        U_n_central: Any = None,
        U_n_spin_orbit: Any = None,
        *,
        energy_dependent: bool | None = None,
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        """DWBA transition matrix for (p,n) quasi-elastic scattering.

        Args:
            U_p_coulomb: proton Coulomb term (arrays/callables per the
                §3.3 potential contract).
            U_p_central: proton central term.
            U_p_spin_orbit: proton spin-orbit form factor (unscaled).
            U_n_central: neutron central term (required).
            U_n_spin_orbit: neutron spin-orbit form factor (unscaled).
            energy_dependent: explicit dispatch flag forwarded to ambiguous
                array shapes / two-argument callables.

        Returns:
            ``(Tpn, Sn, Sp)``, each ``(lmax+1, 2, N_E)``; index 0/1 of the
            second axis is j = l ± ½ and the ``[0, 1]`` entries are zero.
        """
        if U_n_central is None:
            raise TypeError("U_n_central is required")

        terms_p = {
            "central": self._raw_term(
                self.engine_p,
                U_p_central,
                "U_p_central",
                energy_dependent=energy_dependent,
            ),
            "spin_orbit": self._raw_term(
                self.engine_p,
                U_p_spin_orbit,
                "U_p_spin_orbit",
                energy_dependent=energy_dependent,
                optional=True,
            ),
        }
        terms_n = {
            "central": self._raw_term(
                self.engine_n,
                U_n_central,
                "U_n_central",
                energy_dependent=energy_dependent,
            ),
            "spin_orbit": self._raw_term(
                self.engine_n,
                U_n_spin_orbit,
                "U_n_spin_orbit",
                energy_dependent=energy_dependent,
                optional=True,
            ),
        }

        # distorting potentials (interior-rescaled by the engines)
        v_p = self.engine_p.interaction(
            U_p_central, energy_dependent=energy_dependent, name="U_p_central"
        ) + self.engine_p.interaction(
            U_p_coulomb, energy_dependent=energy_dependent, name="U_p_coulomb"
        )
        if U_p_spin_orbit is not None:
            v_p = v_p + self.engine_p.spin_orbit_pair(
                U_p_spin_orbit, energy_dependent=energy_dependent, name="U_p_spin_orbit"
            )
        v_n = self.engine_n.interaction(
            U_n_central, energy_dependent=energy_dependent, name="U_n_central"
        )
        if U_n_spin_orbit is not None:
            v_n = v_n + self.engine_n.spin_orbit_pair(
                U_n_spin_orbit, energy_dependent=energy_dependent, name="U_n_spin_orbit"
            )

        u1_plus, u1_minus = self._u1_interactions(terms_p, terms_n)

        n_e = self.engine_p.grid.n_energies
        Tpn = np.zeros((self.lmax + 1, 2, n_e), dtype=np.complex128)
        Sn = np.zeros_like(Tpn)
        Sp = np.zeros_like(Tpn)

        radius_sq = self.channel_radius_fm**2
        for ij, u1 in ((0, u1_plus), (1, u1_minus)):
            v_p_j = _pair_member(v_p, ij)
            v_n_j = _pair_member(v_n, ij)
            chi_p, s_p, conv_p = self._distorted_waves(self.engine_p, v_p_j)
            chi_n, s_n, conv_n = self._distorted_waves(self.engine_n, v_n_j)
            element = np.asarray(
                self.engine_p.solver.matrix_element(chi_p, chi_n, u1, conjugate=False)
            )  # (N_b, N_E)
            Tpn[:, ij, :] = conv_p * conv_n * element / radius_sq
            Sn[:, ij, :] = s_n
            Sp[:, ij, :] = s_p

        # there is no j = l − ½ channel at l = 0
        Tpn[0, 1] = Sn[0, 1] = Sp[0, 1] = 0.0
        return Tpn, Sn, Sp

    def xs(
        self,
        U_p_coulomb: Any,
        U_p_central: Any,
        U_p_spin_orbit: Any = None,
        U_n_central: Any = None,
        U_n_spin_orbit: Any = None,
        *,
        energy_dependent: bool | None = None,
    ) -> FloatArray:
        """Differential (p,n) cross section in mb/sr, shape ``(N_E, N_θ)``."""
        Tlj, Sn, Sp = self.tmatrix(
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
            energy_dependent=energy_dependent,
        )
        n_e = self.engine_p.grid.n_energies
        Tmmp = np.zeros((2, 2, n_e, self.angles.shape[0]), dtype=np.complex128)
        # NOTE: the l-sum runs to lmax-1, matching the legacy engine (and the
        # goldens); the highest compiled wave acts as a convergence buffer.
        for im in range(2):
            for imp in range(2):
                for ell in range(self.lmax):
                    for ijp, jp in enumerate([ell + 0.5, ell - 0.5]):
                        m = -0.5 + im
                        mp = -0.5 + imp
                        if abs(m - mp) <= ell and jp >= 0:
                            Tmmp[im, imp] += (
                                self.geometric_factor[im, imp, ell, ijp]
                                * Tlj[ell, ijp][:, None]
                            )
        return (
            self.xs_factor[:, None] * 10.0 * np.sum(np.absolute(Tmmp) ** 2, axis=(0, 1))
        )


def _pair_member(potential: Any, ij: int) -> Any:
    """Select the j = l ± ½ member from an Interaction or InteractionPair."""
    from ._lax_engine import InteractionPair

    if isinstance(potential, InteractionPair):
        return potential.plus if ij == 0 else potential.minus
    return potential
