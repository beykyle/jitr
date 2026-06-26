"""Distorted-wave reconstruction on the lax solver engine.

Replaces the legacy ``Wavefunctions`` class (which reconstructed the
rmatrix engine's internal solution in s = k·r units). The new
:class:`DistortedWaves` works in physical fm on an elastic
:class:`~jitr.xs.elastic.IntegralWorkspace` built with
``wavefunctions=True``, and returns the *physically normalized* waves of
the legacy convention,

    u(R) = (i/2)[H⁻(kR) − S·H⁺(kR)],

obtained from lax's boundary-value-driven solution via the closed-form
source-conversion factor (lax DESIGN.md Appendix C.12).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from lax.transforms import compute_B_grid

from ..utils.free_solutions import CoulombAsymptotics, H_minus, H_plus

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


class DistortedWaves:
    """Interior/exterior distorted waves for an elastic workspace.

    Args:
        workspace: an ``IntegralWorkspace`` constructed with
            ``wavefunctions=True``.
        central_potential: the central term, exactly as for
            ``workspace.smatrix``.
        spin_orbit_potential: optional unscaled spin-orbit form factor.
        coulomb_potential: optional Coulomb term.
        dispatch: optional ``energy_dependent=``/``l_dependent=`` flags.

    Attributes:
        splus: j = l+½ S-matrix, ``(lmax+1, N_E)``.
        sminus: j = l−½ S-matrix, ``(lmax, N_E)``.
    """

    def __init__(
        self,
        workspace: Any,
        central_potential: Any,
        spin_orbit_potential: Any = None,
        coulomb_potential: Any = None,
        **dispatch: Any,
    ) -> None:
        from ..xs._lax_engine import InteractionPair

        self.workspace = workspace
        engine = workspace.engine
        self.engine = engine
        self.grid = engine.grid

        potential = workspace._assemble(
            central_potential, spin_orbit_potential, coulomb_potential, **dispatch
        )
        if not isinstance(potential, InteractionPair):
            potential = InteractionPair(potential, potential)
        chi_p, s_p, _ = engine.distorted_waves(potential.plus)
        chi_m, s_m, _ = engine.distorted_waves(potential.minus)
        self.splus = np.asarray(s_p)
        self.sminus = np.asarray(s_m)[1:]
        self._smatrix = {"plus": np.asarray(s_p), "minus": np.asarray(s_m)}

        # physical normalization (module docstring): scale the raw mesh
        # solution so its boundary value equals (i/2)[H⁻(kR) − S·H⁺(kR)],
        # using the solver's own boundary cache and basis values
        boundary = engine.solver.boundary
        h_minus = np.asarray(boundary.H_minus)[:, :, 0]
        h_plus = np.asarray(boundary.H_plus)[:, :, 0]
        basis_boundary = np.asarray(
            compute_B_grid(engine.solver.mesh, np.array([engine.channel_radius_fm]))
        )[0]
        self._coeffs = {}
        for label, chi, s_matrix in (
            ("plus", chi_p, self._smatrix["plus"]),
            ("minus", chi_m, self._smatrix["minus"]),
        ):
            chi_arr = np.asarray(chi)
            raw_boundary = chi_arr @ basis_boundary  # (N_b, N_E)
            target = 0.5j * (h_minus - s_matrix * h_plus)
            self._coeffs[label] = chi_arr * (target / raw_boundary)[:, :, None]

    def interior(self, r: npt.ArrayLike, j: str = "plus") -> ComplexArray:
        """Evaluate the interior waves at radii ``r`` (fm, within [0, R]).

        Returns ``(lmax+1, N_E, len(r))`` for ``j="plus"`` and
        ``(lmax, N_E, len(r))`` for ``j="minus"`` (the j = l−½ branch
        starts at l = 1).
        """
        r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any(r_arr < 0) or np.any(r_arr > self.engine.channel_radius_fm):
            raise ValueError("interior radii must lie within [0, channel radius]")
        basis = np.asarray(compute_B_grid(self.engine.solver.mesh, r_arr))
        values = np.einsum("len,rn->ler", self._coeffs[j], basis)
        return values if j == "plus" else values[1:]

    def exterior(self, r: npt.ArrayLike, j: str = "plus") -> ComplexArray:
        """Evaluate the asymptotic waves ``(i/2)[H⁻ − S·H⁺]`` at radii ``r``.

        Valid for ``r >= channel radius``; same shapes as :meth:`interior`.
        """
        r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
        s_matrix = self._smatrix[j]
        n_b, n_e = s_matrix.shape
        values = np.zeros((n_b, n_e, r_arr.size), dtype=np.complex128)
        for ell in range(n_b):
            for ie in range(n_e):
                k = self.grid.k[ie]
                eta = self.grid.eta[ie]
                rho = k * r_arr
                h_minus = np.array(
                    [H_minus(x, ell, eta, asym=CoulombAsymptotics) for x in rho]
                )
                h_plus = np.array(
                    [H_plus(x, ell, eta, asym=CoulombAsymptotics) for x in rho]
                )
                values[ell, ie] = 0.5j * (h_minus - s_matrix[ell, ie] * h_plus)
        return values if j == "plus" else values[1:]
