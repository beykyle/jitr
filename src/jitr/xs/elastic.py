"""Elastic-scattering observables on the lax solver engine.

Workspaces are energy-vectorized: the ``kinematics`` object passed at
construction may carry scalar or ``(N_E,)`` fields, and every observable
gains the energy axis. Axis conventions (design doc §3.3-3.5, §8 Q2):

- partial-wave arrays are trailing-E: ``Splus``/``Tplus`` are
  ``(lmax+1, N_E)``; ``Sminus``/``Tminus`` are ``(lmax, N_E)`` (the j = l−½
  branch starts at l = 1);
- observables are leading-E: ``dsdo``/``Ay``/``Q`` are ``(N_E, N_θ)``,
  ``t``/``rxn`` are ``(N_E,)``.

The energy grid, ``lmax``, channel radius, and basis size are compile-time:
changing any of them means a new workspace (seconds, mpmath-dominated for
charged channels); changing the *potential* re-executes the jitted pipeline
only. Outputs are JAX arrays so the potential → cross-section pipeline is
differentiable end-to-end (use ``method="linear_solve"`` for gradients;
the default complex-potential spectral path is not differentiable).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre, gamma, lpmv

from ..reactions import Reaction
from ..utils.kinematics import ChannelKinematics
from ._lax_engine import BlockedEngine, InteractionPair

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass
class ElasticXS:
    """Elastic observables on the workspace energy × angle grids.

    Attributes:
        dsdo: Differential cross section in mb/sr, ``(N_E, N_θ)``.
        Ay: Analyzing power, ``(N_E, N_θ)``.
        Q: Spin rotation, ``(N_E, N_θ)``.
        t: Total cross section in mb, ``(N_E,)`` (neutral channels only).
        rxn: Reaction cross section in mb, ``(N_E,)``.
    """

    dsdo: FloatArray
    Ay: FloatArray
    Q: FloatArray
    t: FloatArray
    rxn: FloatArray


@lru_cache(maxsize=1)
def _kernels():
    """Build the jitted observable kernels (lazy: keeps jax optional)."""
    import jax
    import jax.numpy as jnp

    def _weights(n_l: int):
        ls = jnp.arange(n_l, dtype=jnp.float64)
        return ls + 1.0, ls

    @jax.jit
    def integral(k, splus, sminus_padded):
        w_plus, w_minus = _weights(splus.shape[0])
        rxn = w_plus @ (1.0 - jnp.abs(splus) ** 2) + w_minus @ (
            1.0 - jnp.abs(sminus_padded) ** 2
        )
        t = w_plus @ (1.0 - jnp.real(splus)) + w_minus @ (1.0 - jnp.real(sminus_padded))
        rxn = rxn * 10.0 * jnp.pi / k**2
        t = t * 10.0 * 2.0 * jnp.pi / k**2
        return t, rxn

    @jax.jit
    def differential(k, splus, sminus_padded, p_l, p_1_l, f_c, sigma_l, eps=1e-30):
        w_plus, w_minus = _weights(splus.shape[0])
        phase = jnp.exp(2j * sigma_l) / (2j * k[:, None])  # (N_E, L+1)
        coeff_a = (
            w_plus[:, None] * (splus - 1.0) + w_minus[:, None] * (sminus_padded - 1.0)
        ).T  # (N_E, L+1)
        coeff_b = (splus - sminus_padded).T
        a = f_c + jnp.einsum("el,lt->et", phase * coeff_a, p_l)
        b = jnp.einsum("el,lt->et", phase * coeff_b, p_1_l)

        dsdo0 = jnp.abs(a) ** 2 + jnp.abs(b) ** 2
        denom = jnp.maximum(dsdo0, eps)
        ay = 2.0 * jnp.imag(jnp.conjugate(a) * b) / denom
        q = 2.0 * jnp.real(jnp.conjugate(a) * b) / denom

        t, rxn = integral(k, splus, sminus_padded)
        return dsdo0 * 10.0, ay, q, t, rxn

    return integral, differential


def _pad_sminus(splus: Any, sminus: Any) -> Any:
    """Prepend the zero-weight l = 0 row so both j branches align."""
    import jax.numpy as jnp

    return jnp.concatenate([jnp.ones_like(splus[:1]), jnp.asarray(sminus)])


def integral_elastic_xs(
    k: npt.ArrayLike, splus: Any, sminus: Any
) -> tuple[FloatArray, FloatArray]:
    """Return (σ_total, σ_reaction) in mb, each ``(N_E,)``.

    Args:
        k: CM wavenumbers, ``(N_E,)``.
        splus: j = l+½ S-matrix, ``(lmax+1, N_E)``.
        sminus: j = l−½ S-matrix, ``(lmax, N_E)`` (starting at l = 1).
    """
    import jax.numpy as jnp

    integral, _ = _kernels()
    k_arr = jnp.atleast_1d(jnp.asarray(k, dtype=jnp.float64))
    return integral(k_arr, jnp.asarray(splus), _pad_sminus(splus, sminus))


def differential_elastic_xs(
    k: npt.ArrayLike,
    splus: Any,
    sminus: Any,
    P_l_costheta: npt.ArrayLike,
    P_1_l_costheta: npt.ArrayLike,
    f_c: npt.ArrayLike,
    sigma_l: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(dσ/dΩ, A_y, Q, σ_total, σ_reaction)`` over (E, θ).

    Args:
        k: CM wavenumbers, ``(N_E,)``.
        splus: j = l+½ S-matrix, ``(lmax+1, N_E)``.
        sminus: j = l−½ S-matrix, ``(lmax, N_E)``.
        P_l_costheta: Legendre table, ``(lmax+1, N_θ)``.
        P_1_l_costheta: Associated Legendre table, ``(lmax+1, N_θ)``.
        f_c: Coulomb amplitude, ``(N_E, N_θ)`` (zeros when neutral).
        sigma_l: Coulomb phase shifts, ``(N_E, lmax+1)``.
    """
    import jax.numpy as jnp

    _, differential = _kernels()
    k_arr = jnp.atleast_1d(jnp.asarray(k, dtype=jnp.float64))
    return differential(
        k_arr,
        jnp.asarray(splus),
        _pad_sminus(splus, sminus),
        jnp.asarray(P_l_costheta),
        jnp.asarray(P_1_l_costheta),
        jnp.asarray(f_c),
        jnp.asarray(sigma_l),
    )


def suggest_lmax(
    reaction: Reaction,
    Elab_max: float,
    channel_radius_fm: float,
    margin: int = 6,
) -> int:
    """Suggest ``lmax`` for the *highest* grid energy (design doc §3.4).

    Blocks are compile-time static — there is no per-l early exit — so the
    truncation must hold at the largest wavenumber. The grazing partial wave
    is l_gr ≈ k·R; ``margin`` waves are added for the classically forbidden
    tail (more may be needed for very sharp forward-angle structure).
    """
    kinematics = reaction.kinematics(Elab_max)
    k_max = float(np.max(np.asarray(kinematics.k)))
    return int(np.ceil(k_max * channel_radius_fm)) + int(margin)


class IntegralWorkspace:
    """Energy-vectorized workspace for integral elastic observables.

    One compiled blocked lax solver per (reaction, kinematics grid, lmax,
    channel radius, nbasis). Potentials are supplied per call — arrays on
    :meth:`radial_grid` (local ``(N,)``/``(N_E, N)``, non-local ``(N, N)``/
    ``(N_E, N, N)``, optionally with a leading ``(lmax+1,)`` axis), callables,
    or pre-built term objects from :meth:`central` & friends.
    """

    def __init__(
        self,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        channel_radius_fm: float,
        lmax: int,
        nbasis: int = 40,
        *,
        V_is_complex: bool = True,
        method: str | None = None,
        wavefunctions: bool = False,
        dps: int = 40,
        dtype: Any = None,
        device: Any = None,
    ) -> None:
        if reaction.process is None or reaction.process.lower() != "el":
            raise ValueError("Reaction must be elastic!")
        self.reaction = reaction
        self.kinematics = kinematics
        self.lmax = int(lmax)
        self.nbasis = int(nbasis)
        self.channel_radius_fm = float(channel_radius_fm)
        self.engine = BlockedEngine(
            kinematics,
            channel_radius_fm,
            lmax,
            nbasis,
            (reaction.projectile.Z, reaction.target.Z),
            V_is_complex=V_is_complex,
            method=method,
            wavefunctions=wavefunctions,
            dps=dps,
            dtype=dtype,
            device=device,
        )
        self.ls = np.arange(self.lmax + 1, dtype=np.int64)[:, np.newaxis]

    @property
    def n_energies(self) -> int:
        return self.engine.grid.n_energies

    def radial_grid(self) -> FloatArray:
        """Physical quadrature grid in fm (energy-independent)."""
        return self.engine.radial_grid()

    # -- potential term builders (design doc §3.3) --------------------------

    def central(self, term: Any, **dispatch: Any) -> Any:
        """Build the central term: local/non-local, static/energy-dependent."""
        return self.engine.interaction(term, name="central_potential", **dispatch)

    def coulomb(self, term: Any, **dispatch: Any) -> Any:
        """Build the Coulomb term (same contract as :meth:`central`)."""
        return self.engine.interaction(term, name="coulomb_potential", **dispatch)

    def nonlocal_(self, term: Any, **dispatch: Any) -> Any:
        """Build a non-local kernel term K(r, r') (raw values; Gauss scaling
        is applied by the solver)."""
        return self.engine.interaction(term, name="nonlocal_potential", **dispatch)

    def spin_orbit(self, term: Any, **dispatch: Any) -> InteractionPair:
        """Build the ⟨l·σ⟩-scaled (V⁺, V⁻) spin-orbit pair from an unscaled
        radial form factor."""
        return self.engine.spin_orbit_pair(
            term, name="spin_orbit_potential", **dispatch
        )

    # -- observables ---------------------------------------------------------

    def _assemble(
        self,
        central_potential: Any,
        spin_orbit_potential: Any = None,
        coulomb_potential: Any = None,
        **dispatch: Any,
    ) -> Any:
        lax = _lax()
        if isinstance(central_potential, (InteractionPair, lax.Interaction)):
            potential = central_potential
        else:
            potential = self.central(central_potential, **dispatch)
        if spin_orbit_potential is not None:
            if isinstance(spin_orbit_potential, InteractionPair):
                potential = potential + spin_orbit_potential
            else:
                potential = potential + self.spin_orbit(spin_orbit_potential)
        if coulomb_potential is not None:
            if isinstance(coulomb_potential, lax.Interaction):
                potential = potential + coulomb_potential
            else:
                potential = potential + self.coulomb(coulomb_potential)
        return potential

    def smatrix(
        self,
        central_potential: Any,
        spin_orbit_potential: Any = None,
        coulomb_potential: Any = None,
        **dispatch: Any,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Compute the elastic S-matrix for ``j = l ± ½`` channels.

        Returns:
            ``(Splus, Sminus)`` with shapes ``(lmax+1, N_E)`` and
            ``(lmax, N_E)`` (the j = l−½ branch starts at l = 1).
        """
        potential = self._assemble(
            central_potential, spin_orbit_potential, coulomb_potential, **dispatch
        )
        splus, sminus = self.engine.smatrix(potential)
        return splus, sminus[1:]

    def xs(
        self,
        central_potential: Any,
        spin_orbit_potential: Any = None,
        coulomb_potential: Any = None,
        **dispatch: Any,
    ) -> tuple[FloatArray, FloatArray]:
        """Return total and reaction cross sections in mb, each ``(N_E,)``."""
        splus, sminus = self.smatrix(
            central_potential, spin_orbit_potential, coulomb_potential, **dispatch
        )
        return integral_elastic_xs(self.engine.grid.k, splus, sminus)

    def transmission_coefficients(
        self,
        central_potential: Any,
        spin_orbit_potential: Any = None,
        coulomb_potential: Any = None,
        **dispatch: Any,
    ) -> tuple[FloatArray, FloatArray]:
        """Return transmission coefficients for ``j = l ± ½`` channels,
        shapes ``(lmax+1, N_E)`` and ``(lmax, N_E)``."""
        splus, sminus = self.smatrix(
            central_potential, spin_orbit_potential, coulomb_potential, **dispatch
        )
        return 1.0 - np.absolute(splus) ** 2, 1.0 - np.absolute(sminus) ** 2


class DifferentialWorkspace:
    """Workspace for angular elastic observables over the energy grid."""

    @classmethod
    def build_from_system(
        cls,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        channel_radius_fm: float,
        lmax: int,
        angles: FloatArray,
        nbasis: int = 40,
        **solver_kwargs: Any,
    ) -> DifferentialWorkspace:
        """Construct a differential workspace from the raw system inputs."""
        integral_workspace = IntegralWorkspace(
            reaction,
            kinematics,
            channel_radius_fm,
            lmax,
            nbasis,
            **solver_kwargs,
        )
        return cls(integral_workspace, angles)

    def __init__(
        self, integral_workspace: IntegralWorkspace, angles: FloatArray
    ) -> None:
        """Precompute angular and Coulomb factors for the energy grid."""
        self.integral_workspace = integral_workspace
        self.reaction = integral_workspace.reaction
        self.kinematics = integral_workspace.kinematics
        self.grid = integral_workspace.engine.grid

        check_angles(angles)
        self.angles = angles
        self.ls = integral_workspace.ls
        self.P_l_costheta = eval_legendre(self.ls, np.cos(self.angles))
        self.P_1_l_costheta = lpmv(1, self.ls, np.cos(self.angles))

        self.Zz = self.reaction.projectile.Z * self.reaction.target.Z
        # (N_E, lmax+1): σ_l(E) through η(E)
        self.sigma_l = self.coulomb_phase_shift(self.ls[:, 0].astype(np.float64))
        if self.Zz > 0:
            self.rutherford: FloatArray | None = self.rutherford_xs(self.angles)
            self.f_c: ComplexArray = self.coulomb_amplitude(
                self.angles, self.sigma_l[:, 0]
            )
        else:
            self.f_c = np.zeros(
                (self.grid.n_energies, self.angles.size), dtype=np.complex128
            )
            self.rutherford = None

    def radial_grid(self) -> FloatArray:
        """Physical quadrature grid in fm (energy-independent)."""
        return self.integral_workspace.radial_grid()

    def rutherford_xs(self, angles: FloatArray) -> FloatArray:
        """Rutherford cross section in mb/sr, ``(N_E, N_θ)``."""
        check_angles(angles)
        sin2 = np.sin(angles / 2.0) ** 2
        eta = self.grid.eta[:, None]
        k = self.grid.k[:, None]
        return 10 * eta**2 / (4 * k**2 * sin2[None, :] ** 2)

    def coulomb_amplitude(
        self, angles: FloatArray, sigma_0: npt.ArrayLike
    ) -> ComplexArray:
        """Coulomb scattering amplitude, ``(N_E, N_θ)``."""
        sin2 = np.sin(angles / 2.0)[None, :]
        eta = self.grid.eta[:, None]
        k = self.grid.k[:, None]
        sigma_0_col = np.asarray(sigma_0, dtype=np.float64).reshape(-1, 1)
        return np.asarray(
            -eta
            / (2 * k * sin2**2)
            * np.exp(2j * sigma_0_col - 2j * eta * np.log(sin2)),
            dtype=np.complex128,
        )

    def coulomb_phase_shift(self, ls: FloatArray) -> FloatArray:
        """Coulomb phase shifts σ_l(E), ``(N_E, len(ls))``."""
        eta = self.grid.eta[:, None]
        return np.angle(gamma(1 + ls[None, :] + 1j * eta))

    def xs(
        self,
        central_potential: Any,
        spin_orbit_potential: Any = None,
        coulomb_potential: Any = None,
        **dispatch: Any,
    ) -> ElasticXS:
        """Return differential and integral elastic observables."""
        splus, sminus = self.integral_workspace.smatrix(
            central_potential, spin_orbit_potential, coulomb_potential, **dispatch
        )
        return ElasticXS(
            *differential_elastic_xs(
                self.grid.k,
                splus,
                sminus,
                self.P_l_costheta,
                self.P_1_l_costheta,
                self.f_c,
                self.sigma_l,
            )
        )


def check_angles(angles: FloatArray) -> None:
    """Validate a 1-D angle array on ``[0, π]``."""
    angle_array = np.asarray(angles)
    if angle_array.ndim != 1:
        raise ValueError("angles must be a 1D array")
    if angle_array[0] < 0 or angle_array[-1] > np.pi:
        raise ValueError("angles must be a grid in radians on [0, pi]")


def _lax():
    from ._lax_engine import _import_lax

    return _import_lax()
