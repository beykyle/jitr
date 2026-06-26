"""Shared lax-backed solver engine for the ``jitr.xs`` workspaces.

One :class:`BlockedEngine` wraps one compiled ``lax.Solver`` with a
compile-time energy grid and one single-channel symmetry block per partial
wave ``l = 0 .. lmax`` (design doc §3.1). The elastic workspaces own one
engine; the quasi-elastic (p,n) workspace owns two (proton and neutron).

Kinematics mapping (pinned by ``tests/equivalence/``, supersedes design doc
§3.2 items 2-3): the legacy engine solves ``[T_s + V/Ecm]ψ = ψ`` in
s = k·r units with boundary ``H±(k·R, η)``; lax derives ``k = √(E/mf)`` and
``η = z₁z₂e²/(2·mf·k)``. The faithful mapping for an arbitrary
``ChannelKinematics`` is therefore::

    mf_e    = ħ²c²/(2 μ_e)
    E_lax,e = mf_e · k_e²            (= Ecm only for classical kinematics)
    V_lax,e = V · E_lax,e / Ecm_e    (interior rescale; ≡ 1 classically)

The interior rescale is applied by the engine's term builders, never by the
caller. Potentials supplied as already-built ``lax.Interaction`` objects are
accepted only when the rescale is identically 1.

Spin-orbit convention: user arrays are multiplied by ⟨l·σ⟩ = {l, −(l+1)}
per j = l ± ½, matching the legacy ``spin_half_orbit_coupling``.

Axis conventions: S-matrices are returned as ``(lmax+1, N_E)`` per j; the
ignored l = 0 entry of the j = l − ½ call is *not* stripped here (the public
workspaces slice it).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lax
import numpy as np
import numpy.typing as npt

from ..utils.constants import HBARC
from ..utils.kinematics import ChannelKinematics

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


def ldots(lmax: int) -> tuple[FloatArray, FloatArray]:
    """Return ⟨l·σ⟩ for j = l + ½ and j = l − ½ per partial wave.

    The l = 0 entry of the minus branch is physically meaningless (there is
    no j = −½ channel); it carries zero weight in every observable kernel.
    """
    ls = np.arange(lmax + 1, dtype=np.float64)
    return ls, -(ls + 1.0)


@dataclass(frozen=True)
class KinematicsGrid:
    """Normalized per-energy kinematic quantities for one channel.

    All arrays are ``(N_E,)``. ``energies``/``mass_factors`` are the lax
    compile inputs; ``interior_scale`` is the per-energy potential rescale
    (module docstring). ``Ecm`` is the physical CM energy grid that
    energy-dependent potential callables are evaluated on.
    """

    Ecm: FloatArray
    k: FloatArray
    mu: FloatArray
    eta: FloatArray
    energies: FloatArray
    mass_factors: FloatArray
    interior_scale: FloatArray
    uniform_mass_factor: bool

    @property
    def n_energies(self) -> int:
        return self.Ecm.size


def normalize_kinematics(kinematics: ChannelKinematics) -> KinematicsGrid:
    """Build the lax-facing kinematics grid from a user kinematics object."""
    Ecm = np.atleast_1d(np.asarray(kinematics.Ecm, dtype=np.float64))
    k = np.broadcast_to(np.asarray(kinematics.k, dtype=np.float64), Ecm.shape).copy()
    mu = np.broadcast_to(np.asarray(kinematics.mu, dtype=np.float64), Ecm.shape).copy()
    eta = np.broadcast_to(
        np.asarray(kinematics.eta, dtype=np.float64), Ecm.shape
    ).copy()
    if Ecm.ndim != 1:
        raise ValueError(
            f"kinematics fields must be scalars or 1-D arrays, got shape {Ecm.shape}"
        )
    mass_factors = HBARC**2 / (2.0 * mu)
    energies = mass_factors * k**2
    interior_scale = energies / Ecm
    if np.allclose(interior_scale, 1.0, rtol=1e-12, atol=0.0):
        # classical kinematics: E_lax == Ecm up to the mf·k² round trip
        interior_scale = np.ones_like(interior_scale)
    uniform = bool(np.all(mass_factors == mass_factors[0]))
    return KinematicsGrid(
        Ecm=Ecm,
        k=k,
        mu=mu,
        eta=eta,
        energies=energies,
        mass_factors=mass_factors,
        interior_scale=interior_scale,
        uniform_mass_factor=uniform,
    )


@dataclass(frozen=True)
class InteractionPair:
    """A (V⁺, V⁻) pair of lax ``Interaction`` s for j = l ± ½.

    Supports ``+`` with another pair or with a single ``Interaction``
    (added to both members), so spin-orbit and central terms compose::

        V = ws.central(U) + ws.spin_orbit(U_so) + ws.coulomb(U_c)
    """

    plus: Any
    minus: Any

    def __add__(self, other: Any) -> InteractionPair:
        if isinstance(other, InteractionPair):
            return InteractionPair(self.plus + other.plus, self.minus + other.minus)
        return InteractionPair(self.plus + other, self.minus + other)

    __radd__ = __add__


# ---------------------------------------------------------------------------
# array-shape dispatch (design doc §3.3 + §8 Q1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Interpretation:
    l_dependent: bool
    energy_dependent: bool
    is_nonlocal: bool

    def expected_shape(self, n: int, n_e: int, n_b: int) -> tuple[int, ...]:
        shape: tuple[int, ...] = ()
        if self.l_dependent:
            shape += (n_b,)
        if self.energy_dependent:
            shape += (n_e,)
        shape += (n, n) if self.is_nonlocal else (n,)
        return shape

    def describe(self, n: int, n_e: int, n_b: int) -> str:
        kind = "nonlocal" if self.is_nonlocal else "local"
        flags = []
        if self.l_dependent:
            flags.append("l-dependent")
        if self.energy_dependent:
            flags.append("energy-dependent")
        label = " ".join([*flags, kind])
        return f"{label} {self.expected_shape(n, n_e, n_b)}"


def _infer_interpretation(
    shape: tuple[int, ...],
    n: int,
    n_e: int,
    n_b: int,
    energy_dependent: bool | None,
    l_dependent: bool | None,
    name: str,
) -> _Interpretation:
    candidates = [
        interp
        for interp in (
            _Interpretation(ld, ed, nl)
            for ld in (False, True)
            for ed in (False, True)
            for nl in (False, True)
        )
        if (energy_dependent is None or interp.energy_dependent == energy_dependent)
        and (l_dependent is None or interp.l_dependent == l_dependent)
        and interp.expected_shape(n, n_e, n_b) == shape
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"{name}: array shape {shape} matches no potential layout for "
            f"N={n}, N_E={n_e}, N_b={n_b} "
            f"(given energy_dependent={energy_dependent}, l_dependent={l_dependent}); "
            "see the jitr.xs potential contract"
        )
    options = "; ".join(c.describe(n, n_e, n_b) for c in candidates)
    raise ValueError(
        f"{name}: array shape {shape} is ambiguous ({options}); pass explicit "
        "energy_dependent=/l_dependent= keywords to disambiguate"
    )


def _call_arity(fn: Callable[..., Any]) -> int:
    import inspect

    parameters = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    return len(parameters)


class BlockedEngine:
    """One compiled blocked lax solver plus potential-dispatch helpers."""

    def __init__(
        self,
        kinematics: ChannelKinematics,
        channel_radius_fm: float,
        lmax: int,
        nbasis: int,
        z1z2: tuple[int, int],
        *,
        V_is_complex: bool = True,
        method: str | None = None,
        solvers: tuple[str, ...] | None = None,
        wavefunctions: bool = False,
        energy_dependent: bool = False,
        dps: int = 40,
        dtype: Any = None,
        device: Any = None,
    ) -> None:
        self.grid = normalize_kinematics(kinematics)
        self.lmax = int(lmax)
        self.nbasis = int(nbasis)
        self.channel_radius_fm = float(channel_radius_fm)
        self.method = method
        self._ldots_plus, self._ldots_minus = ldots(self.lmax)
        if solvers is None:
            # the spectral S-matrix path is eigh/eig-only; linear_solve uses
            # the direct observables (lax compile contract)
            if method == "linear_solve":
                solvers = ("rmatrix_direct",)
            else:
                solvers = ("spectrum", "smatrix")
            if wavefunctions:
                solvers = (*solvers, "wavefunction")

        mass_factors = self.grid.mass_factors
        compile_kwargs: dict[str, Any] = {}
        if dtype is not None:
            compile_kwargs["dtype"] = dtype
        if device is not None:
            compile_kwargs["device"] = device
        self.solver = lax.compile(
            mesh=lax.MeshSpec(
                "legendre", "x", n=self.nbasis, scale=self.channel_radius_fm
            ),
            blocks=[
                [
                    lax.ChannelSpec(
                        l=ell, threshold=0.0, mass_factor=float(mass_factors[0])
                    )
                ]
                for ell in range(self.lmax + 1)
            ],
            solvers=solvers,
            energies=jnp.asarray(self.grid.energies),
            energy_dependent=energy_dependent,
            mass_factor_grid=(
                None if self.grid.uniform_mass_factor else jnp.asarray(mass_factors)
            ),
            z1z2=z1z2 if z1z2[0] * z1z2[1] != 0 else None,
            V_is_complex=V_is_complex,
            method=method,
            dps=dps,
            **compile_kwargs,
        )
        # Cache of jitted ``interaction_from_array`` builders, keyed by the
        # structural signature ``(is_nonlocal, energy_dependent,
        # block_dependent)``; see :meth:`_array_interaction`.
        self._array_builders: dict[tuple[bool, bool, bool], Callable[[Any], Any]] = {}

    # -- grids -------------------------------------------------------------

    def radial_grid(self) -> FloatArray:
        """Physical quadrature grid in fm (energy-independent)."""
        return np.asarray(self.solver.mesh.radii)

    # -- potential terms ----------------------------------------------------

    def _evaluate_callable(
        self,
        fn: Callable[..., Any],
        *,
        energy_dependent: bool | None,
        name: str,
    ) -> tuple[np.ndarray, bool, bool]:
        """Evaluate a potential callable on the mesh.

        Returns ``(values, energy_dependent, is_nonlocal)``. Energy-dependent
        callables are evaluated on the *physical* ``Ecm`` grid.
        """
        r = self.radial_grid()
        arity = _call_arity(fn)
        if arity == 1:
            return fn(r), False, False
        if arity == 2:
            if energy_dependent is None:
                raise ValueError(
                    f"{name}: a two-argument callable is ambiguous — f(r, E) or "
                    "f(r, r') — pass energy_dependent=True for f(r, E) or "
                    "energy_dependent=False for a non-local f(r, r')"
                )
            if energy_dependent:
                values = jnp.stack([fn(r, e) for e in self.grid.Ecm])
                return values, True, False
            ri, rj = np.meshgrid(r, r, indexing="ij")
            return fn(ri, rj), False, True
        if arity == 3:
            ri, rj = np.meshgrid(r, r, indexing="ij")
            values = jnp.stack([fn(ri, rj, e) for e in self.grid.Ecm])
            return values, True, True
        raise ValueError(
            f"{name}: potential callables must take (r), (r, E), (r, r'), or "
            f"(r, r', E); got a callable of arity {arity}"
        )

    def _term_arrays(
        self,
        term: Any,
        *,
        energy_dependent: bool | None,
        l_dependent: bool | None,
        name: str,
    ) -> tuple[np.ndarray, _Interpretation]:
        """Normalize one potential term to (values, interpretation)."""
        n_b = self.lmax + 1
        if callable(term):
            values, e_dep, nonloc = self._evaluate_callable(
                term, energy_dependent=energy_dependent, name=name
            )
            if l_dependent:
                raise ValueError(
                    f"{name}: l_dependent=True with a single callable; pass a "
                    f"sequence of {n_b} callables instead"
                )
            return values, _Interpretation(False, e_dep, nonloc)
        if (
            isinstance(term, Sequence)
            and not isinstance(term, (str, bytes))
            and len(term) > 0
            and callable(term[0])
        ):
            if l_dependent is False:
                raise ValueError(
                    f"{name}: a sequence of callables is l-dependent by "
                    "construction; l_dependent=False contradicts it"
                )
            if len(term) != n_b:
                raise ValueError(
                    f"{name}: expected one callable per partial wave "
                    f"(lmax+1 = {n_b}), got {len(term)}"
                )
            evaluated = [
                self._evaluate_callable(
                    fn, energy_dependent=energy_dependent, name=f"{name}[{i}]"
                )
                for i, fn in enumerate(term)
            ]
            flags = {(e, nl) for _, e, nl in evaluated}
            if len(flags) != 1:
                raise ValueError(f"{name}: per-l callables must share one signature")
            e_dep, nonloc = next(iter(flags))
            values = jnp.stack([v for v, _, _ in evaluated])
            return values, _Interpretation(True, e_dep, nonloc)

        # keep JAX tracers intact (differentiable pipelines); coerce the rest
        values = term if hasattr(term, "shape") else np.asarray(term)
        interp = _infer_interpretation(
            values.shape,
            self.nbasis,
            self.grid.n_energies,
            n_b,
            energy_dependent,
            l_dependent,
            name,
        )
        return values, interp

    def _apply_interior_scale(
        self, values: np.ndarray, interp: _Interpretation
    ) -> tuple[np.ndarray, _Interpretation]:
        """Apply the per-energy V·(E_lax/Ecm) rescale (module docstring)."""
        scale = self.grid.interior_scale
        if np.all(scale == 1.0):
            return values, interp
        if np.all(scale == scale[0]):
            return values * scale[0], interp
        # non-uniform: promote to energy-dependent
        mesh_axes = 2 if interp.is_nonlocal else 1
        scale_shape = (-1,) + (1,) * mesh_axes
        if not interp.energy_dependent:
            # tracer-safe axis insertion before the mesh axes
            shape = tuple(values.shape)
            values = values.reshape(
                shape[: len(shape) - mesh_axes] + (1,) + shape[len(shape) - mesh_axes :]
            )
            interp = _Interpretation(interp.l_dependent, True, interp.is_nonlocal)
        return values * scale.reshape(scale_shape), interp

    def _array_interaction(
        self,
        values: Any,
        *,
        energy_dependent: bool,
        block_dependent: bool,
        is_nonlocal: bool,
    ) -> Any:
        """Build a lax ``Interaction`` from an array via a cached jitted builder.

        The eager ``solver.interaction_from_array`` path runs several
        un-jitted JAX ops (``eye``/einsum/``zeros``) plus concrete symmetry &
        coupling checks — each a device→host sync — costing milliseconds per
        term. Tracing the build under ``jax.jit`` fuses those ops and skips the
        concrete-only checks (the assembled block is a tracer), so repeated
        ``xs``/``smatrix`` calls with new potential *values* of the same shape
        reuse one compiled builder instead of rebuilding eagerly. One builder
        is cached per structural signature; the static shape/leading-axis
        contract is still enforced (those checks do not depend on values).
        """
        key = (is_nonlocal, energy_dependent, block_dependent)
        builder = self._array_builders.get(key)
        if builder is None:
            builder = self._make_array_builder(
                is_nonlocal=is_nonlocal,
                energy_dependent=energy_dependent,
                block_dependent=block_dependent,
            )
            self._array_builders[key] = builder
        return builder(jnp.asarray(values))

    def _make_array_builder(
        self, *, is_nonlocal: bool, energy_dependent: bool, block_dependent: bool
    ) -> Callable[[Any], Any]:
        """Compile the jitted ``interaction_from_array`` builder for one signature."""
        solver = self.solver
        if is_nonlocal:

            @jax.jit
            def builder(array: Any) -> Any:
                return solver.interaction_from_array(
                    nonlocal_=[array],
                    energy_dependent=energy_dependent,
                    block_dependent=block_dependent,
                )

        else:

            @jax.jit
            def builder(array: Any) -> Any:
                return solver.interaction_from_array(
                    local=[array],
                    energy_dependent=energy_dependent,
                    block_dependent=block_dependent,
                )

        return builder

    def interaction(
        self,
        term: Any,
        *,
        energy_dependent: bool | None = None,
        l_dependent: bool | None = None,
        name: str = "potential",
    ) -> Any:
        """Build a lax ``Interaction`` from one potential term (§3.3 table).

        ``term`` may be an array (shapes per the potential contract), a
        callable ``f(r)``/``f(r, E)``/``f(r, r')``/``f(r, r', E)``, a
        length-(lmax+1) sequence of callables (l-dependent), or an
        already-built ``Interaction`` (passed through; only allowed when the
        interior rescale is 1, i.e. classical kinematics).
        """
        if isinstance(term, lax.Interaction):
            if not np.all(self.grid.interior_scale == 1.0):
                raise ValueError(
                    f"{name}: pre-built Interactions cannot be used with "
                    "non-classical kinematics (the engine cannot apply the "
                    "interior V·E_lax/Ecm rescale); pass arrays or callables"
                )
            return term
        if isinstance(term, InteractionPair):
            raise TypeError(
                f"{name}: an InteractionPair is a full (V⁺, V⁻) potential, "
                "not a single term"
            )

        values, interp = self._term_arrays(
            term,
            energy_dependent=energy_dependent,
            l_dependent=l_dependent,
            name=name,
        )
        values, interp = self._apply_interior_scale(values, interp)
        return self._array_interaction(
            values,
            energy_dependent=interp.energy_dependent,
            block_dependent=interp.l_dependent,
            is_nonlocal=interp.is_nonlocal,
        )

    def spin_orbit_pair(
        self,
        term: Any,
        *,
        energy_dependent: bool | None = None,
        l_dependent: bool | None = None,
        name: str = "spin_orbit",
    ) -> InteractionPair:
        """Build the ⟨l·σ⟩-scaled (V⁺, V⁻) pair from a radial form factor.

        ``term`` is the *unscaled* spin-orbit form factor — ``(N,)`` or
        ``(N_E, N)`` array, ``f(r)``/``f(r, E)`` callable, or ``(N, N)`` /
        ``(N_E, N, N)`` non-local kernel — exactly as for the legacy
        workspaces. A leading ``(lmax+1,)`` axis is also accepted for
        intrinsically l-dependent form factors (e.g. Perey–Buck-type
        non-local spin-orbit kernels, whose partial-wave projection depends
        on l). The per-l ⟨l·σ⟩ scaling and the j = l ± ½ split happen here
        in either case.
        """
        if callable(term):
            values, e_dep, nonloc = self._evaluate_callable(
                term, energy_dependent=energy_dependent, name=name
            )
            interp = _Interpretation(False, e_dep, nonloc)
        else:
            values, interp = self._term_arrays(
                term,
                energy_dependent=energy_dependent,
                l_dependent=l_dependent,
                name=name,
            )
        values, interp = self._apply_interior_scale(values, interp)

        if interp.l_dependent:
            # the term already carries the (lmax+1,) block axis: scale it
            # per block instead of prepending a new axis
            expand = (slice(None),) + (None,) * (values.ndim - 1)
        else:
            expand = (slice(None),) + (None,) * values.ndim
        scaled_interp = _Interpretation(
            True, interp.energy_dependent, interp.is_nonlocal
        )
        members = []
        for couplings in (self._ldots_plus, self._ldots_minus):
            scaled = couplings[expand] * (
                values if interp.l_dependent else values[None]
            )
            members.append(
                self._array_interaction(
                    scaled,
                    energy_dependent=scaled_interp.energy_dependent,
                    block_dependent=True,
                    is_nonlocal=scaled_interp.is_nonlocal,
                )
            )
        return InteractionPair(plus=members[0], minus=members[1])

    # -- observables ---------------------------------------------------------

    def _smatrix_single(self, interaction: Any) -> ComplexArray:
        """(lmax+1, N_E) S-matrix for one Interaction, regime-dispatched.

        Returns a JAX array so differentiable pipelines stay intact.
        """
        if self.method == "linear_solve":
            s = self.solver.smatrix_direct(interaction)
        else:
            spectrum = self.solver.spectrum(interaction)
            use_grid = interaction.energy_dependent or not self.grid.uniform_mass_factor
            if use_grid:
                s = self.solver.smatrix_grid(spectrum)
            else:
                s = self.solver.smatrix(spectrum)
        return s[:, :, 0, 0]

    def distorted_waves(
        self, interaction: Any
    ) -> tuple[Any, ComplexArray, ComplexArray]:
        """Return ``(χ, S, conv)`` for one Interaction (not a pair).

        ``χ`` are the raw lax wavefunction mesh coefficients ``(N_b, N_E, M)``
        (boundary-value-driven); ``S`` and ``conv`` are ``(N_b, N_E)``, where
        ``conv = (i/2)(H⁻′ − S·H⁺′)/H⁻`` is the closed-form source-convention
        factor (lax DESIGN.md Appendix C.12): ``conv·χ`` is the physically
        normalized distorted wave of the legacy engine. Requires the engine to
        be built with ``wavefunctions=True``.
        """
        solver = self.solver
        if self.method == "linear_solve":
            s = solver.smatrix_direct(interaction)[:, :, 0, 0]
            chi = solver.wavefunction_direct_grid(interaction)
        else:
            spectrum = solver.spectrum(interaction)
            use_grid = interaction.energy_dependent or not self.grid.uniform_mass_factor
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

    def smatrix(self, interaction: Any) -> tuple[ComplexArray, ComplexArray]:
        """Return (S⁺, S⁻), each ``(lmax+1, N_E)``.

        For a plain ``Interaction`` (no spin-orbit) the solve runs once and
        is reused for both j. The l = 0 row of S⁻ is computed-but-meaningless
        (§3.1); callers slice it off the public surface.
        """
        if isinstance(interaction, InteractionPair):
            return (
                self._smatrix_single(interaction.plus),
                self._smatrix_single(interaction.minus),
            )
        s = self._smatrix_single(interaction)
        return s, s
