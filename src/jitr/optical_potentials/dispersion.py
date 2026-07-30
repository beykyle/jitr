"""Numerical Kramers-Kronig dispersion utilities for dispersive OMP workflows.

Two families of helpers live here:

- :class:`DispersionSolver` evaluates the plain (non-subtracted) DOM
  dispersion integral of a radially resolved ``W(r, E')`` at a fixed radial
  grid and fixed scalar energy, with quantities depending only on
  ``(r_grid, E, segments)`` precomputed at construction time and a single
  jitted (JAX) inner loop per online evaluation.
- :func:`subtracted_dispersion_correction` evaluates the *once-subtracted*
  principal-value dispersion integral of an energy-dependent imaginary
  depth ``W(E')`` about a Fermi energy numerically, as used by dispersive
  global potentials (e.g. :mod:`jitr.optical_potentials.mbra`);
  :func:`brown_rho_halfline_partner`,
  :func:`damped_brown_rho_halfline_partner`,
  :func:`sub_fermi_suppression_partner` and
  :func:`sqrt_tail_dispersive_partner` are its closed-form counterparts
  for the half-line depth shapes those potentials are built from.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import exp1, expi

from .._types import ArrayOrScalar, FloatArray

QuadratureSegment = tuple[float, float, int]
QuadratureSegments = Sequence[QuadratureSegment]

# Default piecewise Gauss-Legendre layout: about 140 nodes total, with denser
# coverage around the origin where the kernel changes most rapidly.
DEFAULT_SEGMENTS: tuple[QuadratureSegment, ...] = (
    (-400.0, -50.0, 20),
    (-50.0, -15.0, 30),
    (-15.0, 15.0, 40),
    (15.0, 50.0, 30),
    (50.0, 400.0, 20),
)


def build_quadrature(
    segments: QuadratureSegments = DEFAULT_SEGMENTS,
) -> tuple[FloatArray, FloatArray, float]:
    """Construct piecewise Gauss-Legendre quadrature nodes and weights.

    Args:
        segments: Sequence of ``(a, b, n)`` segment descriptors covering the
            integration interval.

    Raises:
        ValueError: If any segment has ``b <= a`` or ``n < 1``.

    Returns:
        ``(x_quad, w_quad, E_cut)`` where ``x_quad`` are the quadrature
        nodes, ``w_quad`` are the weights, and ``E_cut`` is the outer cutoff
        inferred from the segment endpoints.
    """

    nodes: list[FloatArray] = []
    weights: list[FloatArray] = []
    for a, b, n in segments:
        if not b > a:
            raise ValueError(f"segment endpoints must satisfy b > a, got ({a}, {b})")
        if n < 1:
            raise ValueError(f"segment node count must be >= 1, got {n}")
        x_u, w_u = leggauss(n)
        nodes.append((0.5 * (b - a) * x_u + 0.5 * (a + b)).astype(np.float64))
        weights.append((0.5 * (b - a) * w_u).astype(np.float64))

    x_quad = np.concatenate(nodes).astype(np.float64)
    w_quad = np.concatenate(weights).astype(np.float64)

    # Use the segment endpoints as the analytic integration interval. The
    # outermost Gauss-Legendre nodes lie strictly inside their segments, so
    # max(abs(x_quad)) is slightly smaller than the true cutoff used by the log
    # boundary term.
    E_cut = float(max(abs(segments[0][0]), abs(segments[-1][1])))
    return x_quad, w_quad, E_cut


@lru_cache(maxsize=1)
def _jitted_dispersion_kernel():
    """Build the jitted dispersion reduction (cached so tracing happens once)."""

    @jax.jit
    def kernel(W_grid, W_at_E, dx_inv_w, log_term):
        total = (W_grid - W_at_E[:, None]) @ dx_inv_w
        return (total + W_at_E * log_term) / jnp.pi

    return kernel


def _dispersion_kernel(
    W_grid: np.ndarray,
    W_at_E: np.ndarray,
    dx_inv_w: np.ndarray,
    log_term: float,
) -> np.ndarray:
    """Evaluate the preconditioned dispersion sum.

    Plain-NumPy fast path for concrete arrays (a single BLAS matvec, so
    per-call dispatch overhead stays negligible); the jitted JAX path engages
    when the inputs are JAX arrays/tracers, so dispersive potentials sit
    inside the differentiable potential → observable pipeline (§4).
    """
    if isinstance(W_grid, np.ndarray) and isinstance(W_at_E, np.ndarray):
        # Σ_k w_k (W_ik − W_E,i) + W_E,i·L = (W @ w)_i + W_E,i (L − Σ_k w_k)
        return (W_grid @ dx_inv_w + W_at_E * (log_term - dx_inv_w.sum())) / np.pi
    return _jitted_dispersion_kernel()(W_grid, W_at_E, dx_inv_w, log_term)


class DispersionSolver:
    r"""Precompute quadrature data for repeated DOM dispersion evaluation.

    The solver targets repeated evaluation of

    .. math::

       \Delta V(r, E) = \frac{1}{\pi}\,\mathrm{p.v.}
       \int_{-E_\mathrm{cut}}^{E_\mathrm{cut}} \frac{W(r, x)}{x - E}\,dx

    at fixed ``r_grid`` and fixed scalar ``E``. The principal-value singularity
    is handled by subtraction of ``W(r, E)`` from the integrand and an analytic
    logarithmic boundary term.

    Args:
        r_grid: Radial mesh in fm.
        E: Target energy in MeV.
        segments: Piecewise Gauss-Legendre quadrature segments.
        min_node_distance: Minimum tolerated distance between ``E`` and any
            quadrature node.

    Raises:
        ValueError: If ``r_grid`` is not one-dimensional, if ``E`` lies
            outside the quadrature interval, or if ``E`` is too close to a node.
    """

    def __init__(
        self,
        r_grid: FloatArray | Sequence[float],
        E: float,
        segments: QuadratureSegments = DEFAULT_SEGMENTS,
        min_node_distance: float = 1e-6,
    ) -> None:
        radial_grid = np.ascontiguousarray(np.asarray(r_grid, dtype=np.float64))
        if radial_grid.ndim != 1:
            raise ValueError(f"r_grid must be 1-D, got shape {radial_grid.shape}")

        energy = float(E)
        x_quad, w_quad, E_cut = build_quadrature(segments)

        if abs(energy) >= E_cut:
            raise ValueError(
                f"E = {energy} must lie strictly inside (-E_cut, E_cut) = "
                f"({-E_cut}, {E_cut})"
            )

        dx = x_quad - energy
        closest = int(np.argmin(np.abs(dx)))
        if abs(dx[closest]) < min_node_distance:
            raise ValueError(
                f"E = {energy} is within {min_node_distance} of quadrature node "
                f"x_quad[{closest}] = {x_quad[closest]}. Adjust segments, perturb E, "
                "or relax min_node_distance."
            )

        self._r_grid = radial_grid
        self._E = energy
        self._E_cut = E_cut
        self._x_quad = x_quad
        self._w_quad = w_quad
        self._dx_inv_w = np.asarray(w_quad / dx, dtype=np.float64)
        self._log_term = float(np.log(abs((E_cut - energy) / (E_cut + energy))))
        self._N_r = radial_grid.size
        self._N_q = x_quad.size

    @property
    def r_grid(self) -> FloatArray:
        """Return the radial mesh used by this solver."""

        return self._r_grid

    @property
    def E(self) -> float:
        """Return the target energy in MeV."""

        return self._E

    @property
    def E_cut(self) -> float:
        """Return the solver's outer integration cutoff in MeV."""

        return self._E_cut

    @property
    def x_quad(self) -> FloatArray:
        """Return the quadrature nodes in MeV."""

        return self._x_quad

    @property
    def w_quad(self) -> FloatArray:
        """Return the quadrature weights."""

        return self._w_quad

    @property
    def n_nodes(self) -> int:
        """Return the number of quadrature nodes."""

        return self._N_q

    @property
    def n_radial(self) -> int:
        """Return the number of radial grid points."""

        return self._N_r

    def __call__(self, W_grid: np.ndarray, W_at_E: np.ndarray) -> FloatArray:
        """Evaluate the dispersion correction on the stored radial grid.

        Args:
            W_grid: Array with shape ``(n_radial, n_nodes)`` containing
                ``W(r_grid[i], x_quad[k])``.
            W_at_E: Array with shape ``(n_radial,)`` containing
                ``W(r_grid[i], E)``.

        Raises:
            ValueError: If the input shapes do not match the solver
                configuration.

        Returns:
            Dispersion correction ``ΔV(r_grid, E)``.
        """

        W_grid_array = np.ascontiguousarray(W_grid, dtype=np.float64)
        W_at_E_array = np.ascontiguousarray(W_at_E, dtype=np.float64)

        if W_grid_array.shape != (self._N_r, self._N_q):
            raise ValueError(
                f"W_grid shape {W_grid_array.shape} does not match expected "
                f"({self._N_r}, {self._N_q})"
            )
        if W_at_E_array.shape != (self._N_r,):
            raise ValueError(
                "W_at_E shape "
                f"{W_at_E_array.shape} does not match expected ({self._N_r},)"
            )

        return _dispersion_kernel(
            W_grid_array,
            W_at_E_array,
            self._dx_inv_w,
            self._log_term,
        )


def dispersion_correction_reference(
    W_func: Callable[[float, float], float],
    r_grid: FloatArray | Sequence[float],
    E: float,
    E_cut: float = 400.0,
    **quad_kwargs: float,
) -> FloatArray:
    """Evaluate a high-accuracy SciPy Cauchy-weighted reference solution.

    Args:
        W_func: Scalar callable returning ``W(r, x)``.
        r_grid: Radial points at which to evaluate the dispersion correction.
        E: Target energy in MeV.
        E_cut: Outer integration cutoff in MeV.
        **quad_kwargs: Extra keyword arguments forwarded to
            :func:`scipy.integrate.quad`.

    Returns:
        Reference dispersion correction on ``r_grid``.
    """

    from scipy import integrate

    radial_grid = np.asarray(r_grid, dtype=float)
    quad_options = {"epsabs": 1e-7, "epsrel": 1e-6, "limit": 500, **quad_kwargs}

    out = np.empty(radial_grid.size, dtype=np.float64)
    for i, radius in enumerate(radial_grid):
        val, _ = integrate.quad(
            lambda x, radius=radius: W_func(radius, x),
            -E_cut,
            E_cut,
            weight="cauchy",
            wvar=float(E),
            **quad_options,
        )
        out[i] = val / np.pi

    return out


# -- once-subtracted dispersion about a Fermi energy ---------------------------

# Default piecewise Gauss-Legendre layout for the once-subtracted integral,
# as offsets relative to the Fermi energy: 512 nodes over E_F ± 3e4 MeV with
# denser coverage near E_F, wide enough that truncation of Brown-Rho-type
# depths (which tend to a constant at large |E'|) contributes < 0.1 MeV for
# evaluation energies up to a few hundred MeV.
DEFAULT_SUBTRACTED_SEGMENT_OFFSETS: tuple[QuadratureSegment, ...] = (
    (-3.0e4, -3.0e3, 64),
    (-3.0e3, -3.0e2, 64),
    (-3.0e2, 3.0e2, 256),
    (3.0e2, 3.0e3, 64),
    (3.0e3, 3.0e4, 64),
)


@lru_cache(maxsize=8)
def _subtracted_quadrature(
    Ef: float, segment_offsets: tuple[QuadratureSegment, ...]
) -> tuple[FloatArray, FloatArray, float, float]:
    """Quadrature for the subtracted PV dispersion integral around ``Ef``."""
    segments = tuple((Ef + a, Ef + b, n) for a, b, n in segment_offsets)
    x_quad, w_quad, _ = build_quadrature(segments)
    return x_quad, w_quad, segments[0][0], segments[-1][1]


def subtracted_dispersion_correction(
    W: Callable[[FloatArray], ArrayOrScalar],
    E: ArrayOrScalar,
    Ef: float,
    segment_offsets: QuadratureSegments | None = None,
    min_node_distance: float = 1e-6,
) -> ArrayOrScalar:
    r"""Once-subtracted PV dispersion integral of an imaginary depth, in MeV.

    .. math::

       \Delta V(E) = \frac{E - E_F}{\pi}\,\mathcal{P}\!\int
           \frac{W(E')}{(E' - E)(E' - E_F)}\, dE',

    which satisfies ΔV(E_F) = 0 and converges for depths growing slower
    than linearly (the Mahaux-Sartor :math:`\sqrt{E}` tail included).

    Args:
        W: Vectorized imaginary depth ``W(E')``; must vanish at ``E_F``
            at least quadratically.
        E: Evaluation energies in MeV.
        Ef: Fermi energy in MeV.
        segment_offsets: Piecewise Gauss-Legendre segments as ``(a, b, n)``
            offsets relative to ``Ef``; defaults to
            :data:`DEFAULT_SUBTRACTED_SEGMENT_OFFSETS`.
        min_node_distance: Minimum tolerated distance between any evaluation
            energy and any quadrature node.

    Raises:
        ValueError: If any evaluation energy lies outside the quadrature
            interval, or within ``min_node_distance`` of a quadrature node
            (the subtracted integrand's finite limit is not evaluated there;
            perturb ``E``, adjust ``segment_offsets``, or relax
            ``min_node_distance``).

    Returns:
        ΔV(E), same shape as ``E``.
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    offsets = (
        DEFAULT_SUBTRACTED_SEGMENT_OFFSETS
        if segment_offsets is None
        else tuple(tuple(seg) for seg in segment_offsets)
    )
    x_quad, w_quad, lo, hi = _subtracted_quadrature(Ef, offsets)
    if np.any(E_arr <= lo) or np.any(E_arr >= hi):
        raise ValueError(
            f"subtracted_dispersion_correction: E must lie inside the "
            f"quadrature interval ({lo:.1f}, {hi:.1f}) MeV"
        )

    g_quad = np.asarray(W(x_quad), dtype=float) / (x_quad - Ef)
    x = E_arr - Ef
    W_at_E = np.asarray(W(E_arr), dtype=float)
    g_at_E = np.divide(W_at_E, x, out=np.zeros_like(W_at_E), where=np.abs(x) > 1e-12)

    denom = x_quad[None, :] - E_arr[:, None]
    closest = np.min(np.abs(denom), axis=1)
    if np.any(closest < min_node_distance):
        i = int(np.argmin(closest))
        k = int(np.argmin(np.abs(denom[i])))
        raise ValueError(
            f"E = {E_arr[i]} is within {min_node_distance} of quadrature node "
            f"x_quad[{k}] = {x_quad[k]}. Perturb E, adjust segment_offsets, "
            "or relax min_node_distance."
        )
    diff = g_quad[None, :] - g_at_E[:, None]
    ratio = np.divide(
        diff, denom, out=np.zeros_like(diff), where=np.abs(denom) >= min_node_distance
    )
    pv = ratio @ w_quad + g_at_E * np.log((hi - E_arr) / (E_arr - lo))
    out = x * pv / np.pi
    return out.item() if np.ndim(E) == 0 else out


def brown_rho_halfline_partner(
    E: ArrayOrScalar, Ef: float, B: float
) -> ArrayOrScalar:
    r"""Closed-form partner of a half-line Brown-Rho depth, per unit amplitude.

    Once-subtracted PV dispersion (the integral of
    :func:`subtracted_dispersion_correction`) of

    .. math::

       W(E') = \frac{x'^2}{x'^2 + B^2}, \quad x' = E' - E_F > 0

    (zero below the Fermi energy). With :math:`x = E - E_F`,

    .. math::

       D^+(x) = \frac{x}{x^2 + B^2}
           \left[\frac{B}{2} + \frac{x}{\pi} \ln\frac{B}{|x|}\right],

    obtained by partial fractions (the pole/residue technique of Quesada,
    Capote, Molina, Lozano & Raynal, Phys. Rev. C 67, 067601 (2003),
    applied to a single half-line). The depth supported on ``E' < E_F``
    instead contributes ``-D^+(-x)``, i.e. this function evaluated at the
    reflected energy ``2 E_F - E`` and negated; the symmetric combination
    recovers the textbook :math:`B x/(x^2 + B^2)`. Exact for all ``E``
    (no quadrature interval or node restrictions), and exactly zero at
    ``E = E_F``.
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = E_arr - Ef
    x_safe = np.where(x == 0.0, 1.0, x)
    out = (
        x_safe
        / (x_safe**2 + B**2)
        * (B / 2.0 + x_safe / np.pi * np.log(B / np.abs(x_safe)))
    )
    out = np.where(x == 0.0, 0.0, out)
    return out.item() if np.ndim(E) == 0 else out


def damped_brown_rho_halfline_partner(
    E: ArrayOrScalar, Ef: float, B: float, C: float
) -> ArrayOrScalar:
    r"""Closed-form partner of a half-line damped Brown-Rho depth.

    Once-subtracted PV dispersion, per unit amplitude, of the
    exponentially-damped Brown-Rho surface shape

    .. math::

       W(E') = \frac{x'^2\, e^{-C x'}}{x'^2 + B^2}, \quad x' = E' - E_F > 0

    (zero below the Fermi energy). Partial fractions leave one real
    principal-value pole at :math:`x = E - E_F` and the conjugate pair
    :math:`\pm iB`; the half-line exponential integrals give (Quesada
    et al., Phys. Rev. C 67, 067601 (2003), Eq. 14, per half-line)

    .. math::

       D_S^+(x) = \frac{x}{\pi}\left[-\frac{x\, e^{-Cx}\,
           \mathrm{Ei}(Cx)}{x^2 + B^2}
           + 2\,\mathrm{Re}\,\frac{e^{-iBC} E_1(-iBC)}{2(iB - x)}\right].

    The depth supported on ``E' < E_F`` contributes ``-D_S^+(-x)`` (this
    function at ``2 E_F - E``, negated). Exact for all ``E`` and exactly
    zero at ``E = E_F``.
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = E_arr - Ef
    x_safe = np.where(x == 0.0, 1.0, x)
    real_pole = -x_safe / (x_safe**2 + B**2) * np.exp(-C * x_safe) * expi(C * x_safe)
    pair = 2.0 * np.real(
        np.exp(-1j * C * B) * exp1(-1j * C * B) / (2.0 * (1j * B - x_safe))
    )
    out = x_safe / np.pi * (real_pole + pair)
    out = np.where(x == 0.0, 0.0, out)
    return out.item() if np.ndim(E) == 0 else out


def sub_fermi_suppression_partner(
    E: ArrayOrScalar, Ef: float, B: float, ev_minus: float
) -> ArrayOrScalar:
    r"""Closed-form partner of the sub-Fermi Brown-Rho suppression piece.

    A Brown-Rho depth suppressed below ``E_F - E_V^-`` by the factor
    :math:`E_V^{-2}/(x_l^2 + E_V^{-2})` (``x_l = x' + E_V^-``; Eq. 16 of
    the MBRA paper, arXiv:2403.05843) equals the plain sub-Fermi Brown-Rho
    minus the piece dispersed here, per unit amplitude:

    .. math::

       W(E') = \frac{x'^2}{x'^2 + B^2}\,
               \frac{x_l^2}{x_l^2 + E_V^{-2}}, \quad x' < -E_V^-.

    Substituting :math:`v = -(x' + E_V^-)` maps the once-subtracted PV
    integral onto a rational half-line integral with five simple poles
    whose residues sum to zero, so (Quesada et al., Phys. Rev. C 67,
    067601 (2003), Eq. 17 limit) :math:`D_c(x) = -(x/\pi) \sum_k R_k(x)
    \ln(-p_k)`, with the real-pole log taken as a principal value
    :math:`\ln|E_V^- + x|`. Its residue vanishes like
    :math:`(E_V^- + x)^2`, so ``D_c`` is continuous through the branch
    point ``E = E_F - E_V^-``; it is exactly zero at ``E = E_F``.
    """
    e = ev_minus
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = E_arr - Ef
    x_safe = np.where(x == 0.0, 1.0, x)
    # residues of (v+e) v^2 / [((v+e)^2 + B^2)(v^2 + e^2)(v + e + x)]
    R0 = -x_safe * (e + x_safe) ** 2 / ((x_safe**2 + B**2) * ((e + x_safe) ** 2 + e**2))
    p_ie = 1j * e
    R_ie = ((p_ie + e) * p_ie**2) / (
        ((p_ie + e) ** 2 + B**2) * (2.0 * p_ie) * (p_ie + e + x_safe)
    )
    p_b = -e + 1j * B
    R_b = ((p_b + e) * p_b**2) / (
        (2.0 * (p_b + e)) * (p_b**2 + e**2) * (p_b + e + x_safe)
    )
    ex = np.abs(e + x_safe)
    log_real = np.log(np.where(ex < 1e-300, 1.0, ex))  # R0 ~ (e+x)^2 kills it
    val = (
        -R0 * log_real
        - 2.0 * np.real(R_ie * np.log(-p_ie))
        - 2.0 * np.real(R_b * np.log(-p_b))
    )
    out = x_safe / np.pi * val
    out = np.where(x == 0.0, 0.0, out)
    return out.item() if np.ndim(E) == 0 else out


def sqrt_tail_dispersive_partner(
    el: float, E: ArrayOrScalar, Ef: float
) -> FloatArray:
    r"""Dispersive partner of the Mahaux-Sartor sqrt(E) tail (per unit α).

    Closed form for the once-subtracted dispersion integral of the
    asymptotic Mahaux-Sartor term
    :math:`\sqrt{E} + e_l^{3/2}/(2E) - \tfrac{3}{2}\sqrt{e_l}` for
    ``E > e_l`` (zero below), transcribed from the ECIS-06 routine ``dlpe``
    (J. Raynal), which implements the same imaginary-volume form. Vanishes
    at ``E = E_F``. Validated against brute-force PV integration to
    < 0.05 MeV.
    """
    ex = np.atleast_1d(np.asarray(E, dtype=float))
    ff = np.sqrt(abs(Ef))
    fl = np.sqrt(abs(el))
    fx = np.sqrt(np.abs(ex))
    base = (
        2.0 * ff * np.arctan2(ff, fl)
        + 0.5 * el * fl / Ef * np.log(1.0 - Ef / el)
        - 1.5 * fl * np.log(abs(el - Ef))
    )
    out = np.full_like(ex, base)

    pos = ex > 0.0
    neg = ~pos
    xn = ex[neg]
    small_n = np.abs(xn) <= el * 1e-5
    xn_safe = np.where(small_n, 1.0, xn)
    t_neg = np.where(
        small_n,
        0.5 * fl * (1.0 + xn / el / 2.0 + (xn / el) ** 2 / 3.0),
        -0.5 * el * fl / xn_safe * np.log(np.abs(1.0 - xn / el)),
    )
    out[neg] += (
        t_neg
        - 2.0 * fx[neg] * np.arctan2(fx[neg], fl)
        + 1.5 * fl * np.log(np.abs(el - xn))
    )

    xp = ex[pos]
    fxp = fx[pos]
    t_pos = (fxp + 1.5 * fl - 0.5 * el * fl / xp) * np.log(fl + fxp) + (
        el * fl / xp
    ) * np.log(fl)
    far = np.abs(xp - el) > 1e-3
    t_pos -= np.where(
        far,
        (fxp - 1.5 * fl + 0.5 * el * fl / xp)
        * np.log(np.where(far, np.abs(fl - fxp), 1.0)),
        0.0,
    )
    out[pos] += t_pos

    out = np.where(np.abs(ex - Ef) < 1e-12, 0.0, out)
    return out / np.pi
