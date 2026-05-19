"""Numerical Kramers-Kronig dispersion utilities for local DOM workflows.

The solver is optimized for repeated evaluation at a fixed radial grid and a
fixed scalar energy. Quantities that depend only on ``(r_grid, E, segments)``
are precomputed at construction time, leaving a single numba-compiled inner
loop for each online evaluation.

This makes the module useful for non-analytic or non-separable imaginary
potentials ``W(r, E)`` where an analytic dispersive correction is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss

from .._types import FloatArray

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


@njit(cache=True, fastmath=True)
def _dispersion_kernel(
    W_grid: np.ndarray,
    W_at_E: np.ndarray,
    dx_inv_w: np.ndarray,
    log_term: float,
) -> np.ndarray:
    """Evaluate the preconditioned dispersion sum in numba."""

    N_r, N_q = W_grid.shape
    inv_pi = 1.0 / np.pi
    dV = np.empty(N_r, dtype=np.float64)

    for i in range(N_r):
        W_E = W_at_E[i]
        total = 0.0
        for k in range(N_q):
            total += dx_inv_w[k] * (W_grid[i, k] - W_E)
        dV[i] = (total + W_E * log_term) * inv_pi

    return dV


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
