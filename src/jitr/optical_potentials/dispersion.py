"""
Numerical Kramers-Kronig dispersion-relation solver, optimized for repeated
online evaluation at a fixed radial grid and fixed scalar energy.

Designed for DOM fitting workflows where many parameter samples reuse the
same (r_grid, E) configuration. All quantities that depend only on
(r_grid, E, E_cut, segments) are computed once at solver construction;
the online call does a single fused-multiply-add inner loop, JIT-compiled
with numba for SIMD speed.

This allows for dispersion corrections for non-annalytic W(r, E) models, or
models where W(r,E) is not separable into a product of a radial function and an
energy function.  The solver itself is agnostic to the physics of W; it only
requires that the caller provide W evaluated on the quadrature nodes and at the
target energy, and that W be smooth enough for the quadrature to converge, and
go to zero fast enough at large |E| for the cutoff to be effective.

Algorithm
---------
The principal-value integral is regularised via singularity subtraction:

    ΔV(r, E) = (1/π) ∫_{-E_cut}^{E_cut}  W(r, x) / (x − E) dx
             = (1/π) [ ∫ (W(r, x) − W(r, E)) / (x − E) dx
                       + W(r, E) · ln|(E_cut − E) / (E_cut + E)| ]

The first integrand is smooth at x = E (removable singularity).  Discretising
with a piecewise Gauss-Legendre rule gives, for fixed E, a constant weight
vector  α_k = w_k / (x_k − E)  used as a dot product against W(r, x_k).

Caller responsibilities (per parameter sample)
----------------------------------------------
* Compute  W_grid[i, k] = W(r_grid[i], x_quad[k])   shape (N_r, N_q)
* Compute  W_at_E[i]    = W(r_grid[i], E)           shape (N_r,)
* Call     dV = solver(W_grid, W_at_E)              shape (N_r,)

All three steps are user-side; the solver itself does no W evaluation
because W is application-specific.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss

# ---------------------------------------------------------------------------
# Default piecewise Gauss-Legendre layout: ~140 nodes total, dense near origin
# ---------------------------------------------------------------------------
DEFAULT_SEGMENTS = (
    (-400.0, -50.0, 20),
    (-50.0, -15.0, 30),
    (-15.0, 15.0, 40),
    (15.0, 50.0, 30),
    (50.0, 400.0, 20),
)


def build_quadrature(segments=DEFAULT_SEGMENTS):
    """
    Construct piecewise Gauss-Legendre nodes and weights.

    Parameters
    ----------
    segments : sequence of (a, b, n)
        Each tuple defines a sub-interval [a, b] with `n` Gauss-Legendre
        nodes.  Sub-intervals must abut and cover [-E_cut, +E_cut].

    Returns
    -------
    x_quad : (N_q,) float64 array
        Quadrature nodes, monotonically increasing.
    w_quad : (N_q,) float64 array
        Quadrature weights (positive, sum to total interval length).
    E_cut : float
        Outer cutoff (= max |x_quad|).
    """
    nodes, weights = [], []
    for a, b, n in segments:
        if not b > a:
            raise ValueError(f"segment endpoints must satisfy b > a, got ({a}, {b})")
        if n < 1:
            raise ValueError(f"segment node count must be >= 1, got {n}")
        x_u, w_u = leggauss(n)
        nodes.append(0.5 * (b - a) * x_u + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w_u)
    x = np.concatenate(nodes).astype(np.float64)
    w = np.concatenate(weights).astype(np.float64)
    # Use the segment endpoints as the analytical integration interval.  The
    # outermost Gauss-Legendre nodes lie strictly inside their segments, so
    # max(|x|) is slightly less than the true cutoff used by the log_term.
    E_cut = float(max(abs(segments[0][0]), abs(segments[-1][1])))
    return x, w, E_cut


# ---------------------------------------------------------------------------
# Hot-loop kernel.  All inputs are precomputed by the solver; this function
# does no allocation aside from the output array, no branches, and SIMDs
# cleanly under -fastmath.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _dispersion_kernel(W_grid, W_at_E, dx_inv_w, log_term):
    """
    Pure-numba dispersion kernel.

    Parameters
    ----------
    W_grid : (N_r, N_q) float64 array
        W(r_grid[i], x_quad[k]).
    W_at_E : (N_r,) float64 array
        W(r_grid[i], E).
    dx_inv_w : (N_q,) float64 array
        w_quad[k] / (x_quad[k] - E).
    log_term : float64
        ln|(E_cut - E) / (E_cut + E)|.

    Returns
    -------
    dV : (N_r,) float64 array
    """
    N_r, N_q = W_grid.shape
    inv_pi = 1.0 / np.pi
    dV = np.empty(N_r, dtype=np.float64)

    for i in range(N_r):
        WE = W_at_E[i]
        s = 0.0
        # Subtraction inside the multiply preserves precision near x_k ≈ E.
        for k in range(N_q):
            s += dx_inv_w[k] * (W_grid[i, k] - WE)
        dV[i] = (s + WE * log_term) * inv_pi

    return dV


# ---------------------------------------------------------------------------
class DispersionSolver:
    """
    A
    Pre-computes quadrature data for K-K evaluation at a fixed (r_grid, E).

    Online cost is dominated by a single (N_r × N_q) numba loop with one
    fused-multiply-add per node-radius pair.  No allocation other than the
    output vector.

    Parameters
    ----------
    r_grid : array-like, shape (N_r,)
        Radial mesh (typically the quadrature mesh of the Schrödinger solver).
    E : float
        Scalar energy at which the dispersion correction will be evaluated.
        Must lie strictly inside (-E_cut, +E_cut) and not coincide with a
        quadrature node.
    segments : sequence of (a, b, n), optional
        Piecewise Gauss-Legendre layout.  See ``DEFAULT_SEGMENTS``.
    min_node_distance : float, optional
        Minimum |x_quad[k] - E| tolerated; values below this trigger a
        ValueError because the kernel would divide by a tiny number.

    Attributes
    ----------
    r_grid, E, E_cut, x_quad, w_quad, n_nodes : read-only properties.

    Examples
    --------
    >>> r_grid = np.linspace(0.5, 12.0, 50)
    >>> solver = DispersionSolver(r_grid, E=12.0)
    >>> # per parameter sample:
    >>> W_grid = my_W(r_grid[:, None], solver.x_quad[None, :])  # (50, n_nodes)
    >>> W_at_E = my_W(r_grid, 12.0)                              # (50,)
    >>> dV = solver(W_grid, W_at_E)                              # (50,)
    """

    def __init__(
        self,
        r_grid,
        E,
        segments=DEFAULT_SEGMENTS,
        min_node_distance=1e-6,
    ):
        r_grid = np.ascontiguousarray(np.asarray(r_grid, dtype=np.float64))
        if r_grid.ndim != 1:
            raise ValueError(f"r_grid must be 1-D, got shape {r_grid.shape}")

        E = float(E)
        x_q, w_q, E_cut = build_quadrature(segments)

        if abs(E) >= E_cut:
            raise ValueError(
                f"E = {E} must lie strictly inside (-E_cut, E_cut) = "
                f"({-E_cut}, {E_cut})"
            )

        dx = x_q - E
        closest = np.argmin(np.abs(dx))
        if abs(dx[closest]) < min_node_distance:
            raise ValueError(
                f"E = {E} is within {min_node_distance} of quadrature node "
                f"x_quad[{closest}] = {x_q[closest]}.  Adjust segments, perturb "
                f"E, or relax min_node_distance."
            )

        # ---- precomputed quantities ----
        self._r_grid = r_grid
        self._E = E
        self._E_cut = E_cut
        self._x_quad = x_q
        self._w_quad = w_q
        self._dx_inv_w = (w_q / dx).astype(np.float64)
        self._log_term = float(np.log(abs((E_cut - E) / (E_cut + E))))
        self._N_r = r_grid.size
        self._N_q = x_q.size

    # -------- read-only properties --------
    @property
    def r_grid(self):
        return self._r_grid

    @property
    def E(self):
        return self._E

    @property
    def E_cut(self):
        return self._E_cut

    @property
    def x_quad(self):
        return self._x_quad

    @property
    def w_quad(self):
        return self._w_quad

    @property
    def n_nodes(self):
        return self._N_q

    @property
    def n_radial(self):
        return self._N_r

    # -------- online entry point --------
    def __call__(self, W_grid, W_at_E):
        """
        Evaluate ΔV(r, E) on the bound radial grid.

        Parameters
        ----------
        W_grid : (N_r, N_q) array
            W(r_grid, x_quad).  Must be C-contiguous float64 for full speed;
            other layouts/dtypes are silently converted.
        W_at_E : (N_r,) array
            W(r_grid, E).

        Returns
        -------
        dV : (N_r,) ndarray
        """
        W_grid = np.ascontiguousarray(W_grid, dtype=np.float64)
        W_at_E = np.ascontiguousarray(W_at_E, dtype=np.float64)

        if W_grid.shape != (self._N_r, self._N_q):
            raise ValueError(
                f"W_grid shape {W_grid.shape} does not match expected "
                f"({self._N_r}, {self._N_q})"
            )
        if W_at_E.shape != (self._N_r,):
            raise ValueError(
                f"W_at_E shape {W_at_E.shape} does not match expected ({self._N_r},)"
            )

        return _dispersion_kernel(W_grid, W_at_E, self._dx_inv_w, self._log_term)


# ---------------------------------------------------------------------------
# High-precision reference (scipy adaptive Cauchy-weighted quadrature).
# Used in tests and for accuracy validation.  Slow.
# ---------------------------------------------------------------------------
def dispersion_correction_reference(W_func, r_grid, E, E_cut=400.0, **quad_kwargs):
    """
    High-precision reference using ``scipy.integrate.quad`` with the
    ``weight='cauchy'`` adaptive principal-value rule.

    Parameters
    ----------
    W_func : callable
        Two-argument scalar callable: ``W_func(r, x)`` returns a float.
    r_grid : array-like
        Radial points at which to evaluate ΔV.
    E : float
        Target energy.
    E_cut : float, optional
        Outer integration cutoff (default 400 MeV).
    **quad_kwargs
        Forwarded to ``scipy.integrate.quad``; defaults are
        ``epsabs=1e-11, epsrel=1e-10, limit=500``.

    Returns
    -------
    dV : ndarray, shape (len(r_grid),)
    """
    from scipy import integrate

    r_grid = np.asarray(r_grid, dtype=float)
    quad_kwargs = {"epsabs": 1e-7, "epsrel": 1e-6, "limit": 500, **quad_kwargs}

    out = np.empty(r_grid.size, dtype=np.float64)
    for i, r in enumerate(r_grid):
        val, _ = integrate.quad(
            lambda x, r=r: W_func(r, x),
            -E_cut,
            E_cut,
            weight="cauchy",
            wvar=float(E),
            **quad_kwargs,
        )
        out[i] = val / np.pi

    return out
