"""
Test suite for dom_dispersion module.

Tests are organized into:
    TestBuildQuadrature      — quadrature setup, boundaries, weights
    TestSolverInitialization — construction, validation, properties
    TestKernelCorrectness    — physics tests (constant, separable, parity, ...)
    TestAccuracyVsReference  — numerical agreement with scipy reference
    TestEdgeCases            — E near boundary, E=0, dtype handling
    TestUserAPIShape         — input validation, output shapes
    TestPerformance          — sanity: solver is much faster than reference
"""

import time
import unittest

import numpy as np

from jitr.optical_potentials import dom
from jitr.optical_potentials.dispersion import (
    DEFAULT_SEGMENTS,
    DispersionSolver,
    build_quadrature,
    dispersion_correction_reference,
)


# ===========================================================================
# Fixtures: physics functions for the typical DOM surface use case
# ===========================================================================
def Ws_depth(x, Ws0=16.2, ws1=12.5, ws2=0.0214):
    return dom.Ws_depth(x, Ws0, ws1, ws2)


def quartic_a(E, a_min=0.30, a_max=0.60, Lambda_a=12.0):
    E2 = E * E
    return a_min + (a_max - a_min) * E2 * E2 / (E2 * E2 + Lambda_a**4)


def W_surface(r, x, R=4.55, **kw):
    """Surface imaginary potential W_d(x) · sech²((r-R)/(2 a(x)))."""
    a = quartic_a(
        x, **{k: v for k, v in kw.items() if k in ("a_min", "a_max", "Lambda_a")}
    )
    Wd_kw = {k: v for k, v in kw.items() if k in ("Ws0", "ws1", "ws2")}
    Wd = Ws_depth(x, **Wd_kw)
    return Wd / np.cosh((r - R) / (2.0 * a)) ** 2


def W_surface_vec(r_grid, x, **kw):
    """W_surface vectorised over r_grid for fixed scalar x."""
    return np.array([W_surface(r, x, **kw) for r in r_grid])


def delta_Wd_analytic(E, Ws0=16.2, ws1=12.5, ws2=0.0214):
    return dom.delta_Vs_analytic(E, Ws0, ws1, ws2)


# ===========================================================================
class TestBuildQuadrature(unittest.TestCase):
    """Quadrature setup primitives."""

    def test_default_node_count(self):
        x, w, E_cut = build_quadrature()
        # Sum of segment node counts in DEFAULT_SEGMENTS
        expected = sum(n for (_, _, n) in DEFAULT_SEGMENTS)
        self.assertEqual(len(x), expected)
        self.assertEqual(len(w), expected)

    def test_E_cut_matches_segment_endpoints(self):
        _, _, E_cut = build_quadrature()
        outer = max(abs(DEFAULT_SEGMENTS[0][0]), abs(DEFAULT_SEGMENTS[-1][1]))
        self.assertAlmostEqual(E_cut, outer)

    def test_nodes_are_inside_interval(self):
        x, _, E_cut = build_quadrature()
        self.assertGreaterEqual(x.min(), -E_cut)
        self.assertLessEqual(x.max(), E_cut)

    def test_nodes_are_strictly_increasing(self):
        x, _, _ = build_quadrature()
        self.assertTrue(np.all(np.diff(x) > 0))

    def test_weights_positive(self):
        _, w, _ = build_quadrature()
        self.assertTrue(np.all(w > 0))

    def test_weights_sum_to_total_length(self):
        _, w, _ = build_quadrature()
        total = sum(b - a for (a, b, _) in DEFAULT_SEGMENTS)
        self.assertAlmostEqual(w.sum(), total, places=8)

    def test_custom_segments(self):
        segs = ((-100.0, 0.0, 10), (0.0, 100.0, 10))
        x, w, E_cut = build_quadrature(segs)
        self.assertEqual(len(x), 20)
        self.assertAlmostEqual(E_cut, 100.0)
        self.assertAlmostEqual(w.sum(), 200.0, places=8)

    def test_invalid_segment_order_raises(self):
        with self.assertRaises(ValueError):
            build_quadrature(((0.0, -10.0, 5),))  # b < a

    def test_invalid_node_count_raises(self):
        with self.assertRaises(ValueError):
            build_quadrature(((-1.0, 1.0, 0),))


# ===========================================================================
class TestSolverInitialization(unittest.TestCase):
    """Solver construction and input validation."""

    def setUp(self):
        self.r_grid = np.linspace(0.5, 12.0, 20)

    def test_basic_construction(self):
        solver = DispersionSolver(self.r_grid, E=12.0)
        self.assertEqual(solver.n_radial, 20)
        self.assertEqual(solver.E, 12.0)
        self.assertGreater(solver.n_nodes, 100)

    def test_properties_match_inputs(self):
        solver = DispersionSolver(self.r_grid, E=10.0)
        np.testing.assert_array_equal(solver.r_grid, self.r_grid)
        self.assertEqual(solver.E_cut, 400.0)
        self.assertEqual(len(solver.x_quad), solver.n_nodes)
        self.assertEqual(len(solver.w_quad), solver.n_nodes)

    def test_E_outside_range_raises(self):
        with self.assertRaises(ValueError):
            DispersionSolver(self.r_grid, E=500.0)
        with self.assertRaises(ValueError):
            DispersionSolver(self.r_grid, E=-500.0)
        with self.assertRaises(ValueError):
            DispersionSolver(self.r_grid, E=400.0)  # exactly on boundary

    def test_E_at_origin_works(self):
        # E = 0 is a generic point — log_term = 0, dx_inv_w well-defined
        solver = DispersionSolver(self.r_grid, E=0.0)
        self.assertEqual(solver.E, 0.0)

    def test_E_too_close_to_node_raises(self):
        # construct quadrature, then place E exactly on a node
        x, _, _ = build_quadrature()
        node = x[len(x) // 2]
        with self.assertRaises(ValueError):
            DispersionSolver(self.r_grid, E=float(node))

    def test_2d_r_grid_raises(self):
        with self.assertRaises(ValueError):
            DispersionSolver(np.zeros((2, 3)), E=10.0)

    def test_r_grid_dtype_promoted(self):
        # int input should be silently promoted to float64
        solver = DispersionSolver([1, 2, 3, 4, 5], E=10.0)
        self.assertEqual(solver.r_grid.dtype, np.float64)


# ===========================================================================
class TestKernelCorrectness(unittest.TestCase):
    """
    Physics-style correctness tests using exactly known closed forms.
    """

    def setUp(self):
        self.r_grid = np.linspace(0.5, 12.0, 30)
        self.E = 12.0
        self.solver = DispersionSolver(self.r_grid, E=self.E)
        self.N_r = self.solver.n_radial
        self.N_q = self.solver.n_nodes

    def test_zero_W_gives_zero_dV(self):
        W_grid = np.zeros((self.N_r, self.N_q))
        W_at_E = np.zeros(self.N_r)
        dV = self.solver(W_grid, W_at_E)
        np.testing.assert_array_equal(dV, np.zeros(self.N_r))

    def test_constant_W_gives_W_log_over_pi(self):
        """
        For W(r, x) ≡ c (independent of x), the integrand is c·log_term
        and the K-K should give exactly c · log_term / π.
        """
        c = 3.0
        W_grid = np.full((self.N_r, self.N_q), c)
        W_at_E = np.full(self.N_r, c)
        dV = self.solver(W_grid, W_at_E)

        E_cut = self.solver.E_cut
        log_term = np.log(abs((E_cut - self.E) / (E_cut + self.E)))
        expected = c * log_term / np.pi
        np.testing.assert_allclose(dV, np.full(self.N_r, expected), rtol=1e-12)

    def test_separable_W(self):
        """
        For W(r, x) = f(r) · g(x), the K-K factorises:
            ΔV(r) = f(r) · KK[g](E)
        Use g(x) = Wd(x), whose K-K we know in closed form.
        """
        f_r = 1.0 + 0.3 * np.cos(self.r_grid)
        g_x = Ws_depth(self.solver.x_quad)
        W_grid = f_r[:, None] * g_x[None, :]
        W_at_E = f_r * Ws_depth(self.E)

        dV = self.solver(W_grid, W_at_E)

        kk_g = delta_Wd_analytic(self.E)
        expected = f_r * kk_g
        # Quadrature is finite-resolution: tolerance reflects truncation/
        # discretisation error of the depth K-K, not solver bugs.
        np.testing.assert_allclose(dV, expected, atol=1e-3, rtol=1e-3)

    def test_separable_W_high_accuracy_with_dense_quad(self):
        """Same as above but with denser quadrature AND wider E_cut."""
        # Wide E_cut to suppress truncation tail (analytic K-K is on full line);
        # dense nodes near origin where the integrand has structure.
        dense = (
            (-2000.0, -200.0, 30),
            (-200.0, -50.0, 40),
            (-50.0, -15.0, 60),
            (-15.0, 15.0, 80),
            (15.0, 50.0, 60),
            (50.0, 200.0, 40),
            (200.0, 2000.0, 30),
        )
        solver = DispersionSolver(self.r_grid, E=self.E, segments=dense)
        f_r = 1.0 + 0.3 * np.cos(self.r_grid)
        g_x = Ws_depth(solver.x_quad)
        W_grid = f_r[:, None] * g_x[None, :]
        W_at_E = f_r * Ws_depth(self.E)
        dV = solver(W_grid, W_at_E)

        expected = f_r * delta_Wd_analytic(self.E)
        np.testing.assert_allclose(dV, expected, atol=1e-5, rtol=1e-5)

    def test_parity_for_even_W(self):
        """
        For W(r, x) even in x, ΔV(r, E) is odd in E.
        """
        E = 8.0
        s_pos = DispersionSolver(self.r_grid, E=+E)
        s_neg = DispersionSolver(self.r_grid, E=-E)

        # W_surface is even in x
        W_grid_pos = np.array(
            [[W_surface(r, x) for x in s_pos.x_quad] for r in self.r_grid]
        )
        W_grid_neg = np.array(
            [[W_surface(r, x) for x in s_neg.x_quad] for r in self.r_grid]
        )
        W_at_pos = np.array([W_surface(r, +E) for r in self.r_grid])
        W_at_neg = np.array([W_surface(r, -E) for r in self.r_grid])

        dV_pos = s_pos(W_grid_pos, W_at_pos)
        dV_neg = s_neg(W_grid_neg, W_at_neg)
        np.testing.assert_allclose(dV_pos + dV_neg, np.zeros_like(dV_pos), atol=5e-3)


# ===========================================================================
class TestAccuracyVsReference(unittest.TestCase):
    """
    Cross-validate against scipy adaptive Cauchy-weighted reference.
    """

    def setUp(self):
        self.R = 4.55
        self.r_grid = np.array(
            [
                self.R - 0.5,
                self.R,
                self.R + 0.6,
                self.R + 1.2,
                self.R + 1.8,
                self.R + 2.5,
            ]
        )

    def _W_func(self, r, x):
        return W_surface(r, x, R=self.R)

    def _solver_eval(self, E):
        solver = DispersionSolver(self.r_grid, E=E)
        W_grid = np.array(
            [[self._W_func(r, x) for x in solver.x_quad] for r in self.r_grid]
        )
        W_at_E = np.array([self._W_func(r, E) for r in self.r_grid])
        return solver(W_grid, W_at_E)

    def _reference(self, E):
        return dispersion_correction_reference(self._W_func, self.r_grid, E)

    def _check_at_E(self, E, atol=1e-3, rtol=1e-3):
        ref = self._reference(E)
        got = self._solver_eval(E)
        np.testing.assert_allclose(
            got, ref, atol=atol, rtol=rtol, err_msg=f"mismatch at E={E}"
        )

    def test_at_low_energy(self):
        self._check_at_E(2.0, atol=1e-3, rtol=1e-3)

    def test_at_peak_dispersion_energy(self):
        self._check_at_E(12.0, atol=1e-3, rtol=1e-3)

    def test_at_zero_crossing(self):
        self._check_at_E(25.0, atol=1e-3, rtol=1e-3)

    def test_at_high_energy(self):
        self._check_at_E(60.0, atol=1e-3, rtol=1e-3)

    def test_at_negative_energy(self):
        self._check_at_E(-10.0, atol=1e-3, rtol=1e-3)


# ===========================================================================
class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.r_grid = np.linspace(0.5, 10.0, 15)

    def test_E_zero_gives_zero_for_even_W(self):
        """K-K of an even W at E=0 is exactly zero by parity."""
        solver = DispersionSolver(self.r_grid, E=0.0)
        W_grid = np.array(
            [[W_surface(r, x) for x in solver.x_quad] for r in self.r_grid]
        )
        W_at_E = np.array([W_surface(r, 0.0) for r in self.r_grid])
        dV = solver(W_grid, W_at_E)
        np.testing.assert_allclose(dV, np.zeros_like(dV), atol=5e-4)

    def test_E_close_to_boundary(self):
        # Should still construct successfully; log_term is finite but large
        solver = DispersionSolver(self.r_grid, E=395.0)
        self.assertLess(abs(solver._log_term), 10.0)

    def test_dtype_promotion_of_inputs(self):
        # int W_grid / W_at_E should be silently promoted
        solver = DispersionSolver(self.r_grid, E=12.0)
        W_grid = np.zeros((solver.n_radial, solver.n_nodes), dtype=np.int32)
        W_at_E = np.zeros(solver.n_radial, dtype=np.int32)
        dV = solver(W_grid, W_at_E)
        self.assertEqual(dV.dtype, np.float64)
        np.testing.assert_array_equal(dV, np.zeros_like(dV))


# ===========================================================================
class TestUserAPIShape(unittest.TestCase):
    def setUp(self):
        self.r_grid = np.linspace(0.5, 12.0, 20)
        self.solver = DispersionSolver(self.r_grid, E=12.0)

    def test_output_shape(self):
        W_grid = np.zeros((self.solver.n_radial, self.solver.n_nodes))
        W_at_E = np.zeros(self.solver.n_radial)
        dV = self.solver(W_grid, W_at_E)
        self.assertEqual(dV.shape, (self.solver.n_radial,))

    def test_wrong_W_grid_shape_raises(self):
        bad = np.zeros((self.solver.n_radial + 1, self.solver.n_nodes))
        good_at_E = np.zeros(self.solver.n_radial)
        with self.assertRaises(ValueError):
            self.solver(bad, good_at_E)

    def test_wrong_W_at_E_shape_raises(self):
        good = np.zeros((self.solver.n_radial, self.solver.n_nodes))
        bad = np.zeros(self.solver.n_radial + 1)
        with self.assertRaises(ValueError):
            self.solver(good, bad)


# ===========================================================================
class TestPerformance(unittest.TestCase):
    """
    Sanity-check the production claim: solver should be ≫ reference speed.
    Not strict in absolute timings, just relative.
    """

    def test_solver_faster_than_reference(self):
        r_grid = np.linspace(0.5, 12.0, 50)
        E = 12.0
        solver = DispersionSolver(r_grid, E=E)

        # warm up numba
        W_grid = np.array([[W_surface(r, x) for x in solver.x_quad] for r in r_grid])
        W_at_E = np.array([W_surface(r, E) for r in r_grid])
        _ = solver(W_grid, W_at_E)

        # time online call only (W_grid construction is user-side and
        # would be done once per parameter sample in production)
        n_rep = 200
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = solver(W_grid, W_at_E)
        t_solver = (time.perf_counter() - t0) / n_rep

        t0 = time.perf_counter()
        _ = dispersion_correction_reference(lambda r, x: W_surface(r, x), r_grid, E)
        t_ref = time.perf_counter() - t0

        speedup = t_ref / t_solver
        self.assertGreater(
            speedup,
            50.0,
            msg=f"solver online {1e6 * t_solver:.1f} µs, "
            f"ref {1e3 * t_ref:.1f} ms, "
            f"speedup {speedup:.1f}× (expected > 50×)",
        )


# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
