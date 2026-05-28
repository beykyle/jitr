"""Tests for :mod:`jitr.utils.poly`."""

from __future__ import annotations

import numpy as np
import pytest

from jitr.utils import poly


class TestPolyEvaluation:
    def test_poly1d_evaluates_scalar_and_vector_inputs(self):
        coeffs = np.array([1.5, -2.0, 0.25])

        assert poly.poly1d(2.0, coeffs) == pytest.approx(1.5 - 4.0 + 1.0)
        np.testing.assert_allclose(
            poly.poly1d(np.array([0.0, 1.0, 2.0]), coeffs),
            np.array([1.5, -0.25, -1.5]),
        )

    def test_poly1d_respects_start_index(self):
        coeffs = np.array([2.0, 3.0])
        x = np.array([1.0, 2.0, 4.0])

        np.testing.assert_allclose(
            poly.poly1d(x, coeffs, start_i=1),
            2.0 * x + 3.0 * x**2,
        )

    def test_poly2d_evaluates_scalar_and_vector_inputs(self):
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])

        assert poly.poly2d(2.0, 5.0, coeffs) == pytest.approx(1.0 + 10.0 + 6.0 + 40.0)
        np.testing.assert_allclose(
            poly.poly2d(np.array([1.0, 2.0]), np.array([3.0, 4.0]), coeffs),
            np.array([22.0, 47.0]),
        )

    def test_poly2d_respects_start_indices(self):
        coeffs = np.array([[2.0, -1.0], [0.5, 4.0]])
        x = np.array([2.0, 3.0])
        y = np.array([5.0, 7.0])
        expected = 2.0 * x * y - x * y**2 + 0.5 * x**2 * y + 4.0 * x**2 * y**2

        np.testing.assert_allclose(
            poly.poly2d(x, y, coeffs, start_i=1, start_j=1),
            expected,
        )


class TestPolyDerivatives:
    def test_poly1d_derivative_drops_constant_term_at_zero_start(self):
        coeffs = np.array([3.0, -4.0, 5.0])
        derived_coeffs, start_i = poly.poly1d_deriv(coeffs, start_i=0)

        assert start_i == 0
        np.testing.assert_allclose(derived_coeffs, np.array([-4.0, 10.0]))

    def test_poly1d_derivative_preserves_offset_start(self):
        coeffs = np.array([2.0, -3.0])
        derived_coeffs, start_i = poly.poly1d_deriv(coeffs, start_i=2)

        assert start_i == 1
        np.testing.assert_allclose(derived_coeffs, np.array([4.0, -9.0]))

    def test_poly2d_derivative_matches_analytic_x_and_y_derivatives(self):
        coeffs = np.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 0.5]])
        x = np.array([1.5, 2.0])
        y = np.array([0.5, 1.0])

        coeffs_x, start_i_x, start_j_x = poly.poly2d_deriv(
            coeffs,
            start_i=1,
            start_j=0,
            wrt="x",
        )
        coeffs_y, start_i_y, start_j_y = poly.poly2d_deriv(
            coeffs,
            start_i=1,
            start_j=0,
            wrt="y",
        )

        expected_dx = (
            1.0 - 2.0 * y + 6.0 * x + 8.0 * x * y - 3.0 * x**2 + 1.5 * x**2 * y
        )
        expected_dy = -2.0 * x + 4.0 * x**2 + 0.5 * x**3

        np.testing.assert_allclose(
            poly.poly2d(x, y, coeffs_x, start_i=start_i_x, start_j=start_j_x),
            expected_dx,
        )
        np.testing.assert_allclose(
            poly.poly2d(x, y, coeffs_y, start_i=start_i_y, start_j=start_j_y),
            expected_dy,
        )

    def test_poly2d_derivative_rejects_invalid_variable(self):
        with pytest.raises(ValueError, match="wrt must be 'x' or 'y'"):
            poly.poly2d_deriv(np.ones((2, 2)), wrt="z")
