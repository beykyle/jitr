from jitr import quadrature
from numba import njit
import numpy as np

channel_radii = np.array([np.pi])


solver_le = quadrature.Kernel(30, basis="Legendre")
solver_la = quadrature.Kernel(30, basis="Laguerre")
a = channel_radii[0]


@njit
def V(x, a, b):
    return a + b * x**2


@njit
def Vnl_sym(x, y, a, b):
    return a * x * y + b * (x - y) ** 2 / (y**2 + x**2)


@njit
def Vnl_asym(x, y, a, b):
    return a * x * y + b * x**2 / (y**2 + 2)


def test_local_integration():
    analytic = 37.2894619874794
    quadrature = solver_le.integrate_local(V, a, (2, 3))
    np.testing.assert_almost_equal(quadrature, analytic)


def test_nonlocal_integration():
    analytic = 18.298993686404124
    quadrature = solver_le.double_integrate_nonlocal(
        Vnl_asym, a, is_symmetric=False, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic)
    quadrature = solver_le.double_integrate_nonlocal(
        Vnl_asym, a, is_symmetric=False, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic)

    analytic = 48.58040025635857
    quadrature = solver_le.double_integrate_nonlocal(
        Vnl_sym, 2 * np.pi, is_symmetric=False, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic, decimal=3)

    quadrature = solver_le.double_integrate_nonlocal(
        Vnl_sym, 2 * np.pi, is_symmetric=True, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic, decimal=3)

    lower = solver_le.weight_matrix[solver_le.lower_mask] * Vnl_sym(
        solver_le.Xn[solver_le.lower_mask] * 2 * np.pi,
        solver_le.Xm[solver_le.lower_mask] * 2 * np.pi,
        1 / 16,
        2,
    )
    upper = solver_le.weight_matrix[solver_le.upper_mask] * Vnl_sym(
        solver_le.Xn[solver_le.upper_mask] * 2 * np.pi,
        solver_le.Xm[solver_le.upper_mask] * 2 * np.pi,
        1 / 16,
        2,
    )
    full = solver_le.weight_matrix * Vnl_sym(
        solver_le.Xn * 2 * np.pi, solver_le.Xm * 2 * np.pi, 1 / 16, 2
    )
    np.testing.assert_allclose(
        np.diag(solver_le.weight_matrix) - solver_le.quadrature.weights**2,
        0,
        atol=1e-7,
    )
    x = np.zeros((30, 30))
    x[solver_le.lower_mask] = lower
    x[solver_le.upper_mask] = upper
    np.testing.assert_allclose(x - full, 0.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(full), np.sum(lower) + np.sum(upper))
