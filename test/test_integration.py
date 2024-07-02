import jitr
from numba import njit
import numpy as np

E = 32
sys = jitr.ProjectileTargetSystem(
    reduced_mass=np.array([939.0]),
    channel_radii=np.array([np.pi]),
    l=np.array([0]),
)
ch = sys.build_channels(E)
a = ch[0].domain[1]
solver = jitr.LagrangeRMatrixSolver(30, 1, sys, ecom=E)


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
    quadrature = solver.integrate_local(V, a, (2, 3))
    np.testing.assert_almost_equal(quadrature, analytic)


def test_nonlocal_integration():
    analytic = 18.298993686404124
    quadrature = solver.double_integrate_nonlocal(
        Vnl_asym, a, is_symmetric=False, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic)
    quadrature = solver.double_integrate_nonlocal(
        Vnl_asym, a, is_symmetric=False, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic)

    analytic = 48.58040025635857
    quadrature = solver.double_integrate_nonlocal(
        Vnl_sym, 2 * np.pi, is_symmetric=False, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic, decimal=3)

    quadrature = solver.double_integrate_nonlocal(
        Vnl_sym, 2 * np.pi, is_symmetric=True, args=(1.0 / 16, 2)
    )
    np.testing.assert_almost_equal(quadrature, analytic, decimal=3)

    lower = solver.weight_matrix[solver.lower_mask] * Vnl_sym(
        solver.Xn[solver.lower_mask] * 2 * np.pi,
        solver.Xm[solver.lower_mask] * 2 * np.pi,
        1 / 16,
        2,
    )
    upper = solver.weight_matrix[solver.upper_mask] * Vnl_sym(
        solver.Xn[solver.upper_mask] * 2 * np.pi,
        solver.Xm[solver.upper_mask] * 2 * np.pi,
        1 / 16,
        2,
    )
    full = solver.weight_matrix * Vnl_sym(
        solver.Xn * 2 * np.pi, solver.Xm * 2 * np.pi, 1 / 16, 2
    )
    full_sum = np.sum(full) * 4 * np.pi**2
    np.testing.assert_allclose(
        np.diag(solver.weight_matrix) - solver.kernel.weights**2, 0, atol=1e-7
    )
    x = np.zeros((30, 30))
    x[solver.lower_mask] = lower
    x[solver.upper_mask] = upper
    np.testing.assert_allclose(x - full, 0.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(full), np.sum(lower) + np.sum(upper))
