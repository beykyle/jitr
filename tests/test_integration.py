import numpy as np
import pytest
import scipy.special as sc

from jitr import quadrature

CHANNEL_RADIUS = np.pi
KERNEL = quadrature.Kernel(80, basis="Legendre")

# A single Laguerre kernel shared by all Laguerre tests.
LAGUERRE_NBASIS = 60
LAGUERRE_RADIUS = 1.0
LAGUERRE_KERNEL = quadrature.Kernel(LAGUERRE_NBASIS, basis="Laguerre")


def local_polynomial(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * x**2


def nonlocal_symmetric(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x * y + b * (x - y) ** 2 / (y**2 + x**2)


def nonlocal_asymmetric(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x * y + b * x**2 / (y**2 + 2)


def spherical_bessel_constant_transform(k: np.ndarray, radius: float) -> np.ndarray:
    x = k * radius
    transformed = np.empty_like(x, dtype=np.float64)
    zero_mask = np.isclose(x, 0.0)
    transformed[zero_mask] = radius**3 / 3.0
    transformed[~zero_mask] = (
        radius**3
        * (np.sin(x[~zero_mask]) - x[~zero_mask] * np.cos(x[~zero_mask]))
        / x[~zero_mask] ** 3
    )
    return transformed


def spherical_bessel_linear_transform(k: np.ndarray, radius: float) -> np.ndarray:
    x = k * radius
    transformed = np.empty_like(x, dtype=np.float64)
    zero_mask = np.isclose(x, 0.0)
    transformed[zero_mask] = radius**4 / 4.0
    transformed[~zero_mask] = radius**4 * (
        -np.cos(x[~zero_mask]) / x[~zero_mask] ** 2
        + 2.0 * np.sin(x[~zero_mask]) / x[~zero_mask] ** 3
        + 2.0 * (np.cos(x[~zero_mask]) - 1.0) / x[~zero_mask] ** 4
    )
    return transformed


def test_local_integration() -> None:
    analytic = 37.2894619874794
    values = local_polynomial(KERNEL.radial_grid(CHANNEL_RADIUS), 2.0, 3.0)
    quadrature_value = KERNEL.integrate_local(values, CHANNEL_RADIUS)
    np.testing.assert_allclose(quadrature_value, analytic)


def test_nonlocal_integration() -> None:
    analytic = 18.298993686404124
    xn, xm = KERNEL.nonlocal_radial_grids(CHANNEL_RADIUS)
    quadrature_value = KERNEL.double_integrate_nonlocal(
        nonlocal_asymmetric(xn, xm, 1.0 / 16.0, 2.0),
        CHANNEL_RADIUS,
        is_symmetric=False,
    )
    np.testing.assert_allclose(quadrature_value, analytic)

    analytic = 48.58040025635857
    radius = 2.0 * np.pi
    xn, xm = KERNEL.nonlocal_radial_grids(radius)
    values = nonlocal_symmetric(xn, xm, 1.0 / 16.0, 2.0)
    quadrature_value = KERNEL.double_integrate_nonlocal(
        values, radius, is_symmetric=False
    )
    np.testing.assert_allclose(quadrature_value, analytic, atol=1e-3)

    quadrature_value = KERNEL.double_integrate_nonlocal(
        values, radius, is_symmetric=True
    )
    np.testing.assert_allclose(quadrature_value, analytic, atol=1e-3)

    lower = KERNEL.weight_matrix[KERNEL.lower_mask] * values[KERNEL.lower_mask]
    upper = KERNEL.weight_matrix[KERNEL.upper_mask] * values[KERNEL.upper_mask]
    full = KERNEL.weight_matrix * values
    np.testing.assert_allclose(
        np.diag(KERNEL.weight_matrix) - KERNEL.quadrature.weights**2,
        0.0,
        atol=1e-7,
    )
    masked = np.zeros((KERNEL.quadrature.nbasis, KERNEL.quadrature.nbasis))
    masked[KERNEL.lower_mask] = lower
    masked[KERNEL.upper_mask] = upper
    np.testing.assert_allclose(masked - full, 0.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(full), np.sum(lower) + np.sum(upper))


def test_fourier_bessel_transform_constant_function() -> None:
    values = np.ones(KERNEL.quadrature.nbasis)
    k_grid = np.array([0.0, 0.35, 0.9, 1.7])

    transformed = KERNEL.fourier_bessel_transform(0, values, k_grid, CHANNEL_RADIUS)
    expected = spherical_bessel_constant_transform(k_grid, CHANNEL_RADIUS)

    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-8)


def test_fourier_bessel_transform_linear_function() -> None:
    r_grid = KERNEL.radial_grid(CHANNEL_RADIUS)
    k_grid = np.array([0.0, 0.2, 0.75, 1.3])

    transformed = KERNEL.fourier_bessel_transform(0, r_grid, k_grid, CHANNEL_RADIUS)
    expected = spherical_bessel_linear_transform(k_grid, CHANNEL_RADIUS)

    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-8)


def test_double_fourier_bessel_transform_separable_constant_kernel() -> None:
    values = np.ones((KERNEL.quadrature.nbasis, KERNEL.quadrature.nbasis))
    k_grid = np.array([0.0, 0.5, 1.1])

    transformed = KERNEL.double_fourier_bessel_transform(
        0, values, k_grid, CHANNEL_RADIUS
    )
    expected_1d = spherical_bessel_constant_transform(k_grid, CHANNEL_RADIUS)
    expected = (2.0 / np.pi) * np.outer(expected_1d, expected_1d)

    np.testing.assert_allclose(transformed, expected, rtol=2e-6, atol=1e-8)


def test_double_fourier_bessel_transform_separable_linear_kernel() -> None:
    r_grid = KERNEL.radial_grid(CHANNEL_RADIUS)
    values = np.outer(r_grid, r_grid)
    k_grid = np.array([0.0, 0.35, 0.9])

    transformed = KERNEL.double_fourier_bessel_transform(
        0, values, k_grid, CHANNEL_RADIUS
    )
    expected_1d = spherical_bessel_linear_transform(k_grid, CHANNEL_RADIUS)
    expected = (2.0 / np.pi) * np.outer(expected_1d, expected_1d)

    np.testing.assert_allclose(transformed, expected, rtol=3e-6, atol=1e-8)


def test_fourier_bessel_transform_matches_dense_trapezoid() -> None:
    k_grid = np.array([0.15, 0.55, 1.05, 1.6])
    values = np.exp(-0.3 * KERNEL.radial_grid(CHANNEL_RADIUS) ** 2)
    transformed = KERNEL.fourier_bessel_transform(1, values, k_grid, CHANNEL_RADIUS)

    dense_r = np.linspace(0.0, CHANNEL_RADIUS, 20001)
    dense_values = np.exp(-0.3 * dense_r**2)
    trapz_reference = np.array(
        [
            np.trapezoid(
                dense_r**2 * sc.spherical_jn(1, momentum * dense_r) * dense_values,
                dense_r,
            )
            for momentum in k_grid
        ]
    )

    np.testing.assert_allclose(transformed, trapz_reference, rtol=2e-4, atol=1e-7)


def test_matrix_helpers_and_dwba() -> None:
    local_values = np.linspace(1.0, 2.0, KERNEL.quadrature.nbasis)
    np.testing.assert_allclose(KERNEL.matrix_local(local_values), local_values)

    nonlocal_values = np.arange(
        KERNEL.quadrature.nbasis**2, dtype=np.float64
    ).reshape(KERNEL.quadrature.nbasis, KERNEL.quadrature.nbasis)
    expected_matrix = np.sqrt(KERNEL.weight_matrix) * nonlocal_values * CHANNEL_RADIUS
    np.testing.assert_allclose(
        KERNEL.matrix_nonlocal(nonlocal_values, CHANNEL_RADIUS), expected_matrix
    )

    bra = np.linspace(1.0, 2.0, KERNEL.quadrature.nbasis).astype(np.complex128)
    ket = np.linspace(0.5, 1.5, KERNEL.quadrature.nbasis).astype(np.complex128)
    expected_local = np.sum(bra * local_values * ket)
    np.testing.assert_allclose(
        KERNEL.dwba_local(bra, ket, local_values), expected_local
    )

    expected_nonlocal = bra.T @ expected_matrix @ ket
    np.testing.assert_allclose(
        KERNEL.dwba_nonlocal(bra, ket, nonlocal_values, CHANNEL_RADIUS),
        expected_nonlocal,
    )


def test_transforms_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="local values"):
        KERNEL.fourier_bessel_transform(
            0, np.ones((KERNEL.quadrature.nbasis, 1)), np.array([0.0]), CHANNEL_RADIUS
        )

    with pytest.raises(ValueError, match="nonlocal values"):
        KERNEL.double_fourier_bessel_transform(
            0, np.ones(KERNEL.quadrature.nbasis), np.array([0.0]), CHANNEL_RADIUS
        )

    with pytest.raises(ValueError, match="momentum grid"):
        KERNEL.fourier_bessel_transform(
            0,
            np.ones(KERNEL.quadrature.nbasis),
            np.ones((2, 2)),
            CHANNEL_RADIUS,
        )


# ---------------------------------------------------------------------------
# Laguerre quadrature tests
# ---------------------------------------------------------------------------


def test_laguerre_kernel_creation() -> None:
    """LagrangeLaguerreQuadrature is constructed correctly."""
    q = LAGUERRE_KERNEL.quadrature
    assert q.nbasis == LAGUERRE_NBASIS
    assert len(q.abscissa) == LAGUERRE_NBASIS
    assert len(q.weights) == LAGUERRE_NBASIS
    # Abscissa must be strictly positive.
    assert np.all(q.abscissa > 0)
    # Weights must be strictly positive.
    assert np.all(q.weights > 0)


def test_laguerre_integrate_exponentially_weighted() -> None:
    r"""Laguerre `integrate_local` computes :math:`\int_0^\infty f(r)\,e^{-r/R}\,dr`.

    The Gauss-Laguerre rule computes ``radius * sum(f(x_i * radius) * w_i)``,
    which equals ``integral_0^infinity f(r) exp(-r/radius) dr``.
    """
    R = LAGUERRE_RADIUS
    r_grid = LAGUERRE_KERNEL.radial_grid(R)

    # ∫_0^∞ e^{-r/R} dr = R
    np.testing.assert_allclose(
        LAGUERRE_KERNEL.integrate_local(np.ones_like(r_grid), R), R, rtol=1e-10
    )
    # ∫_0^∞ r e^{-r/R} dr = R²
    np.testing.assert_allclose(
        LAGUERRE_KERNEL.integrate_local(r_grid, R), R**2, rtol=1e-10
    )
    # ∫_0^∞ r² e^{-r/R} dr = 2 R³
    np.testing.assert_allclose(
        LAGUERRE_KERNEL.integrate_local(r_grid**2, R), 2.0 * R**3, rtol=1e-10
    )


def test_laguerre_kinetic_matrix_is_symmetric() -> None:
    """Lagrange-Laguerre kinetic matrix is real and symmetric."""
    T = LAGUERRE_KERNEL.quadrature.kinetic_matrix(1.0, 0)
    T_real = T.real
    np.testing.assert_allclose(T_real, T_real.T, atol=1e-12)
    # Imaginary part should be negligible.
    np.testing.assert_allclose(T.imag, 0.0, atol=1e-12)


def test_laguerre_qho_eigenvalues() -> None:
    r"""Laguerre mesh reproduces the 3D harmonic-oscillator (l=0) eigenvalues.

    The radial Schrödinger equation for the 3D isotropic QHO with l=0
    (natural units, :math:`\hbar = m = \omega = 1`) is

    .. math::

        \left[-\frac{d^2}{dr^2} + r^2\right] u(r) = 2E\, u(r)

    The analytic eigenvalues are :math:`2E_n = 4n + 3` for
    :math:`n = 0, 1, 2, \ldots`.  The Lagrange-Laguerre matrix approximation
    of this eigenvalue problem should reproduce the lowest eigenvalues to
    better than 1 %.
    """
    q = LAGUERRE_KERNEL.quadrature
    x = q.abscissa  # dimensionless radial coordinates for a=1

    # Kinetic matrix: matrix elements of -d²/dr², a=1, l=0.
    T = q.kinetic_matrix(1.0, 0).real

    # Potential matrix (diagonal): V(r) = r²
    V = np.diag(x**2)

    # Solve the eigenvalue problem.
    eigenvalues = np.sort(np.linalg.eigvalsh(T + V))

    # Analytic eigenvalues: 2E_n = 4n + 3 for n = 0, 1, 2, 3, 4
    analytic = np.array([4 * n + 3 for n in range(5)], dtype=float)

    np.testing.assert_allclose(eigenvalues[:5], analytic, rtol=0.01)
