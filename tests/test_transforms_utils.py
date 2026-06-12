"""jitr.utils.transforms vs the Phase 1 transforms golden + analytic oracles.

Successor to the deprecated-quadrature half of ``test_integration.py``
(deleted with ``jitr.quadrature``). No lax dependency: these are plain
quadrature reductions on the shared Gauss-Legendre mesh.
"""

import numpy as np

from jitr.utils.transforms import (
    double_fourier_bessel_transform,
    fourier_bessel_transform,
    integrate_local,
    laguerre_mesh,
    legendre_mesh,
)

from .characterization._cases import (
    TRANSFORMS_KGRID,
    TRANSFORMS_NBASIS,
    TRANSFORMS_RADIUS,
    load_golden,
)


def _mesh():
    return legendre_mesh(TRANSFORMS_NBASIS, TRANSFORMS_RADIUS)


def test_mesh_matches_golden_grid():
    radii, _ = _mesh()
    golden = load_golden("transforms_fb")
    np.testing.assert_allclose(radii, golden["rgrid"], rtol=1e-13)


def test_fb_transforms_match_golden():
    radii, weights = _mesh()
    golden = load_golden("transforms_fb")
    np.testing.assert_allclose(
        fourier_bessel_transform(0, radii, TRANSFORMS_KGRID, radii, weights),
        golden["fb_l0_linear"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        fourier_bessel_transform(
            1, np.exp(-0.3 * radii**2), TRANSFORMS_KGRID, radii, weights
        ),
        golden["fb_l1_gaussian"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        double_fourier_bessel_transform(
            0, np.outer(radii, radii), TRANSFORMS_KGRID, radii, weights
        ),
        golden["dfb_l0_separable"],
        rtol=1e-10,  # summation-order round-off near small matrix entries
    )


def test_integrate_local_analytic():
    radii, weights = _mesh()
    # ∫₀^π (2 + 3 r²) dr = 2π + π³
    analytic = 2 * np.pi + np.pi**3
    np.testing.assert_allclose(
        integrate_local(2.0 + 3.0 * radii**2, weights), analytic, rtol=1e-12
    )


def test_fb_transform_analytic_constant():
    # ∫₀^R j_0(kr) r² dr = R³/3 at k=0; (sin x − x cos x)/k³ otherwise
    radii, weights = _mesh()
    k_grid = np.array([0.0, 0.35, 0.9, 1.7])
    transformed = fourier_bessel_transform(
        0, np.ones_like(radii), k_grid, radii, weights
    )
    x = k_grid * TRANSFORMS_RADIUS
    expected = np.empty_like(x)
    expected[0] = TRANSFORMS_RADIUS**3 / 3.0
    expected[1:] = (np.sin(x[1:]) - x[1:] * np.cos(x[1:])) / k_grid[1:] ** 3
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-8)


def test_laguerre_integration_analytic():
    radii, weights = _mesh()
    radii_lag, weights_lag = laguerre_mesh(60, 1.0)
    # ∫₀^∞ r^m e^{-r} dr = m!
    for power, factorial in ((0, 1.0), (1, 1.0), (2, 2.0), (3, 6.0)):
        np.testing.assert_allclose(
            integrate_local(radii_lag**power, weights_lag), factorial, rtol=1e-10
        )
