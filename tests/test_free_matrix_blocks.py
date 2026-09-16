"""The vectorised kinetic matrices equal Baye's closed-form matrix elements, the
Laguerre basis is consistent with its exact integrals, and the uncoupled free-matrix
blocks equal the blocks of the coupled matrix without sharing its memory."""

import numpy as np
import pytest
from scipy.integrate import quad

from jitr.quadrature import Kernel, laguerre
from jitr.rmatrix import Solver
from jitr.utils import block


def _laguerre_element(q, n, m, a, l):
    """Eqs. 3.75-3.77 in Baye (2015), scaled by ``1 / a**2``, plus centrifugal."""
    xn, xm, N = q.abscissa[n - 1], q.abscissa[m - 1], q.nbasis
    correction = (-1) ** (n - m) / (4 * np.sqrt(xn * xm))
    if n == m:
        radial = -(xn**2 - 2 * (2 * N + 1) * xn - 4) / (12 * xn**2)
        return (radial - correction) / a**2 + l * (l + 1) / (a * xn) ** 2
    return (
        (-1) ** (n - m) * (xn + xm) / np.sqrt(xn * xm) / (xn - xm) ** 2 - correction
    ) / a**2


def _legendre_element(q, n, m, a, l):
    """Eqs. 3.128-3.129 in Baye (2015), scaled by ``1 / a**2``, plus centrifugal."""
    xn, xm, N = q.abscissa[n - 1], q.abscissa[m - 1], q.nbasis
    if n == m:
        radial = ((4 * N**2 + 4 * N + 3) * xn * (1 - xn) - 6 * xn + 1) / (
            3 * xn**2 * (1 - xn) ** 2
        )
        return radial / a**2 + l * (l + 1) / (a * xn) ** 2
    return (
        (-1.0) ** (n + m)
        * (
            (N**2 + N + 1.0)
            + (xn + xm - 2 * xn * xm) / (xn - xm) ** 2
            - 1.0 / (1.0 - xn)
            - 1.0 / (1.0 - xm)
        )
        / np.sqrt(xn * xm * (1.0 - xn) * (1.0 - xm))
        / a**2
    )


@pytest.mark.parametrize(
    "basis,element", [("Legendre", _legendre_element), ("Laguerre", _laguerre_element)]
)
def test_kinetic_matrix_matches_closed_form_elements(basis, element):
    for nbasis in (5, 20, 45):
        q = Kernel(nbasis, basis).quadrature
        for a, l in ((1.0, 0), (3.7, 0), (37.3, 7), (120.0, 40)):
            expected = np.array(
                [
                    [element(q, n, m, a, l) for m in range(1, nbasis + 1)]
                    for n in range(1, nbasis + 1)
                ]
            )
            got = q.kinetic_matrix(a, l)
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
            np.testing.assert_allclose(got, got.T, rtol=1e-12, atol=0)


def test_laguerre_matrix_elements_are_exact_integrals():
    """The kinetic and overlap matrices are the exact integrals over the regularised
    Laguerre basis (Eq. 3.70 in Baye), not just their Gauss approximations."""
    q = Kernel(6, "Laguerre").quadrature
    x = q.abscissa

    def f(n, s):
        return laguerre(n, 1.0, s, q).real

    def minus_f_second(n, s, h=2e-3):
        return -(f(n, s + h) - 2 * f(n, s) + f(n, s - h)) / h**2

    def integrate(integrand, n, m):
        value, _ = quad(
            lambda s: integrand(n, m, s), 0, 80, limit=800, points=[x[m - 1]]
        )
        return value

    T = q.kinetic_matrix(1.0, 0).real
    for n, m in ((1, 1), (1, 2), (2, 5), (4, 4), (6, 3)):
        overlap = integrate(lambda n, m, s: f(n, s) * f(m, s), n, m)
        kinetic = integrate(lambda n, m, s: f(n, s) * minus_f_second(m, s), n, m)
        assert np.isclose(q.overlap[n - 1, m - 1], overlap, rtol=1e-6, atol=1e-8)
        assert np.isclose(T[n - 1, m - 1], kinetic, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("basis", ["Legendre", "Laguerre"])
def test_kinetic_matrix_scales_as_inverse_square_of_channel_radius(basis):
    q = Kernel(12, basis).quadrature
    for l in (0, 3):
        reference = q.kinetic_matrix(1.0, l)
        for a in (0.5, 2.0, 25.0):
            np.testing.assert_allclose(
                q.kinetic_matrix(a, l), reference / a**2, rtol=1e-14
            )


def test_uncoupled_free_matrix_blocks_match_coupled_matrix():
    for nbasis in (10, 30):
        solver = Solver(nbasis)
        a = 25.0
        l = np.arange(0, 12)
        E = np.linspace(1.0, 0.7, l.size)
        mu = np.linspace(1.0, 1.1, l.size)
        coupled = solver.free_matrix(a, l, E=E, mu=mu, coupled=True)
        blocks = solver.free_matrix(a, l, E=E, mu=mu, coupled=False)
        assert coupled.shape == (nbasis * l.size, nbasis * l.size)
        assert len(blocks) == l.size
        for i, b in enumerate(blocks):
            expected = block(coupled, (i, i), (nbasis, nbasis))
            np.testing.assert_allclose(b, expected, rtol=1e-12, atol=0)
            assert b.shape == (nbasis, nbasis)
            # a block owns its memory (nbasis**2), not the coupled matrix
            assert b.nbytes == nbasis * nbasis * 16
            assert not np.shares_memory(b, coupled)
        # off-diagonal blocks of the free matrix vanish
        off = coupled.copy()
        for i in range(l.size):
            off[i * nbasis : (i + 1) * nbasis, i * nbasis : (i + 1) * nbasis] = 0
        assert not off.any()


def test_uncoupled_free_matrix_defaults_are_unit_energy_and_mass():
    solver = Solver(8)
    l = np.array([0, 1, 2])
    default = solver.free_matrix(4.0, l, coupled=False)
    explicit = solver.free_matrix(4.0, l, E=np.ones(3), mu=np.ones(3), coupled=False)
    for d, e in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(d, e)
