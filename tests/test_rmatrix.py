"""The Lagrange-mesh R-matrix core: mesh, quadrature, and single-block solves.

Each test checks the solver against something independent of it: exact
integrals of the basis, a direct integration of the radial equation, exact
symmetries of the S-matrix, or analytic eigenvalues.
"""

import numpy as np
import pytest
from mpmath import whitw
from scipy.integrate import quad, solve_ivp

from jitr import rmatrix
from jitr.utils.constants import HBARC
from jitr.utils.free_solutions import (
    coulomb_hankel_table,
    whittaker_log_derivative,
)


def hankel(k, eta, a, l):
    """``(H+, H-, H+', H-')`` of order ``l`` at ``rho = k a``."""
    return np.array(coulomb_hankel_table(k * a, eta, l)[:])[:, l]


def solve_single(mesh, l, k, eta, mass_factor, v):
    """S-matrix of one open channel with local potential ``v(r)`` in MeV, where
    ``mass_factor = 2 mu / (hbar c)^2``."""
    return rmatrix.solve(
        mesh,
        np.array([l]),
        np.array([k**2]),
        np.array([[mass_factor]]),
        np.diag(v)[None, None],
        hankel(k, eta, mesh.a, l)[:, None],
        np.zeros(1),
        np.ones(1, bool),
    )[0, 0]


def test_legendre_kinetic_matrix_is_exact_integral():
    """F/a^2 is <f_n| -d^2/dr^2 + delta(r - a) d/dr |f_m> for the x-regularized
    Lagrange-Legendre functions on [0, a], integrated exactly."""
    N, a = 6, 3.0
    m = rmatrix.mesh(N, a)

    def f(n, r):
        x = r / a
        P = np.polynomial.legendre.Legendre.basis(N)(2 * x - 1)
        xn = m.x[n]
        return (
            (-1.0) ** (N - n - 1)
            * np.sqrt((1 - xn) / xn)
            * P
            * x
            / (x - xn)
            / np.sqrt(a)
        )

    def df(n, r, h=1e-4):
        return (f(n, r + h) - f(n, r - h)) / (2 * h)

    def integrand(r, n, k):
        return df(n, r) * df(k, r)

    for n, k in ((0, 0), (1, 4), (3, 3), (5, 2)):
        # integrate by parts: <f_n|-f_k''> + f_n(a) f_k'(a) = <f_n'|f_k'>
        value, _ = quad(integrand, 0, a, args=(n, k), limit=200)
        assert np.isclose(m.F[n, k] / a**2, value, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        m.b / np.sqrt(a), [f(n, a) for n in range(N)], rtol=1e-12
    )


def test_laguerre_matrices_are_exact_integrals():
    """The Laguerre kinetic and overlap matrices are exact integrals over the
    regularized basis (Baye 2015, Eq. 3.70), not just Gauss approximations."""
    N = 6
    r, _, F, overlap = rmatrix.laguerre(N, 1.0)
    L = np.polynomial.laguerre.Laguerre.basis(N)

    def f(n, s):
        return (-1) ** (n + 1) / np.sqrt(r[n]) * L(s) / (s - r[n]) * s * np.exp(-s / 2)

    def minus_f2(n, s, h=2e-3):
        return -(f(n, s + h) - 2 * f(n, s) + f(n, s - h)) / h**2

    def overlap_integrand(s, n, m):
        return f(n, s) * f(m, s)

    def kinetic_integrand(s, n, m):
        return f(n, s) * minus_f2(m, s)

    for n, m in ((0, 0), (0, 1), (1, 4), (3, 3), (5, 2)):
        kw = dict(args=(n, m), limit=800, points=[r[m]])
        o, _ = quad(overlap_integrand, 0, 80, **kw)
        t, _ = quad(kinetic_integrand, 0, 80, **kw)
        assert np.isclose(overlap[n, m], o, rtol=1e-6, atol=1e-8)
        assert np.isclose(F[n, m], t, rtol=1e-4, atol=1e-6)


def test_laguerre_harmonic_oscillator():
    """-u'' + r^2 u = 2E u has 2E = 4n + 3 for l = 0."""
    r, _, F, _ = rmatrix.laguerre(60, 0.1)
    eigenvalues = np.sort(np.linalg.eigvalsh(F + np.diag(r**2)))
    np.testing.assert_allclose(eigenvalues[:6], 4 * np.arange(6) + 3, rtol=1e-6)


def test_quadrature_integrates_polynomials_exactly():
    m = rmatrix.mesh(20, 2.0)
    r = m.r
    np.testing.assert_allclose(rmatrix.integrate(m, r**5), 2.0**6 / 6, rtol=1e-12)
    np.testing.assert_allclose(
        rmatrix.integrate2(m, np.outer(r, r**2)),
        (2.0**2 / 2) * (2.0**3 / 3),
        rtol=1e-12,
    )


@pytest.mark.parametrize("l", [0, 1, 4])
def test_single_channel_matches_direct_integration(l):
    """S from the R-matrix equals S from integrating the radial equation
    outward and matching to Coulomb functions at the channel radius."""
    mu, z12, a = 938.27 * 44657.3 / (938.27 + 44657.3), 20, 14.0
    Elab = 30.0
    k = np.sqrt(2 * mu * Elab * 44657.3 / (938.27 + 44657.3)) / HBARC
    eta = z12 * 1.43996 * mu / (HBARC**2 * k)

    def v(r):
        ws = 1 / (1 + np.exp((r - 4.5) / 0.65))
        coul = np.where(
            r < 4.5, z12 * 1.43996 * (3 - (r / 4.5) ** 2) / 9.0, z12 * 1.43996 / r
        )
        return (-50 - 10j) * ws + coul

    def rhs(r, y):
        return [y[1], (l * (l + 1) / r**2 + 2 * mu / HBARC**2 * v(r) - k**2) * y[0]]

    r0 = 1e-3
    sol = solve_ivp(
        rhs,
        (r0, a),
        np.array([r0 ** (l + 1), (l + 1) * r0**l], dtype=complex),
        rtol=1e-11,
        atol=1e-14,
    )
    u, du = sol.y[0, -1], sol.y[1, -1]
    Hp, Hm, Hpp, Hmp = hankel(k, eta, a, l)
    # u ~ H- - S H+, so S = (u' H- - u k H-') / (u' H+ - u k H+')
    S_ode = (du * Hm - k * u * Hmp) / (du * Hp - k * u * Hpp)

    # the charged-sphere kink at r = 4.5 fm limits Gauss convergence to ~1e-6
    m = rmatrix.mesh(100, a)
    S = solve_single(m, l, k, eta, 2 * mu / HBARC**2, v(m.r))
    np.testing.assert_allclose(S, S_ode, rtol=1e-5)


def _ws(r):
    return 1 / (1 + np.exp((r - 4.4) / 0.65))


def _coulomb(r, eta, k, mu):
    # point Coulomb consistent with the channel's Sommerfeld parameter
    return eta * k * HBARC**2 / mu / r


def _coupled(a, nch, k, mu, eta, coupling, N=40):
    m = rmatrix.mesh(N, a)
    r = m.r
    V = np.zeros((nch, nch, N, N), complex)
    for i in range(nch):
        V[i, i] = np.diag(
            -40.0 * (1 + 0.1 * i) * _ws(r) + _coulomb(r, eta[i], k[i], mu[i])
        )
    for i in range(nch - 1):
        V[i, i + 1] = V[i + 1, i] = np.diag(coupling * _ws(r))
    H = np.array([hankel(k[i], eta[i], a, 1) for i in range(nch)]).T
    M = 2 * np.sqrt(np.outer(mu, mu)) / HBARC**2
    return rmatrix.solve(
        m, np.ones(nch, int), k**2, M, V, H, np.zeros(nch), np.ones(nch, bool)
    )


K = np.array([1.27, 1.13, 0.95])
MU = np.array([918.96, 920.21, 1870.0])
ETA = np.array([0.5, 0.0, 0.8])


def test_uncoupled_channels_equal_single_channel_solves():
    S = _coupled(14.0, 3, K, MU, ETA, coupling=0.0)
    m = rmatrix.mesh(40, 14.0)
    for i in range(3):
        v = -40.0 * (1 + 0.1 * i) * _ws(m.r) + _coulomb(m.r, ETA[i], K[i], MU[i])
        Si = solve_single(m, 1, K[i], ETA[i], 2 * MU[i] / HBARC**2, v)
        np.testing.assert_allclose(S[i, i], Si, rtol=1e-12)
    np.testing.assert_allclose(S - np.diag(np.diag(S)), 0, atol=1e-14)


def test_real_coupling_gives_unitary_symmetric_S():
    """Flux normalization: with real V, S is unitary and symmetric even when
    the channels differ in k, mu and eta."""
    S = np.asarray(_coupled(14.0, 3, K, MU, ETA, coupling=-5.0))
    np.testing.assert_allclose(S @ S.conj().T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(S, S.T, atol=1e-10)


def test_S_is_independent_of_channel_radius():
    S12 = np.asarray(_coupled(12.0, 3, K, MU, ETA, coupling=-5.0, N=50))
    S15 = np.asarray(_coupled(15.0, 3, K, MU, ETA, coupling=-5.0, N=60))
    # the Woods-Saxon tail at 12 fm (~3e-4 MeV) bounds the agreement
    np.testing.assert_allclose(S12, S15, atol=2e-5)


def test_closed_channel_drops_out_and_matches_whittaker_tail():
    """A closed channel coupled to an open one: S is nonzero only in the open
    block, and the closed channel's interior wave continues as its Whittaker
    tail W(2 kappa r) beyond the range of the potential."""
    a, N, mf = 20.0, 120, 1.0  # mass factor hbar^2/2mu = 1 MeV fm^2
    m = rmatrix.mesh(N, a)
    r = m.r
    E, threshold, l = 0.6, 1.0, np.array([0, 2])
    k2 = (E - np.array([0.0, threshold])) / mf
    gauss = lambda w: np.exp(-((r / w) ** 2))  # noqa: E731
    V = np.array(
        [
            [np.diag(-6.0 * gauss(2.1)), np.diag(-1.25 * gauss(2.3))],
            [np.diag(-1.25 * gauss(2.3)), np.diag(-4.5 * gauss(2.6))],
        ]
    )
    kappa = np.sqrt(-k2[1])
    Bc = np.array([0.0, whittaker_log_derivative(2, 0.0, 2 * kappa * a)])
    H = np.zeros((4, 2), complex)
    H[:, 0] = hankel(np.sqrt(k2[0]), 0.0, a, 0)
    is_open = np.array([True, False])
    S, c = rmatrix.solve(
        m, l, k2, np.ones((2, 2)) / mf, V, H, Bc, is_open, np.array([[1.0], [0.0]])
    )
    S = np.asarray(S)
    assert abs(S[0, 0]) <= 1 + 1e-12
    np.testing.assert_array_equal(S[1, :], 0)
    np.testing.assert_array_equal(S[:, 1], 0)

    u = np.asarray(c[0, 1]) / np.sqrt(a * m.w)
    tail = np.array([float(whitw(0, 2.5, 2 * kappa * x)) for x in r])
    outside = r > 12.0  # the potentials are negligible here
    ratio = u[outside] / tail[outside]
    np.testing.assert_allclose(ratio, ratio[-1], rtol=1e-3)


def test_coefficients_reproduce_exterior_wave_at_boundary():
    """The interior expansion evaluated at r = a equals the asymptotic form
    (i/2) k^{-1/2} [w H- - (S w) H+] for coupled channels and general weights."""
    a, N = 14.0, 40
    m = rmatrix.mesh(N, a)
    r = m.r
    V = np.zeros((3, 3, N, N), complex)
    for i in range(3):
        V[i, i] = np.diag((-40.0 - 3j) * (1 + 0.1 * i) * np.exp(-r / 4))
    for i in range(2):
        V[i, i + 1] = V[i + 1, i] = np.diag(-5.0 * np.exp(-r / 4))
    H = np.array([hankel(K[i], ETA[i], a, 1) for i in range(3)]).T
    M = 2 * np.sqrt(np.outer(MU, MU)) / HBARC**2
    w = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8], [0.0, 0.0, 0.3]])
    S, c = rmatrix.solve(
        m, np.ones(3, int), K**2, M, V, H, np.zeros(3), np.ones(3, bool), w
    )
    u_a = np.asarray(c) @ m.b / np.sqrt(a)  # (nw, nch)
    exterior = (0.5j * (H[1][:, None] * w - H[0][:, None] * (np.asarray(S) @ w))).T
    np.testing.assert_allclose(u_a, exterior / np.sqrt(K), atol=1e-10)
