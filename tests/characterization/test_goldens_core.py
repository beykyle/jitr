"""The Lagrange core reproduces the jitr v3 engine to round-off.

The goldens (see ``generate.py``) hold the v3 inputs and outputs; here every
partial wave is solved directly with :func:`jitr.rmatrix.solve` on the same
inputs. Physics conventions that must agree exactly: the flux-normalized S,
semi-relativistic kinematics (only k and mu enter), per-channel k/mu/eta in
coupled channels, the incoming-weight contraction of the Bloch source, and
the DWBA overlap of distorted waves.
"""

from pathlib import Path

import numpy as np
import pytest

from jitr import rmatrix
from jitr.optical_potentials.dispersion import DispersionSolver
from jitr.utils.constants import HBARC
from jitr.utils.free_solutions import coulomb_hankel_table

DATA = Path(__file__).with_name("data")


def load(name):
    return np.load(DATA / f"{name}.npz")


def hankel(k, eta, a, l):
    return np.array(coulomb_hankel_table(k * a, eta, l)[:])[:, l]


def solve1(mesh, l, k, eta, mu, v, weights=None):
    """One open channel with local potential v (MeV)."""
    return rmatrix.solve(
        mesh,
        np.array([l]),
        np.array([k**2]),
        np.array([[2 * mu / HBARC**2]]),
        np.diag(v)[None, None],
        hankel(k, eta, mesh.a, l)[:, None],
        np.zeros(1),
        np.ones(1, bool),
        weights,
    )


def l_dot_sigma(l, j):
    """``l . sigma`` for j = l + 1/2 (j=0) and j = l - 1/2 (j=1)."""
    return l if j == 0 else -(l + 1)


def partial_waves(lmax):
    return [(l, j) for l in range(lmax + 1) for j in (0, 1) if not (l == 0 and j == 1)]


def test_mesh_and_transforms():
    d = load("transforms_and_grid")
    m = rmatrix.mesh(int(d["nbasis"]), float(d["radius"]))
    np.testing.assert_allclose(m.x, d["x"], rtol=1e-14)
    np.testing.assert_allclose(m.w, d["w"], rtol=1e-14)
    np.testing.assert_allclose(m.F, d["F"].real, rtol=1e-12)
    np.testing.assert_allclose(m.b, d["b"], rtol=1e-12)
    k = d["kgrid"]
    np.testing.assert_allclose(
        rmatrix.fourier_bessel(m, 0, d["f"], k), d["fb0"], rtol=1e-12
    )
    np.testing.assert_allclose(
        rmatrix.fourier_bessel(m, 2, d["f"], k), d["fb2"], rtol=1e-12
    )
    np.testing.assert_allclose(
        rmatrix.fourier_bessel2(m, 1, d["g"], k), d["dfb1"], rtol=1e-12
    )


@pytest.mark.parametrize("name", ["elastic_p48ca", "elastic_alpha44ca"])
def test_elastic_smatrix(name):
    d = load(name)
    m = rmatrix.mesh(int(d["nbasis"]), float(d["radius_fm"]))
    np.testing.assert_allclose(m.r, d["r"], rtol=1e-14)
    splus, sminus = np.atleast_2d(d["splus"]), np.atleast_2d(d["sminus"])
    central = np.atleast_2d(d["central"])
    coulomb = np.atleast_2d(d["coulomb"])
    spin_orbit = np.atleast_2d(d["spin_orbit"]) if "spin_orbit" in d else 0 * central
    for e in range(len(d["kin_k"])):
        k, mu, eta = d["kin_k"][e], d["kin_mu"][e], d["kin_eta"][e]
        for l, j in partial_waves(int(d["lmax"])):
            v = central[e] + coulomb[e] + l_dot_sigma(l, j) * spin_orbit[e]
            S = solve1(m, l, k, eta, mu, v)[0, 0]
            ref = splus[e, l] if j == 0 else sminus[e, l]
            np.testing.assert_allclose(S, ref, rtol=1e-9, atol=1e-12)


def test_coupled_wavefunction_three_channels():
    """Different k, mu and eta per channel with general incoming weights."""
    d = load("wavefunction_3ch")
    a, l = float(d["radius_fm"]), int(d["l"])
    k, mu, eta = d["k"], d["mu"], d["eta"]
    m = rmatrix.mesh(int(d["nbasis"]), a)
    H = np.array([hankel(k[i], eta[i], a, l) for i in range(3)]).T
    M = 2 * np.sqrt(np.outer(mu, mu)) / HBARC**2
    V = np.array([[np.diag(d["V"][i, j]) for j in range(3)] for i in range(3)])
    # v3 weights are raw (unit H- amplitude); flux weights carry sqrt(v_c / v_0)
    v = k / mu
    w = d["weights_raw"] * np.sqrt(v / v[0])
    S, c = rmatrix.solve(
        m, np.full(3, l), k**2, M, V, H, np.zeros(3), np.ones(3, bool), w[:, None]
    )
    np.testing.assert_allclose(S, d["S"], atol=1e-11)
    # v3 u_c = sqrt(k_0 mu_c / mu_0) * u~_c
    scale = np.sqrt(k[0] * mu / mu[0])[:, None]
    u_nodes = scale * np.asarray(c[0]) / np.sqrt(a * m.w)
    np.testing.assert_allclose(u_nodes, d["u_nodes"], atol=1e-11)
    u_a = scale[:, 0] * (np.asarray(c[0]) @ m.b) / np.sqrt(a)
    np.testing.assert_allclose(u_a, d["u_a"], atol=1e-11)


def test_pn_dwba_and_coupled_channels():
    d = load("pn_48ca")
    m = rmatrix.mesh(int(d["nbasis"]), float(d["radius_fm"]))
    kp, mup, etap = d["p_kin_k"][0], d["p_kin_mu"][0], d["p_kin_eta"][0]
    kn, mun = d["n_kin_k"][0], d["n_kin_mu"][0]
    f = d["isovector_factor"]
    U1c = -(d["U_n_central"] - d["U_p_central"]) * f
    U1so = -(d["U_n_spin_orbit"] - d["U_p_spin_orbit"]) * f
    M = 2 * np.sqrt(np.outer([mup, mun], [mup, mun])) / HBARC**2
    for l, j in partial_waves(int(d["lmax"])):
        lds = l_dot_sigma(l, j)
        Up = d["U_p_central"] + d["U_p_coulomb"] + lds * d["U_p_spin_orbit"]
        Un = d["U_n_central"] + lds * d["U_n_spin_orbit"]
        U1 = U1c + lds * U1so
        # DWBA: two single-channel distorted waves, unit flux, not conjugated
        Sp, cp = solve1(m, l, kp, etap, mup, Up, np.ones((1, 1)))
        Sn, cn = solve1(m, l, kn, 0.0, mun, Un, np.ones((1, 1)))
        T = np.sqrt(kp * kn) * np.sum(cp[0, 0] * U1 * cn[0, 0])
        np.testing.assert_allclose(Sp[0, 0], d["Sp"][l, j], rtol=1e-9)
        np.testing.assert_allclose(Sn[0, 0], d["Sn"][l, j], rtol=1e-9)
        np.testing.assert_allclose(T, d["Tpn"][l, j], rtol=1e-9)
        # coupled channels: p and n with different k, mu, eta in one block
        V = np.array([[np.diag(Up), np.diag(U1)], [np.diag(U1), np.diag(Un)]])
        H = np.array([hankel(kp, etap, m.a, l), hankel(kn, 0.0, m.a, l)]).T
        S = rmatrix.solve(
            m,
            np.array([l, l]),
            np.array([kp, kn]) ** 2,
            M,
            V,
            H,
            np.zeros(2),
            np.ones(2, bool),
        )
        np.testing.assert_allclose(S, d["cc_S"][l, j], rtol=1e-9, atol=1e-13)


def test_dom_dispersion():
    """Pins the dispersion kernel through the numba removal."""
    from jitr.optical_potentials import dom

    d = load("dom_dispersion")
    r = d["r"]

    def W(r, x):
        E2 = x * x
        a = 0.30 + 0.30 * E2 * E2 / (E2 * E2 + 12.0**4)
        return dom.Ws_depth(x, 16.2, 12.5, 0.0214) / np.cosh((r - 4.55) / (2 * a)) ** 2

    for E in (-12.3, 5.7, 41.1):
        solver = DispersionSolver(r, E)
        W_grid = W(r[:, None], solver.x_quad[None, :])
        np.testing.assert_allclose(solver(W_grid, W(r, E)), d[f"dV_{E}"], rtol=1e-12)
