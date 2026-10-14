"""J^pi blocks, coupling geometry and the vectorized coupled-channels workspace."""

from pathlib import Path

import jax
import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j, wigner_6j

from jitr.reactions import Level, jpi_blocks, rotor, spin_orbit
from jitr.utils.constants import HBARC
from jitr.utils.kinematics import classical_kinematics, level_kinematics
from jitr.xs import coupled

GOLDENS = Path(__file__).parent / "characterization" / "data"


def test_spin_half_on_spin_zero_has_one_channel_per_block():
    lmax = 6
    blocks = jpi_blocks([Level(0, 1, 0.5)], lmax)
    assert blocks.l.shape == (2 * lmax + 1, 1)
    np.testing.assert_array_equal(blocks.parity, (-1) ** blocks.l[:, 0])
    np.testing.assert_array_equal(np.abs(blocks.j[:, 0] - blocks.l[:, 0]), 0.5)
    np.testing.assert_array_equal(blocks.J, blocks.j[:, 0])
    # l.sigma = l for j = l + 1/2 and -(l + 1) for j = l - 1/2
    ls = spin_orbit(blocks, [Level(0, 1, 0.5)])[:, 0, 0]
    l = blocks.l[:, 0]
    np.testing.assert_array_equal(ls, np.where(blocks.j[:, 0] > l, l, -(l + 1)))


def test_lane_blocks_pair_identical_l_and_j():
    p, n = Level(0, 1, 0.5), Level(0, 1, 0.5)
    blocks = jpi_blocks([p, n], 5)
    assert blocks.l.shape == (11, 2)
    np.testing.assert_array_equal(blocks.level, [[0, 1]] * 11)
    np.testing.assert_array_equal(blocks.l[:, 0], blocks.l[:, 1])
    np.testing.assert_array_equal(blocks.j[:, 0], blocks.j[:, 1])


def test_rotor_blocks_are_padded():
    levels = [Level(0), Level(2), Level(4)]
    blocks = jpi_blocks(levels, 20)
    sizes = blocks.active.sum(axis=1)
    assert blocks.l.shape[1] == sizes.max() == 9
    J3 = (blocks.J == 3) & (blocks.parity == -1)
    np.testing.assert_array_equal(blocks.l[J3][0], [3, 1, 3, 5, 1, 3, 5, 7, 0])
    assert not blocks.active[J3][0, -1]


def descouvemont_rotor(l, I, lp, Ip, J, lam):
    """Spin-0 rotor coupling as written for Descouvemont's Examples 3 and 4."""
    v = float(
        wigner_3j(Ip, lam, I, 0, 0, 0)
        * wigner_3j(l, lam, lp, 0, 0, 0)
        * wigner_6j(I, l, J, lp, Ip, lam)
    )
    v *= np.sqrt(
        (2 * l + 1)
        * (2 * lp + 1)
        * (2 * I + 1)
        * (2 * Ip + 1)
        * (2 * lam + 1)
        / (4 * np.pi)
    )
    return v * (-1) ** (abs(l - lp) // 2) * (-1) ** (J + lam)


@pytest.mark.parametrize(
    "levels,J,parity,row0",
    [
        ([Level(0), Level(2)], 30, 1, [0, -0.169867, -0.141104, -0.175534]),
        (
            [Level(0), Level(2), Level(4)],
            3,
            -1,
            [0, -0.143048, -0.145673, -0.194664, 0, 0, 0, 0],
        ),
    ],
)
def test_spin_zero_rotor_matches_descouvemont(levels, J, parity, row0):
    blocks = jpi_blocks(levels, 40, J=[J], parity=parity)
    C = rotor(blocks, levels, 2)[0]
    np.testing.assert_allclose(C[0], row0, atol=5e-7)
    I = [levels[a].I for a in blocks.level[0]]  # noqa: E741
    l = blocks.l[0]
    expected = [
        [descouvemont_rotor(l[c], I[c], l[d], I[d], J, 2) for d in range(len(l))]
        for c in range(len(l))
    ]
    np.testing.assert_allclose(C, expected, atol=1e-12)


def rotor_workspace(levels, J, parity, E=12.0, a=10.0, N=60, lmax=12):
    """A light alpha-like rotor at one energy (open channels only)."""
    mf = 20.736 / 3.0
    Ex = np.array([[0.0, 1.5]])
    k2 = (E - Ex) / mf
    mu = np.full_like(k2, HBARC**2 / (2 * mf))
    ws = coupled.Workspace(a, N, levels, lmax, k2, mu, 0 * k2, J=J, parity=parity)
    r = ws.r
    f = 1 / (1 + np.exp((r - 4.0) / 0.5))
    g = np.exp((r - 4.0) / 0.5) / (0.5 * (1 + np.exp((r - 4.0) / 0.5)) ** 2)
    terms = [
        (np.eye(ws.nch), (-60 - 10j) * f),
        (rotor(ws.blocks, levels, 2), (-60 - 10j) * 0.5 * g),
    ]
    return ws, ws.smatrix(local=terms)[0]


def test_spin_half_rotor_is_recoupled_spin_zero_rotor():
    """A spin-independent interaction leaves the projectile spin a spectator:
    S in the ((l s)j, I)J basis is an orthogonal recoupling of the spin-0
    S-matrices with K = J -+ 1/2, so the eigenvalues coincide."""
    J = 3.5
    ws_half, S_half = rotor_workspace([Level(0, 1, 0.5), Level(2, 1, 0.5)], [J], -1)
    eig_half = np.linalg.eigvals(np.asarray(S_half[0, 0]))
    eig_zero = []
    for K in (J - 0.5, J + 0.5):
        ws0, S0 = rotor_workspace([Level(0), Level(2)], [K], -1)
        n = ws0.blocks.active[0].sum()
        eig_zero.extend(np.linalg.eigvals(np.asarray(S0[0, 0, :n, :n])))
    assert ws_half.blocks.active[0].sum() == len(eig_zero)
    np.testing.assert_allclose(
        np.sort_complex(eig_half), np.sort_complex(eig_zero), atol=1e-10
    )


def test_padding_does_not_change_a_block():
    """The J = 3- block alone (8 channels) equals the same block inside the
    full workspace, where it is padded to 9 channels; the raw np.eye coupling
    must not leak into the padded slot."""
    levels = [Level(0), Level(2), Level(4)]

    def solve(J, parity):
        mf = 20.736 / 3.0
        k2 = (np.array([[20.0]]) - [0.0, 4.44, 14.08]) / mf
        mu = np.full_like(k2, HBARC**2 / (2 * mf))
        ws = coupled.Workspace(10.0, 50, levels, 8, k2, mu, 0 * k2, J=J, parity=parity)
        f = 1 / (1 + np.exp((ws.r - 4.6) / 0.5))
        g = f * (1 - f) / 0.5
        S, _ = ws.smatrix(
            local=[
                (np.eye(ws.nch), (-110 - 20j) * f),
                (rotor(ws.blocks, levels, 2), -64 * g),
            ]
        )
        sel = (ws.blocks.J == 3) & (ws.blocks.parity == -1)
        return np.asarray(S[0][sel][0])

    alone, padded = solve([3], -1), solve(None, None)
    assert alone.shape == (8, 8) and padded.shape == (9, 9)
    np.testing.assert_allclose(padded[:8, :8], alone, atol=1e-12)
    np.testing.assert_array_equal(padded[8], 0)


def test_closed_channel_rows_vanish_below_threshold():
    """Two channels, thresholds 0 and 1 MeV: below threshold the closed row and
    column of S are zero; above it the coupling populates S_01; without
    coupling S_00 is the one-channel result."""
    a, N, mf = 8.0, 30, 1.0
    levels = [Level(0), Level(2)]
    E = np.array([0.25, 0.75, 1.25, 1.75])
    k2 = E[:, None] - [0.0, 1.0]
    mu = np.full_like(k2, HBARC**2 / (2 * mf))
    ws = coupled.Workspace(a, N, levels, 2, k2, mu, 0 * k2, J=[0], parity=1)
    r = ws.r
    g = lambda w: np.exp(-((r / w) ** 2))  # noqa: E731
    diag = [(np.diag([1.0, 0.0]), -6 * g(2.1)), (np.diag([0.0, 1.0]), -4.5 * g(2.6))]
    off = [(np.array([[0.0, 1.0], [1.0, 0.0]]), -1.25 * g(2.3))]
    S, is_open = ws.smatrix(local=diag + off)
    S0, _ = ws.smatrix(local=diag)
    S, S0 = np.asarray(S[:, 0]), np.asarray(S0[:, 0])
    below = E < 1.0
    np.testing.assert_array_equal(S[below, 1, :], 0)
    np.testing.assert_array_equal(S[below, :, 1], 0)
    assert np.all(np.abs(S[~below, 0, 1]) > 1e-8)
    np.testing.assert_array_equal(is_open[:, 0, 1], ~below)
    one = coupled.Workspace(a, N, [Level(0)], 0, k2[:, :1], mu[:, :1], 0 * k2[:, :1])
    S1, _ = one.smatrix(local=[(np.eye(1), -6 * g(2.1))])
    np.testing.assert_allclose(S0[:, 0, 0], np.asarray(S1)[:, 0, 0, 0], atol=1e-12)


def proton_workspace(E, lmax=8):
    kin = classical_kinematics(44657.3, 938.27, np.atleast_1d(E), 20)
    k2, mu, eta = level_kinematics(kin, [0.0])
    return coupled.Workspace(12.0, 40, [Level(0, 1, 0.5)], lmax, k2, mu, eta)


def proton_terms(ws, depth=50.0):
    r = ws.r
    f = 1 / (1 + np.exp((r - 4.5) / 0.65))
    coul = 20 * 1.43996 * np.where(r < 4.5, (3 - (r / 4.5) ** 2) / 9.0, 1 / r)
    so = spin_orbit(ws.blocks, ws.levels)
    return [(np.eye(1), (-depth - 8j) * f + coul), (so, -0.6 * f * (1 - f) / r)]


def test_energy_grid_equals_single_energies():
    E = np.array([10.0, 25.0, 40.0])
    ws = proton_workspace(E)
    S = np.asarray(ws.smatrix(local=proton_terms(ws))[0])
    for e, Ee in enumerate(E):
        ws1 = proton_workspace(Ee)
        np.testing.assert_allclose(
            S[e], ws1.smatrix(local=proton_terms(ws1))[0][0], atol=1e-13
        )


def test_energy_dependent_form():
    """A form with an energy axis is applied energy by energy."""
    E = np.array([10.0, 25.0])
    ws = proton_workspace(E)
    t = proton_terms(ws)
    depths = np.array([45.0, 52.0])
    f = 1 / (1 + np.exp((ws.r - 4.5) / 0.65))
    central = t[0][1] + 50 * f - depths[:, None] * f
    S = np.asarray(ws.smatrix(local=[(np.eye(1), central), t[1]])[0])
    for e in range(2):
        ws1 = proton_workspace(E[e])
        S1 = ws1.smatrix(local=proton_terms(ws1, depths[e]))[0][0]
        np.testing.assert_allclose(S[e], S1, atol=1e-13)


def test_l_dependent_form():
    """An (L, NE, N) form selects v[l] in each channel; identical slices
    reproduce the l-independent form, and a too-short l axis is rejected."""
    ws = proton_workspace(20.0)
    t = proton_terms(ws)
    stacked = np.broadcast_to(t[0][1], (9, 1, ws.r.size))
    S = ws.smatrix(local=[(np.eye(1), stacked), t[1]])[0]
    np.testing.assert_allclose(S, ws.smatrix(local=t)[0], atol=1e-14)
    with pytest.raises(AssertionError):
        ws.smatrix(local=[(np.eye(1), stacked[:5]), t[1]])


def test_gradient_matches_finite_difference():
    ws = proton_workspace(np.array([15.0, 30.0]))

    def absorption(depth):
        S = ws.smatrix(local=proton_terms(ws, depth))[0]
        return (1 - abs(S) ** 2).sum()

    grad = jax.grad(absorption)(50.0)
    h = 1e-4
    fd = (absorption(50.0 + h) - absorption(50.0 - h)) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_wavefunctions_reproduce_v3_three_channel_golden():
    """Three levels with different k, mu and eta in one l = 1 block, general
    incoming weights: interior wavefunctions agree with the v3 engine."""
    d = np.load(GOLDENS / "wavefunction_3ch.npz")
    k, mu, eta = d["k"], d["mu"], d["eta"]
    ws = coupled.Workspace(
        float(d["radius_fm"]), int(d["nbasis"]), [Level(0)] * 3, 1, k**2, mu, eta, J=[1]
    )
    V = d["V"]
    terms = [
        (np.outer(np.eye(3)[i], np.eye(3)[j]), V[i, j])
        for i in range(3)
        for j in range(3)
    ]
    v = k / mu
    w = (d["weights_raw"] * np.sqrt(v / v[0]))[:, None]
    S, u, u_a = ws.wavefunctions(local=terms, weights=w)
    np.testing.assert_allclose(S[0, 0], d["S"], atol=1e-11)
    scale = np.sqrt(k[0] * mu / mu[0])
    np.testing.assert_allclose(scale[:, None] * u[0, 0, 0], d["u_nodes"], atol=1e-11)
    np.testing.assert_allclose(scale * u_a[0, 0, 0], d["u_a"], atol=1e-11)


def test_level_kinematics_reproduces_entrance_channel():
    kin = classical_kinematics(44657.3, 938.27, np.array([10.0, 30.0]), 20)
    k2, mu, eta = level_kinematics(kin, [0.0, 3.0, 50.0])
    np.testing.assert_allclose(k2[:, 0], kin.k**2)
    np.testing.assert_allclose(eta[:, 0], kin.eta)
    np.testing.assert_allclose(mu, np.full((2, 3), kin.mu))
    np.testing.assert_allclose(
        HBARC**2 * (kin.k[:, None] ** 2 - k2) / (2 * mu), [[0, 3, 50]] * 2
    )
    assert np.all(k2[:, 2] < 0)


def test_level_kinematics_with_energy_dependent_mass():
    """Semi-relativistic kinematics on an energy grid has mu per energy."""
    from jitr.reactions import Reaction

    kin = Reaction(target=(48, 20), projectile=(1, 1), process="El").kinematics(
        np.array([10.0, 50.0, 100.0])
    )
    k2, mu, eta = level_kinematics(kin, [0.0, 2.0])
    assert k2.shape == mu.shape == eta.shape == (3, 2)
    np.testing.assert_allclose(mu[:, 1], kin.mu)
    np.testing.assert_allclose(k2[:, 0], kin.k**2)
