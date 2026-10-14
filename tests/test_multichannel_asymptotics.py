"""Multichannel R-matrix solves with a different k, mu and eta in each channel.

A proton-like channel (with Coulomb) and a neutron-like channel share one physical
channel radius in fm but have different wavenumbers, reduced masses and Sommerfeld
parameters, as in the Lane (p,n) isobaric-analog problem.
"""

import numpy as np
import pytest

from jitr import reactions, rmatrix
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_safe,
)
from jitr.utils.constants import ALPHA, HBARC

K = np.array([1.27, 1.13])
MU = np.array([918.96, 920.21])
ZZ = 20.0
# the proton Sommerfeld parameter must match the interior Coulomb potential
ETA = np.array([ALPHA * ZZ * MU[0] / (HBARC * K[0]), 0.0])
R_WS, A_WS, R_C = 4.4, 0.65, 4.7


def _system(channel_radius_fm, k, mu, eta, lmax):
    nch = np.size(k)
    system = reactions.ProjectileTargetSystem(
        channel_radius=channel_radius_fm * np.atleast_1d(k)[0],
        lmax=lmax,
        mass_target=44657.0,
        mass_projectile=938.3,
        coupling=lambda l: np.eye(nch),
    )
    k, mu, eta = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (k, mu, eta))
    return system.get_partial_wave_channels(0.0, 0.0, mu, k, eta)


def _potential(r, coupling, absorptive):
    W = 8.0j if absorptive else 0.0
    V_p = (-50.0 - W) * woods_saxon_safe(r, R_WS, A_WS) + coulomb_charged_sphere(
        r, ZZ, R_C
    )
    V_n = (-46.0 - W) * woods_saxon_safe(r, R_WS, A_WS)
    V_pn = coupling * woods_saxon_safe(r, R_WS, A_WS)
    return np.array([[V_p, V_pn], [V_pn, V_n]], dtype=np.complex128)


def _coupled_S(solver, channel_radius_fm, l, coupling, absorptive=True):
    channels, asymptotics = _system(channel_radius_fm, K, MU, ETA, l)
    ch = channels[l]
    r = solver.radial_grid(ch.a, ch.k[0])
    _, S, _ = solver.solve(ch, asymptotics[l], _potential(r, coupling, absorptive))
    return S


@pytest.mark.parametrize("l", [0, 1, 4])
def test_uncoupled_channels_match_single_channel_solves(l):
    solver = rmatrix.Solver(40)
    S = _coupled_S(solver, 12.0, l, coupling=0.0)

    np.testing.assert_allclose(S[0, 1], 0, atol=1e-12)
    np.testing.assert_allclose(S[1, 0], 0, atol=1e-12)

    for i in range(2):
        channels, asymptotics = _system(12.0, K[i], MU[i], ETA[i], l)
        ch = channels[l]
        r = solver.radial_grid(ch.a, ch.k[0])
        V = _potential(r, 0.0, absorptive=True)[i, i]
        _, S_single, _ = solver.solve(ch, asymptotics[l], V)
        np.testing.assert_allclose(S[i, i], S_single[0, 0], rtol=1e-10)


@pytest.mark.parametrize("l", [0, 1, 4])
def test_real_coupled_potential_gives_unitary_symmetric_S(l):
    solver = rmatrix.Solver(40)
    S = _coupled_S(solver, 12.0, l, coupling=-3.0, absorptive=False)
    assert abs(S[1, 0]) > 1e-3
    np.testing.assert_allclose(S.conj().T @ S, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(S, S.T, atol=1e-10)


@pytest.mark.parametrize("l", [0, 3])
def test_coupled_S_independent_of_channel_radius(l):
    solver = rmatrix.Solver(60)
    S_a = _coupled_S(solver, 12.0, l, coupling=-3.0)
    S_b = _coupled_S(solver, 15.0, l, coupling=-3.0)
    # a single-channel proton solve varies at the same ~1e-5 level between these
    # radii (Woods-Saxon tail); before the multichannel boundary fix this was ~0.3
    np.testing.assert_allclose(S_a, S_b, atol=5e-5)
