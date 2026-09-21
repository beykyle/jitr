"""The interior and exterior wavefunctions must agree at the channel radius.

This pins the Bloch-surface source used to build the interior expansion
coefficients. It was previously wrong for coupled channels: the outgoing term
contracted the S-matrix with the wrong index and dropped the incoming weights,
so multichannel ``wavefunction=True`` solves were off by O(100%).
"""

import numpy as np
import pytest

from jitr import reactions, rmatrix
from jitr.reactions.wavefunction import Wavefunctions
from jitr.utils.kinematics import classical_kinematics

A = 5 * np.pi
NBASIS = 30
L = 1


def _channels(nch, k=None, mu=None, eta=None):
    system = reactions.ProjectileTargetSystem(
        channel_radius=A,
        lmax=L,
        mass_target=44657.0,
        mass_projectile=938.3,
        Ztarget=20,
        Zproj=1,
        coupling=lambda l: np.eye(nch),
    )
    if k is None:
        return system.get_partial_wave_channels(
            *classical_kinematics(
                system.mass_target, system.mass_projectile, 42.1, 20.0
            )
        )
    return system.get_partial_wave_channels(0.0, 0.0, mu, k, eta)


def _potential(solver, channels, nch):
    r = solver.radial_grid(channels.a, channels.k[0])
    V = np.zeros((nch, nch, NBASIS), dtype=np.complex128)
    diagonal = (-40.0 - 3.0j) * np.exp(-r / 4)
    for i in range(nch):
        V[i, i] = diagonal * (1 + 0.1 * i)
    for i in range(nch - 1):
        V[i, i + 1] = V[i + 1, i] = -5.0 * np.exp(-r / 4)
    return V


@pytest.mark.parametrize(
    "nch,weights",
    [
        (1, [1.0]),
        (3, [1.0, 0.0, 0.0]),
        (3, [0.0, 1.0, 0.0]),
        (3, [0.6, 0.8, 0.0]),
    ],
)
def test_interior_matches_exterior_at_boundary(nch, weights):
    solver = rmatrix.Solver(NBASIS)
    channels, asymptotics = _channels(nch)
    ch, asym = channels[L], asymptotics[L]
    weights = np.array(weights)

    _, S, coeffs, uext_prime = solver.solve(
        ch,
        asym,
        local_potential=_potential(solver, ch, nch),
        weights=weights,
        wavefunction=True,
    )
    wavefunctions = Wavefunctions(solver, coeffs, S, uext_prime, ch, weights)
    u_interior = np.array([u(ch.a) for u in wavefunctions.uint()])
    u_exterior = np.array([u(ch.a)[0] for u in wavefunctions.uext()])
    np.testing.assert_allclose(u_interior, u_exterior, atol=1e-10)


def test_interior_matches_exterior_with_different_k_per_channel():
    # proton-like and neutron-like channels, as in Lane (p,n)
    solver = rmatrix.Solver(NBASIS)
    k = np.array([1.27, 1.13])
    mu = np.array([918.96, 920.21])
    eta = np.array([0.5, 0.0])
    channels, asymptotics = _channels(2, k, mu, eta)
    ch, asym = channels[L], asymptotics[L]

    _, S, coeffs, uext_prime = solver.solve(
        ch, asym, local_potential=_potential(solver, ch, 2), wavefunction=True
    )
    wavefunctions = Wavefunctions(solver, coeffs, S, uext_prime, ch)
    u_interior = np.array([u(ch.a) for u in wavefunctions.uint()])
    u_exterior = np.array([u(ch.a)[0] for u in wavefunctions.uext()])
    np.testing.assert_allclose(u_interior, u_exterior, atol=1e-10)
