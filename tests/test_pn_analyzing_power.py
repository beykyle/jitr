"""Polarization observables for quasi-elastic (p,n) scattering to the IAS.

The analyzing power of a quasi-elastic (p,n) reaction is the observable that
Gosset, Mayer and Escudie, Phys. Rev. C 14, 878 (1976) used to constrain the
isovector spin-orbit part of the Lane potential. These tests pin the amplitude
decomposition and the sign convention it is built on.
"""

import numpy as np
import pytest
from scipy.special import eval_legendre, lpmv

from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_safe,
)
from jitr.reactions import Reaction
from jitr.rmatrix import Solver
from jitr.xs import lane_pn, quasielastic_pn

ANGLES = np.linspace(1e-3, np.pi - 1e-3, 721)
LMAX = 15
RADIUS = 14.0


def _thomas(r, depth, R, a):
    x = np.exp((r - R) / a)
    return -depth * x / (1 + x) ** 2 / (a * r)


@pytest.fixture(scope="module")
def setup():
    reaction = Reaction((48, 20), (1, 1), (1, 0), (48, 21))
    ke = reaction.kinematics(35.0, relativistic=False)
    kx = reaction.kinematics_exit(ke, 6.67, relativistic=False)
    solver = Solver(35)
    cc = lane_pn.Workspace(reaction, ke, kx, solver, ANGLES, LMAX, RADIUS)
    dwba = quasielastic_pn.Workspace(
        reaction, ke, kx, solver, ANGLES, LMAX, RADIUS, tmatrix_abs_tol=0
    )
    return cc, dwba


def _potentials(cc, spin_orbit=True):
    r = cc.radial_grid()
    so = 1.0 if spin_orbit else 0.0
    return {
        "U_p_coulomb": coulomb_charged_sphere(r, 20, 4.7),
        "U_p_central": (-50.0 - 8.0j) * woods_saxon_safe(r, 4.4, 0.65),
        "U_p_spin_orbit": so * _thomas(r, 6.0, 4.0, 0.6),
        "U_n_central": (-46.0 - 8.0j) * woods_saxon_safe(r, 4.4, 0.65),
        "U_n_spin_orbit": so * _thomas(r, 5.5, 4.0, 0.6),
    }


def _default_U1(ws, p):
    f = ws.isovector_factor
    return (
        -(p["U_n_central"] - p["U_p_central"]) * f,
        -(p["U_n_spin_orbit"] - p["U_p_spin_orbit"]) * f,
    )


def _legendre_amplitudes(prefactor, partial_waves):
    """Rebuild the non-spin-flip and spin-flip amplitudes from Legendre sums.

    This is Eq. (13) of Gosset et al., an implementation independent of the
    Clebsch-Gordan / spherical-harmonic construction in
    ``spin_half_transition_geometry``::

        X = sum_l  c_l [(l + 1) I_l+ + l I_l-] P_l(cos t)
        Y = sum_l  c_l [I_l+ - I_l-] P^1_l(cos t)

    Args:
        prefactor: ``c_l`` including the ``1/sqrt(4 pi)`` that the m-basis
            geometry carries, with shape ``(lmax + 1,)``.
        partial_waves: ``I_lj`` with shape ``(lmax + 1, 2)``, ``j`` indexing
            ``(l + 1/2, l - 1/2)``.
    """
    ls = np.arange(partial_waves.shape[0])[:, np.newaxis]
    costheta = np.cos(ANGLES)[np.newaxis, :]
    plus = partial_waves[:, 0][:, np.newaxis]
    minus = partial_waves[:, 1][:, np.newaxis]
    c = prefactor[:, np.newaxis]
    X = np.sum(c * ((ls + 1) * plus + ls * minus) * eval_legendre(ls, costheta), axis=0)
    Y = np.sum(c * (plus - minus) * lpmv(1, ls, costheta), axis=0)
    return X, Y


def test_dsdo_is_the_summed_amplitude_matrix(setup):
    """The cross section is 1/(2s+1) times the full sum over m, m'.

    ``xs`` and ``xs_from_smatrix`` both route through ``pn_observables`` now, so
    comparing them to it would be tautological; the sum is written out here
    instead.
    """
    cc, dwba = setup
    p = _potentials(cc)

    _, S = cc.rsmatrix(**p)
    f = cc.amplitudes_from_smatrix(S)
    np.testing.assert_allclose(
        cc.observables_from_smatrix(S).dsdo,
        10.0 * 0.5 * np.sum(np.absolute(f) ** 2, axis=(0, 1)),
        rtol=1e-14,
    )

    Tlj, _, _ = dwba.tmatrix(**p)
    t = dwba.amplitudes_from_tmatrix(Tlj)
    np.testing.assert_allclose(
        dwba.observables_from_tmatrix(Tlj).dsdo,
        dwba.xs_factor * 10.0 * np.sum(np.absolute(t) ** 2, axis=(0, 1)),
        rtol=1e-14,
    )


def test_amplitude_symmetries(setup):
    """A spin-1/2 transition on a spin-0 target has only two amplitudes."""
    cc, dwba = setup
    p = _potentials(cc)
    _, S = cc.rsmatrix(**p)
    f = cc.amplitudes_from_smatrix(S)
    atol = 1e-14 * np.max(np.absolute(f))
    np.testing.assert_allclose(f[0, 0], f[1, 1], rtol=0, atol=atol)
    np.testing.assert_allclose(f[0, 1], -f[1, 0], rtol=0, atol=atol)


@pytest.mark.parametrize("workspace", ["cc", "dwba"])
def test_ay_vanishes_without_spin_orbit(setup, workspace):
    """With no spin-orbit anywhere the transition cannot analyze the beam.

    This is the statement below Eq. (3e) of Gosset et al.: with the entrance
    and exit spin-orbit potentials zero, the isovector spin-orbit form factor
    vanishes too and the radial matrix elements stop depending on j, so the
    spin-flip amplitude is identically zero.
    """
    cc, dwba = setup
    p = _potentials(cc, spin_orbit=False)
    zero = np.zeros_like(cc.radial_grid(), dtype=np.complex128)

    if workspace == "cc":
        obs = cc.observables(**p, U1_spin_orbit=zero)
        _, S = cc.rsmatrix(**p, U1_spin_orbit=zero)
        f = cc.amplitudes_from_smatrix(S)
    else:
        obs = dwba.observables(**p, U1_spin_orbit=zero)
        Tlj, _, _ = dwba.tmatrix(**p, U1_spin_orbit=zero)
        f = dwba.amplitudes_from_tmatrix(Tlj)

    non_flip, spin_flip = f[1, 1], f[1, 0]
    assert np.max(np.absolute(spin_flip)) < 1e-12 * np.max(np.absolute(non_flip))
    assert np.max(np.absolute(obs.Ay)) < 1e-12
    assert np.max(np.absolute(obs.Q)) < 1e-12


def test_ay_nonzero_with_spin_orbit(setup):
    """The converse: equal but non-zero p/n spin-orbits still analyze.

    The isovector spin-orbit form factor vanishes here, but the distorted
    waves are still j-dependent, so Ay does not.
    """
    cc, _ = setup
    p = _potentials(cc)
    p["U_n_spin_orbit"] = p["U_p_spin_orbit"]
    zero = np.zeros_like(cc.radial_grid(), dtype=np.complex128)
    assert np.max(np.absolute(cc.observables(**p, U1_spin_orbit=zero).Ay)) > 0.1


def test_amplitudes_match_legendre_partial_wave_sum(setup):
    """Pin the geometry, and the sign of Ay, against an independent sum.

    ``scipy.special.lpmv`` carries the Condon-Shortley phase, which is the
    same convention ``jitr.xs.elastic`` uses for its spin-flip amplitude, so
    this also pins the (p,n) analyzing power to the elastic one.
    """
    cc, dwba = setup
    p = _potentials(cc)
    norm = 1.0 / np.sqrt(4 * np.pi)

    _, S = cc.rsmatrix(**p)
    f = cc.amplitudes_from_smatrix(S)
    prefactor = (
        norm
        * np.sqrt(4 * np.pi)
        / (2j * cc.kinematics_entrance.k)
        * np.exp(1j * cc.sigma_c)
    )
    X, Y = _legendre_amplitudes(prefactor, S[:, :, lane_pn.NEUTRON, lane_pn.PROTON])
    scale = np.max(np.absolute(X))
    np.testing.assert_allclose(f[1, 1], X, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(f[1, 0], Y, rtol=0, atol=1e-12 * scale)

    Tlj, _, _ = dwba.tmatrix(**p)
    t = dwba.amplitudes_from_tmatrix(Tlj)
    prefactor = (
        norm
        * (4 * np.pi) ** 1.5
        / (dwba.kinematics_entrance.k * dwba.kinematics_exit.k)
        * np.exp(1j * dwba.sigma_c)
    )
    X, Y = _legendre_amplitudes(prefactor, Tlj)
    scale = np.max(np.absolute(X))
    np.testing.assert_allclose(t[1, 1], X, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(t[1, 0], Y, rtol=0, atol=1e-12 * scale)


def test_weak_coupling_ay_matches_dwba(setup):
    """As U1 -> 0 the coupled-channels analyzing power reduces to DWBA."""
    cc, dwba = setup
    p = _potentials(cc)
    U1_central, U1_spin_orbit = _default_U1(cc, p)
    eps = 1e-3
    weak = dict(U1_central=eps * U1_central, U1_spin_orbit=eps * U1_spin_orbit)

    # Ay is a ratio of amplitudes, both linear in U1, so it needs no rescaling
    np.testing.assert_allclose(
        cc.observables(**p, **weak).Ay, dwba.observables(**p, **weak).Ay, atol=1e-4
    )
