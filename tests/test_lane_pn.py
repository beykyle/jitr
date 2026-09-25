import numpy as np
import pytest

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


def _potentials(cc, absorptive=True, spin_orbit=True):
    r = cc.radial_grid()
    W = 8.0j if absorptive else 0.0
    so = 1.0 if spin_orbit else 0.0
    return {
        "U_p_coulomb": coulomb_charged_sphere(r, 20, 4.7),
        "U_p_central": (-50.0 - W) * woods_saxon_safe(r, 4.4, 0.65),
        "U_p_spin_orbit": so * _thomas(r, 6.0, 4.0, 0.6),
        "U_n_central": (-46.0 - W) * woods_saxon_safe(r, 4.4, 0.65),
        "U_n_spin_orbit": so * _thomas(r, 5.5, 4.0, 0.6),
    }


def _default_U1(ws, p):
    f = ws.isovector_factor
    return (
        -(p["U_n_central"] - p["U_p_central"]) * f,
        -(p["U_n_spin_orbit"] - p["U_p_spin_orbit"]) * f,
    )


def test_grids_match(setup):
    cc, dwba = setup
    np.testing.assert_allclose(cc.radial_grid(), dwba.radial_grid())


def test_weak_coupling_limit_is_dwba(setup):
    # the Born approximation to the coupled-channels S_np is the DWBA, so
    # CC(eps U1) / eps^2 -> DWBA(U1) with a relative error O(eps^2)
    cc, dwba = setup
    p = _potentials(cc)
    U1_central, U1_spin_orbit = _default_U1(cc, p)
    xs_dwba = dwba.xs(**p)
    for eps in (1e-2, 1e-3):
        xs_cc = (
            cc.xs(**p, U1_central=eps * U1_central, U1_spin_orbit=eps * U1_spin_orbit)
            / eps**2
        )
        np.testing.assert_allclose(xs_cc, xs_dwba, rtol=50 * eps**2)


def test_integrated_xs_matches_angular_integral(setup):
    cc, _ = setup
    p = _potentials(cc)
    _, S = cc.rsmatrix(**p)
    dsdo = cc.xs_from_smatrix(S)
    sigma_angular = 2 * np.pi * np.trapezoid(dsdo * np.sin(ANGLES), ANGLES)
    np.testing.assert_allclose(
        sigma_angular, cc.integrated_xs_from_smatrix(S), rtol=1e-4
    )
    np.testing.assert_allclose(cc.integrated_xs(**p), cc.integrated_xs_from_smatrix(S))


def test_real_potentials_give_unitary_symmetric_S(setup):
    cc, _ = setup
    _, S = cc.rsmatrix(**_potentials(cc, absorptive=False))
    assert np.max(np.abs(S[:, :, 1, 0])) > 1e-3
    for l in range(LMAX + 1):
        for j in range(2 if l > 0 else 1):
            np.testing.assert_allclose(S[l, j].conj().T @ S[l, j], np.eye(2), atol=1e-9)
            np.testing.assert_allclose(S[l, j], S[l, j].T, atol=1e-9)


def test_no_spin_orbit_is_j_independent(setup):
    cc, _ = setup
    R, S = cc.rsmatrix(**_potentials(cc, spin_orbit=False))
    np.testing.assert_allclose(S[1:, 0], S[1:, 1], atol=1e-12)
    np.testing.assert_allclose(R[1:, 0], R[1:, 1], atol=1e-12)
    np.testing.assert_array_equal(S[0, 1], 0)


def test_explicit_default_U1_matches_default(setup):
    cc, _ = setup
    p = _potentials(cc)
    U1_central, U1_spin_orbit = _default_U1(cc, p)
    np.testing.assert_allclose(
        cc.xs(**p, U1_central=U1_central, U1_spin_orbit=U1_spin_orbit),
        cc.xs(**p),
        rtol=1e-12,
    )


def test_neutron_central_required(setup):
    cc, _ = setup
    p = _potentials(cc)
    p.pop("U_n_central")
    with pytest.raises(TypeError, match="U_n_central"):
        cc.rsmatrix(**p)
