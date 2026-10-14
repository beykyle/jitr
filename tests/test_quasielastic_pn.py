"""Quasi-elastic (p,n) to the isobaric analog state: DWBA and Lane CC.

The analyzing power of this reaction is what Gosset, Mayer and Escudie,
Phys. Rev. C 14, 878 (1976) used to constrain the isovector spin-orbit Lane
potential; several tests pin the amplitude decomposition it rests on.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_legendre, lpmv

from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_safe,
)
from jitr.reactions import Reaction
from jitr.xs.coupled import Potential
from jitr.xs.quasielastic_pn import NEUTRON, PROTON, Workspace, isovector_factor

GOLDEN = Path(__file__).parent / "characterization" / "data" / "pn_48ca.npz"
REACTION = Reaction((48, 20), (1, 1), (1, 0), (48, 21))
ANGLES = np.linspace(1e-3, np.pi - 1e-3, 361)
LMAX, RADIUS, NBASIS = 15, 14.0, 35


def thomas(r, depth, R, a):
    x = np.exp((r - R) / a)
    return -depth * x / (1 + x) ** 2 / (a * r)


def workspace(E=35.0):
    ke = REACTION.kinematics(E, relativistic=False)
    kx = REACTION.kinematics_exit(ke, 6.67, relativistic=False)
    return Workspace(REACTION, ke, kx, RADIUS, LMAX, NBASIS, angles=ANGLES)


@pytest.fixture(scope="module")
def ws():
    return workspace()


def potentials(ws, absorptive=True, spin_orbit=True):
    r = ws.r
    W = 8.0j if absorptive else 0.0
    so = 1.0 if spin_orbit else 0.0
    Up = Potential(
        central=(-50.0 - W) * woods_saxon_safe(r, 4.4, 0.65),
        spin_orbit=so * thomas(r, 6.0, 4.0, 0.6),
        coulomb=coulomb_charged_sphere(r, 20, 4.7),
    )
    Un = Potential(
        central=(-46.0 - W) * woods_saxon_safe(r, 4.4, 0.65),
        spin_orbit=so * thomas(r, 5.5, 4.0, 0.6),
    )
    return Up, Un


def default_U1(Up, Un, scale=1.0):
    f = scale * isovector_factor(REACTION)
    return Potential(
        central=-(Un.central - Up.central) * f,
        spin_orbit=-(Un.spin_orbit - Up.spin_orbit) * f,
    )


def by_lj(ws, X):
    """Block-indexed X[e, b, ...] as [l, j] (j = l + 1/2, l - 1/2) at e = 0."""
    out = np.zeros((LMAX + 1, 2, *np.shape(X)[2:]), complex)
    out[ws.l, (ws.j < ws.l).astype(int)] = np.asarray(X)[0]
    return out


def test_reproduces_v3_goldens():
    d = np.load(GOLDEN)
    ws = Workspace(
        REACTION,
        REACTION.kinematics(35.0, relativistic=False),
        REACTION.kinematics_exit(
            REACTION.kinematics(35.0, relativistic=False), 6.67, relativistic=False
        ),
        float(d["radius_fm"]),
        int(d["lmax"]),
        int(d["nbasis"]),
        angles=d["angles"],
    )
    Up = Potential(
        central=d["U_p_central"],
        spin_orbit=d["U_p_spin_orbit"],
        coulomb=d["U_p_coulomb"],
    )
    Un = Potential(central=d["U_n_central"], spin_orbit=d["U_n_spin_orbit"])
    T, Sp, Sn = ws.tmatrix(Up, Un)
    np.testing.assert_allclose(by_lj(ws, T), d["Tpn"], atol=1e-12)
    np.testing.assert_allclose(by_lj(ws, Sp), d["Sp"], atol=1e-12)
    np.testing.assert_allclose(by_lj(ws, Sn), d["Sn"], atol=1e-12)
    np.testing.assert_allclose(by_lj(ws, ws.smatrix(Up, Un)), d["cc_S"], atol=1e-12)
    # the DWBA observables go through the Born S_np = -4i sqrt(mu mu') T / ...
    dwba, cc = ws.dwba(Up, Un), ws.cc(Up, Un)
    for name in ("dsdo", "Ay", "Q"):
        np.testing.assert_allclose(
            getattr(dwba, name)[0], d[f"dwba_{name}"], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            getattr(cc, name)[0], d[f"cc_{name}"], rtol=1e-10, atol=1e-12
        )
    np.testing.assert_allclose(cc.rxn_pn[0], d["cc_integrated"], rtol=1e-12)


def test_weak_coupling_limit_is_dwba(ws):
    """CC(eps U1) / eps^2 -> DWBA(U1) with a relative error O(eps^2); the
    analyzing power, a ratio of amplitudes linear in U1, needs no rescaling."""
    Up, Un = potentials(ws)
    dwba = ws.dwba(Up, Un)
    for eps in (1e-2, 1e-3):
        cc = ws.cc(Up, Un, default_U1(Up, Un, eps))
        np.testing.assert_allclose(cc.dsdo / eps**2, dwba.dsdo, rtol=50 * eps**2)
        np.testing.assert_allclose(cc.rxn_pn / eps**2, dwba.rxn_pn, rtol=50 * eps**2)
    np.testing.assert_allclose(cc.Ay, dwba.Ay, atol=1e-4)


def test_integrated_xs_matches_angular_integral(ws):
    Up, Un = potentials(ws)
    xs = ws.cc(Up, Un)
    sigma = 2 * np.pi * np.trapezoid(xs.dsdo[0] * np.sin(ANGLES), ANGLES)
    np.testing.assert_allclose(sigma, xs.rxn_pn[0], rtol=1e-4)


def test_real_potentials_give_unitary_symmetric_S(ws):
    S = np.asarray(ws.smatrix(*potentials(ws, absorptive=False))[0])
    assert np.max(np.abs(S[:, NEUTRON, PROTON])) > 1e-3
    eye = np.eye(2)
    np.testing.assert_allclose(
        np.einsum("bji,bjk->bik", S.conj(), S), np.broadcast_to(eye, S.shape), atol=1e-9
    )
    np.testing.assert_allclose(S, S.transpose(0, 2, 1), atol=1e-9)


def test_no_spin_orbit_is_j_independent(ws):
    S = by_lj(ws, ws.smatrix(*potentials(ws, spin_orbit=False)))
    np.testing.assert_allclose(S[1:, 0], S[1:, 1], atol=1e-12)


def test_transition_geometry(ws):
    G = ws._geometry  # (m, m', block, angle)
    spin_flip = np.abs(G[0, 1]).max(axis=-1)
    assert np.all(spin_flip[ws.l > 0] > 0)
    np.testing.assert_array_equal(spin_flip[ws.l == 0], 0)
    # Clebsch-Gordan orthogonality: j-independent T has no spin flip
    scale = np.abs(G).max()
    for l in range(1, LMAX + 1):
        np.testing.assert_allclose(
            G[0, 1][ws.l == l].sum(axis=0), 0, atol=1e-12 * scale
        )


def test_amplitude_symmetries(ws):
    """A spin-1/2 transition on a spin-0 target has two amplitudes."""
    Up, Un = potentials(ws)
    f = np.asarray(ws.amplitudes(ws.smatrix(Up, Un)[:, :, NEUTRON, PROTON]))[0]
    atol = 1e-14 * np.abs(f).max()
    np.testing.assert_allclose(f[0, 0], f[1, 1], rtol=0, atol=atol)
    np.testing.assert_allclose(f[0, 1], -f[1, 0], rtol=0, atol=atol)


@pytest.mark.parametrize("method", ["cc", "dwba"])
def test_ay_vanishes_without_spin_orbit(ws, method):
    """With no spin-orbit anywhere the radial matrix elements do not depend
    on j, so there is no spin flip (below Eq. (3e) of Gosset et al.)."""
    Up, Un = potentials(ws, spin_orbit=False)
    U1 = Potential(spin_orbit=np.zeros_like(ws.r))
    xs = getattr(ws, method)(Up, Un, U1)
    assert np.abs(xs.Ay).max() < 1e-12 and np.abs(xs.Q).max() < 1e-12


def test_ay_nonzero_with_equal_spin_orbits(ws):
    """Equal p/n spin-orbits make the isovector one vanish, but the
    distorted waves are still j-dependent, so Ay is not zero."""
    Up, Un = potentials(ws)
    Un = Un._replace(spin_orbit=Up.spin_orbit)
    assert np.abs(ws.cc(Up, Un, Potential(spin_orbit=0 * ws.r)).Ay).max() > 0.1


def test_amplitudes_match_legendre_partial_wave_sum(ws):
    """Pin the geometry and the sign of Ay against Eq. (13) of Gosset et al.,
    built from Legendre functions instead of Clebsch-Gordan coefficients:
    X = sum c_l [(l+1) I_l+ + l I_l-] P_l, Y = sum c_l [I_l+ - I_l-] P^1_l.
    ``lpmv`` has the Condon-Shortley phase, as in :mod:`jitr.xs.elastic`."""
    Up, Un = potentials(ws)
    S_np = ws.smatrix(Up, Un)[:, :, NEUTRON, PROTON]
    f = np.asarray(ws.amplitudes(S_np))[0]
    c = np.asarray(ws.prefactor)[0] / np.sqrt(4 * np.pi)
    S = np.asarray(S_np)[0]
    x = np.cos(ANGLES)
    weight = np.where(ws.j > ws.l, ws.l + 1, ws.l)
    sign = np.where(ws.j > ws.l, 1, -1)
    X = np.sum((c * weight * S)[:, None] * eval_legendre(ws.l[:, None], x), axis=0)
    Y = np.sum((c * sign * S)[:, None] * lpmv(1, ws.l[:, None], x), axis=0)
    scale = np.abs(X).max()
    np.testing.assert_allclose(f[1, 1], X, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(f[1, 0], Y, rtol=0, atol=1e-12 * scale)


def test_missing_spin_orbit_is_zero(ws):
    Up, Un = potentials(ws)
    Up0, Un0 = Up._replace(spin_orbit=None), Un._replace(spin_orbit=None)
    zero = 0 * ws.r
    for method in ("dwba", "cc"):
        a = getattr(ws, method)(Up0, Un0)
        b = getattr(ws, method)(
            Up._replace(spin_orbit=zero), Un._replace(spin_orbit=zero)
        )
        np.testing.assert_allclose(a.dsdo, b.dsdo, rtol=1e-12)


def test_U1_defaults(ws):
    """Explicit default U1 equals the default; each term defaults on its own;
    the neutron central potential is required."""
    Up, Un = potentials(ws)
    U1 = default_U1(Up, Un)
    np.testing.assert_allclose(
        ws.dwba(Up, Un, U1).dsdo, ws.dwba(Up, Un).dsdo, rtol=1e-12
    )
    np.testing.assert_allclose(ws.cc(Up, Un, U1).dsdo, ws.cc(Up, Un).dsdo, rtol=1e-12)
    only_central = ws.dwba(Up, Un, Potential(central=U1.central))
    np.testing.assert_allclose(only_central.dsdo, ws.dwba(Up, Un).dsdo, rtol=1e-12)
    no_so = ws.dwba(Up, Un, U1._replace(spin_orbit=0 * ws.r))
    assert not np.allclose(no_so.dsdo, ws.dwba(Up, Un).dsdo, rtol=1e-3)
    with pytest.raises(TypeError, match="central"):
        ws.dwba(Up, Un._replace(central=None))


def test_U1_sets_transition_strength(ws):
    """T is linear in U1, the distorted waves do not depend on it, and the
    DWBA cross section scales as U1^2."""
    Up, Un = potentials(ws)
    zero = 0 * ws.r  # a missing U1 term would default to the isovector one
    U1 = Potential(central=default_U1(Up, Un).central, spin_orbit=zero)
    U2 = Potential(central=2 * U1.central, spin_orbit=zero)
    T, Sp, Sn = ws.tmatrix(Up, Un, U1)
    T2, Sp2, Sn2 = ws.tmatrix(Up, Un, U2)
    np.testing.assert_allclose(T2, 2 * T, rtol=1e-12)
    np.testing.assert_allclose(Sp2, Sp)
    np.testing.assert_allclose(Sn2, Sn)
    np.testing.assert_allclose(
        ws.dwba(Up, Un, U2).dsdo, 4 * ws.dwba(Up, Un, U1).dsdo, rtol=1e-12
    )


def test_nonlocal_U1_on_the_diagonal_equals_local(ws):
    """A transition kernel U1(r) delta(r - r') on the mesh is the local U1,
    in DWBA and in CC, for central and spin-orbit terms and l-dependent
    layouts."""
    Up, Un = potentials(ws)
    U1 = default_U1(Up, Un)
    m = ws.p.mesh
    diag = lambda v: np.diag(v / (m.a * m.w))  # noqa: E731
    zero = 0 * ws.r  # a missing local U1 term would default to the isovector one
    nonlocal_ = Potential(
        central=zero,
        spin_orbit=zero,
        nonlocal_central=diag(U1.central),
        nonlocal_spin_orbit=diag(U1.spin_orbit),
    )
    np.testing.assert_allclose(
        ws.tmatrix(Up, Un, nonlocal_)[0], ws.tmatrix(Up, Un, U1)[0], atol=1e-12
    )
    np.testing.assert_allclose(
        ws.cc(Up, Un, nonlocal_).dsdo, ws.cc(Up, Un, U1).dsdo, rtol=1e-10
    )
    stacked = nonlocal_._replace(
        nonlocal_central=np.broadcast_to(
            nonlocal_.nonlocal_central, (LMAX + 1, 1, *m.F.shape)
        )
    )
    np.testing.assert_allclose(
        ws.tmatrix(Up, Un, stacked)[0], ws.tmatrix(Up, Un, U1)[0], atol=1e-12
    )


def test_energy_grid_equals_single_energies():
    E = np.array([25.0, 35.0])
    ke = REACTION.kinematics(E, relativistic=False)
    kx = REACTION.kinematics_exit(ke, 6.67, relativistic=False)
    grid = Workspace(REACTION, ke, kx, RADIUS, LMAX, NBASIS, angles=ANGLES)
    Up, Un = potentials(grid)
    dwba, cc = grid.dwba(Up, Un), grid.cc(Up, Un)
    for e, Ee in enumerate(E):
        one = workspace(Ee)
        np.testing.assert_allclose(dwba.dsdo[e], one.dwba(Up, Un).dsdo[0], rtol=1e-11)
        np.testing.assert_allclose(cc.Ay[e], one.cc(Up, Un).Ay[0], atol=1e-11)


def test_gradient_with_respect_to_U1_scale(ws):
    Up, Un = potentials(ws)

    def loss(scale, method):
        return jnp.sum(getattr(ws, method)(Up, Un, default_U1(Up, Un, scale)).rxn_pn)

    h = 1e-5
    for method in ("dwba", "cc"):
        fd = (loss(1 + h, method) - loss(1 - h, method)) / (2 * h)
        np.testing.assert_allclose(jax.grad(loss)(1.0, method), fd, rtol=1e-6)


def test_explicit_U1_for_n_minus_z_equal_one():
    """The default isovector factor diverges for N - Z = 1, but an explicit
    U1 needs no default."""
    rxn = Reaction((41, 20), (1, 1), (1, 0), (41, 21))
    ke = rxn.kinematics(30.0, relativistic=False)
    kx = rxn.kinematics_exit(ke, 6.0, relativistic=False)
    ws = Workspace(rxn, ke, kx, RADIUS, 8, 30, angles=ANGLES)
    Up, Un = potentials(ws)
    assert np.all(
        np.isfinite(
            ws.dwba(
                Up, Un, Potential(central=0.1 * Un.central, spin_orbit=0 * ws.r)
            ).dsdo
        )
    )
    with pytest.raises(ValueError, match="N - Z = 1"):
        ws.dwba(Up, Un)
