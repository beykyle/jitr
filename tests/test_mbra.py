"""Tests for the MBRA global nonlocal dispersive neutron OMP.

The no-lax tests pin the depth functions, dispersion corrections and the
Perey-Buck partial-wave kernel against Table I/II formulas, closed forms,
direct angular projection, and anchor values read from Fig. 3 of
arXiv:2403.05843. The lax tests compute n+208Pb total cross sections and
compare against experiment / Fig. 5.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from jitr.optical_potentials import mbra

from .conftest import requires_lax

EF_PB = -5.653  # jitr (ame2020) neutron Fermi energy for 208Pb [MeV]


@pytest.fixture(scope="module")
def pb():
    return mbra.resolve_coefficients(208)


def test_resolved_coefficients_match_table(pb):
    # Table I mass formulas at A = 208
    np.testing.assert_allclose(pb.VV, -69.71 - 1.140e-2 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.VS, -8.600 - 8.000e-3 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.VSO, -9.787 - 1.140e-2 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.AS_plus, -19.62 - 1.500e-2 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.AV_plus, -32.40 - 2.000e-2 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.EV_plus, 40.00 - 9.000e-2 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.ALPHA, 0.300 + 2.000e-3 * 208, rtol=1e-12)
    # Table II geometry
    np.testing.assert_allclose(pb.a, 0.6160 - 1.8200e-4 * 208, rtol=1e-12)
    np.testing.assert_allclose(pb.R, (1.1446 + 2.42e-4 * 208) * 208 ** (1 / 3))
    assert pb.beta == 0.915


def test_reduced_radius_piecewise_continuous_at_70():
    geom = mbra.DEFAULT_PARAMS[-9:-3]
    below = mbra.reduced_radius(70, *geom)
    above = mbra.reduced_radius(71, *geom)
    # the cubic (A < 70) and linear (A > 70) branches published in Table II
    # meet at A = 70 to ~2e-4 fm
    assert abs(below - (1.1446 + 2.42e-4 * 70)) < 5e-4
    assert abs(above - below) < 5e-3


def test_depths_match_fig3_anchors(pb):
    """Depth extrema/wings read off Fig. 3 of arXiv:2403.05843 (Pb dotted)."""
    # panel (a): W_S minima at x = +/-30: -15.2 (E > Ef), -10.7 (E < Ef)
    ws = mbra.Ws_depth(EF_PB + 30.0, EF_PB, pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    assert abs(ws - (-15.2)) < 0.3
    ws = mbra.Ws_depth(EF_PB - 30.0, EF_PB, pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    assert abs(ws - (-10.7)) < 0.3
    assert mbra.Ws_depth(EF_PB, EF_PB, pb.AS_plus, pb.AS_minus, pb.BS, pb.CS) == 0.0
    # panel (b): W_V wings and sub-Fermi dip
    wv_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    assert abs(mbra.Wv_depth(250.0, EF_PB, *wv_args) - (-21.4)) < 0.5
    assert abs(mbra.Wv_depth(-60.0, EF_PB, *wv_args) - (-2.96)) < 0.2
    # panel (c): W_so minimum -1.95 near x = 11 and +2.45 wings
    so_args = (pb.ASO, pb.BSO, pb.CSO, pb.DSO)
    assert abs(mbra.Wso_depth(EF_PB + 11.0, EF_PB, *so_args) - (-1.94)) < 0.05
    assert abs(mbra.Wso_depth(EF_PB + 1e4, EF_PB, *so_args) - 2.446) < 0.01


def test_wv_depth_branch_continuity(pb):
    wv_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    for boundary in (EF_PB + pb.EV_plus, EF_PB - pb.EV_minus):
        below = mbra.Wv_depth(boundary - 1e-9, EF_PB, *wv_args)
        above = mbra.Wv_depth(boundary + 1e-9, EF_PB, *wv_args)
        assert abs(above - below) < 1e-6


def test_dispersion_numeric_matches_closed_form(pb):
    """Above E_F the numerical PV integral of the symmetric Brown-Rho
    spin-orbit depth must reproduce the closed form A C x/(x^2+C^2)."""
    E = np.linspace(EF_PB + 5.0, EF_PB + 240.0, 20)
    so_args = (pb.ASO, pb.BSO, pb.CSO, pb.DSO)
    numeric = mbra.dispersion_correction(
        lambda Ep: mbra.Wso_depth(Ep, EF_PB, *so_args), E, EF_PB
    )
    closed = mbra.delta_Vso_depth(E, EF_PB, *so_args)
    np.testing.assert_allclose(numeric, closed, atol=0.02)


def test_dispersive_corrections_vanish_at_fermi_energy(pb):
    s_args = (pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    so_args = (pb.ASO, pb.BSO, pb.CSO, pb.DSO)
    assert abs(mbra.delta_Vs_depth(EF_PB, EF_PB, *s_args)) < 1e-10
    assert abs(mbra.delta_Vv_depth(EF_PB, EF_PB, *v_args)) < 1e-10
    assert abs(mbra.delta_Vso_depth(EF_PB, EF_PB, *so_args)) < 1e-10


def test_dispersive_corrections_match_fig3_anchors(pb):
    """ΔV anchor values read off Fig. 3 (Pb; generous tolerances)."""
    s_args = (pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    # panel (a): spike pair at the Fermi energy and the high-E plateau
    assert abs(mbra.delta_Vs_depth(-15.0, EF_PB, *s_args) - 6.4) < 1.0
    assert abs(mbra.delta_Vs_depth(100.0, EF_PB, *s_args) - 10.3) < 1.0
    assert abs(mbra.delta_Vs_depth(-250.0, EF_PB, *s_args) - (-1.2)) < 1.0
    # panel (c): even, +2.25 peaks at |x| ~ 50, +0.9 wings
    so_args = (pb.ASO, pb.BSO, pb.CSO, pb.DSO)
    assert abs(mbra.delta_Vso_depth(EF_PB + 50.0, EF_PB, *so_args) - 2.26) < 0.05
    assert abs(mbra.delta_Vso_depth(EF_PB - 50.0, EF_PB, *so_args) - 2.26) < 0.05
    assert abs(mbra.delta_Vso_depth(-250.0, EF_PB, *so_args) - 0.91) < 0.05
    # panel (b): minimum near +80 MeV
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    assert abs(mbra.delta_Vv_depth(80.0, EF_PB, *v_args) - (-8.6)) < 1.6
    assert abs(mbra.delta_Vv_depth(0.0, EF_PB, *v_args) - 0.5) < 1.6


def test_kernel_symmetric_finite_and_matches_direct_projection():
    beta = 0.915
    r = np.array([0.5, 2.0, 5.0, 9.5, 14.0])
    ls = np.arange(0, 5)
    K = mbra.perey_buck_kernel(r, ls, beta)
    assert K.shape == (5, 5, 5)
    assert np.all(np.isfinite(K))
    np.testing.assert_allclose(K, np.transpose(K, (0, 2, 1)), rtol=1e-12)

    # nu_l(r, r') = 2 pi r r' Int_{-1}^{1} H(|vec r - vec r'|) P_l(mu) dmu
    mu, w = leggauss(400)
    for l in ls:
        s2 = r[:, None, None] ** 2 + r[None, :, None] ** 2
        s2 = s2 - 2.0 * r[:, None, None] * r[None, :, None] * mu[None, None, :]
        H = np.exp(-s2 / beta**2) / (np.pi**1.5 * beta**3)
        direct = (
            2.0
            * np.pi
            * r[:, None]
            * r[None, :]
            * np.sum(w * H * eval_legendre(l, mu), axis=-1)
        )
        np.testing.assert_allclose(K[l], direct, rtol=1e-7, atol=1e-30)


def test_calculate_params_rejects_non_neutron():
    with pytest.raises(ValueError, match="neutron-only"):
        mbra.calculate_params((1, 1), (208, 82), 10.0, EF_PB)


def test_calculate_params_assembles_complex_depths():
    nl, loc, so = mbra.calculate_params((1, 0), (208, 82), 14.1, EF_PB)
    VV, VS, R, a, beta = nl
    assert VV < -70.0  # attractive real volume
    assert VS.imag < 0.0  # absorptive surface
    UV, _, _ = loc
    assert UV.imag < 0.0  # absorptive volume
    VSO, _, _, _ = so
    assert VSO.real < 0.0


@requires_lax
def test_n208pb_observables_match_experiment():
    """n+208Pb total cross sections vs well-known experimental values
    (the calibration data of Fig. 5 of arXiv:2403.05843)."""
    import jitr

    target = (208, 82)
    neutron = (1, 0)
    reaction = jitr.reactions.Reaction(target=target, projectile=neutron, process="EL")
    Elab = np.array([14.1, 100.0, 200.0])
    kin = reaction.kinematics(Elab)
    R = 15.0
    lmax = jitr.xs.elastic.suggest_lmax(reaction, float(Elab[-1]), R)
    ws = jitr.xs.elastic.IntegralWorkspace(
        reaction=reaction, kinematics=kin, channel_radius_fm=R, nbasis=55, lmax=lmax
    )
    rgrid = ws.radial_grid()
    ls = np.arange(lmax + 1)

    Knl, Wloc, Kso = [], [], []
    for Ecm in np.atleast_1d(kin.Ecm):
        nl_p, loc_p, so_p = mbra.calculate_params(
            neutron, target, float(Ecm), reaction.Ef
        )
        Knl.append(mbra.central_nonlocal(rgrid, ls, *nl_p))
        Wloc.append(mbra.central_local(rgrid, *loc_p))
        Kso.append(mbra.spin_orbit_nonlocal(rgrid, ls, *so_p))
    V = (
        ws.central(
            np.transpose(np.array(Knl), (1, 0, 2, 3)),
            l_dependent=True,
            energy_dependent=True,
        )
        + ws.central(np.array(Wloc), energy_dependent=True)
        + ws.spin_orbit(
            np.transpose(np.array(Kso), (1, 0, 2, 3)),
            l_dependent=True,
            energy_dependent=True,
        )
    )
    splus, sminus = ws.smatrix(V)
    assert float(np.max(np.abs(splus))) <= 1.0 + 1e-9
    assert float(np.max(np.abs(sminus))) <= 1.0 + 1e-9

    sig_t, _ = jitr.xs.elastic.integral_elastic_xs(ws.engine.grid.k, splus, sminus)
    sig_t = np.asarray(sig_t) / 1e3  # mb -> b
    # experimental sigma_tot(n + 208Pb): 5.38 b (14.1 MeV), ~4.6 b (100 MeV),
    # ~3.0 b (200 MeV)
    np.testing.assert_allclose(sig_t[0], 5.38, rtol=0.06)
    np.testing.assert_allclose(sig_t[1], 4.6, rtol=0.08)
    np.testing.assert_allclose(sig_t[2], 3.0, rtol=0.10)
    # regression pins (computed with this module at first validation)
    np.testing.assert_allclose(sig_t, [5.169, 4.673, 2.851], rtol=2e-3)
