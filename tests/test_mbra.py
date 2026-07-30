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

from jitr.optical_potentials import dispersion, mbra, potential_forms

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
    # panel (b): high-E wing (A_V^+ + sqrt tail) matches the figure
    wv_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    assert abs(mbra.Wv_depth(250.0, EF_PB, *wv_args) - (-21.4)) < 0.5
    # sub-Fermi dip: Fig. 3(b) shows ~-3.0 near E = -60, which the
    # author-confirmed A_V^- = -8.4 MeV cannot reproduce (the figure
    # corresponds to a depth near -48.4 MeV); pin the module value
    assert abs(mbra.Wv_depth(-60.0, EF_PB, *wv_args) - (-0.514)) < 0.05
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
    numeric = dispersion.subtracted_dispersion_correction(
        lambda Ep: mbra.Wso_depth(Ep, EF_PB, *so_args), E, EF_PB
    )
    closed = mbra.delta_Vso_depth(E, EF_PB, *so_args)
    np.testing.assert_allclose(numeric, closed, atol=0.02)


def test_dispersive_corrections_vanish_at_fermi_energy(pb):
    s_args = (pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    so_args = (pb.ASO, pb.BSO, pb.CSO, pb.DSO)
    for method in ("analytic", "numerical"):
        assert abs(mbra.delta_Vs_depth(EF_PB, EF_PB, *s_args, method=method)) < 1e-10
        assert abs(mbra.delta_Vv_depth(EF_PB, EF_PB, *v_args, method=method)) < 1e-10
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
    # panel (b): the figure reads ~0.5 at E = 0 and ~-8.6 at the minimum
    # near +80 MeV, but corresponds to a sub-Fermi depth near -48.4 MeV;
    # with the author-confirmed A_V^- = -8.4 MeV pin the module values
    # (exact closed-form dispersion of Eqs. 14-16)
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    assert abs(mbra.delta_Vv_depth(80.0, EF_PB, *v_args) - (-6.4618)) < 0.01
    assert abs(mbra.delta_Vv_depth(0.0, EF_PB, *v_args) - (-0.6288)) < 0.01


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


def test_perey_buck_nonlocal_matches_grid_kernel():
    """The exported pointwise kernel must agree with the grid kernel."""
    beta = 0.915
    r = np.array([0.5, 2.0, 5.0, 14.0])
    ls = np.arange(0, 5)
    K = mbra.perey_buck_kernel(r, ls, beta)
    for l in ls:
        for i, ri in enumerate(r):
            for j, rj in enumerate(r):
                point = potential_forms.perey_buck_nonlocal(ri, rj, beta, l)
                np.testing.assert_allclose(point, K[l, i, j], rtol=1e-12)


def test_perey_buck_kernel_local_limit_normalization():
    """For r >> beta the kernel norm over r' approaches the local limit:
    exactly 1 for l = 0, and 1 - l(l+1)β²/(4r²) + O(β⁴) for higher l
    (large-z asymptotics of ive(l+1/2, z))."""
    beta, r = 0.915, 5.0
    rp = np.linspace(1e-3, 12.0, 4000)
    for l in (0, 5):
        g = np.array([potential_forms.perey_buck_nonlocal(r, x, beta, l) for x in rp])
        expected = 1.0 - l * (l + 1) * beta**2 / (4.0 * r**2)
        tol = 1e-6 if l == 0 else 0.05
        np.testing.assert_allclose(np.trapezoid(g, rp), expected, rtol=tol)


def test_perey_buck_kernel_rejects_nonpositive_r():
    with pytest.raises(ValueError, match="strictly positive"):
        mbra.perey_buck_kernel(np.array([0.0, 1.0]), np.array([0]), 0.915)
    with pytest.raises(ValueError, match="strictly positive"):
        potential_forms.perey_buck_nonlocal(0.0, 1.0, 0.915, 0)


def test_dispersion_correction_raises_on_node_coincidence(pb):
    """An evaluation energy exactly on a quadrature node must raise, not
    silently drop that node's contribution (was a ~0.5 MeV silent error).
    The analytic default has no quadrature and is immune."""
    s_args = (pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    x_quad, _, _, _ = dispersion._subtracted_quadrature(
        EF_PB, dispersion.DEFAULT_SUBTRACTED_SEGMENT_OFFSETS
    )
    node = float(x_quad[128])
    with pytest.raises(ValueError, match="quadrature node"):
        mbra.delta_Vs_depth(node, EF_PB, *s_args, method="numerical")
    nearby = mbra.delta_Vs_depth(node + 1e-3, EF_PB, *s_args, method="numerical")
    assert np.isfinite(nearby)
    assert np.isfinite(mbra.delta_Vs_depth(node, EF_PB, *s_args))


def test_delta_depth_rejects_unknown_method(pb):
    s_args = (pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    with pytest.raises(ValueError, match="method"):
        mbra.delta_Vs_depth(10.0, EF_PB, *s_args, method="exact")
    with pytest.raises(ValueError, match="method"):
        mbra.delta_Vv_depth(10.0, EF_PB, *v_args, method="exact")


def test_delta_Vs_analytic_matches_numerical(pb):
    """The closed-form surface correction must agree with the PV quadrature
    (which carries a ~0.01 MeV discretization floor)."""
    s_args = (pb.AS_plus, pb.AS_minus, pb.BS, pb.CS)
    E = np.linspace(-240.0, 240.0, 97) + 0.1371
    ana = mbra.delta_Vs_depth(E, EF_PB, *s_args)
    num = mbra.delta_Vs_depth(E, EF_PB, *s_args, method="numerical")
    np.testing.assert_allclose(ana, num, atol=0.02)


def test_delta_Vv_analytic_matches_numerical(pb):
    """The closed-form volume correction must agree with the PV quadrature,
    across both branch points (E_F - E_V^- and E_F + E_V^+). The quadrature
    truncates the Brown-Rho constant tails at +/-3e4 MeV, a ~0.1 MeV floor
    at the grid edges; the analytic form is exact."""
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    E = np.linspace(-240.0, 240.0, 97) + 0.1371
    ana = mbra.delta_Vv_depth(E, EF_PB, *v_args)
    num = mbra.delta_Vv_depth(E, EF_PB, *v_args, method="numerical")
    np.testing.assert_allclose(ana, num, atol=0.11)


def test_halfline_partners_symmetric_limit(pb):
    """Equal amplitudes on both half-lines must recover the symmetric
    closed forms (the Quesada et al. Eq. 7 limit): B x/(x^2+B^2) for the
    Brown-Rho, and the quadrature of the symmetrized damped shape."""
    E = np.linspace(-240.0, 240.0, 97) + 0.1371
    x = E - EF_PB
    sym = mbra.brown_rho_halfline_partner(
        E, EF_PB, pb.BV
    ) - mbra.brown_rho_halfline_partner(2.0 * EF_PB - E, EF_PB, pb.BV)
    np.testing.assert_allclose(sym, pb.BV * x / (x**2 + pb.BV**2), rtol=1e-12)

    sym = dispersion.damped_brown_rho_halfline_partner(
        E, EF_PB, pb.BS, pb.CS
    ) - dispersion.damped_brown_rho_halfline_partner(2.0 * EF_PB - E, EF_PB, pb.BS, pb.CS)
    x_abs = np.abs(x)
    num = dispersion.subtracted_dispersion_correction(
        lambda Ep: (Ep - EF_PB) ** 2
        * np.exp(-pb.CS * np.abs(Ep - EF_PB))
        / ((Ep - EF_PB) ** 2 + pb.BS**2),
        E,
        EF_PB,
    )
    assert np.all(x_abs > 0)
    np.testing.assert_allclose(sym, num, atol=0.02)


def test_delta_Vv_analytic_continuous_at_branch_points(pb):
    v_args = (pb.AV_plus, pb.AV_minus, pb.BV, pb.EV_plus, pb.EV_minus, pb.ALPHA)
    for boundary in (EF_PB - pb.EV_minus, EF_PB + pb.EV_plus):
        below = mbra.delta_Vv_depth(boundary - 1e-9, EF_PB, *v_args)
        above = mbra.delta_Vv_depth(boundary + 1e-9, EF_PB, *v_args)
        assert abs(above - below) < 1e-6
    assert mbra.delta_Vv_depth(EF_PB, EF_PB, *v_args) == 0.0


def test_partners_scalar_return_convention():
    """Scalar E in -> Python float out for the closed-form partners."""
    for val in (
        dispersion.brown_rho_halfline_partner(10.0, EF_PB, 135.0),
        dispersion.damped_brown_rho_halfline_partner(10.0, EF_PB, 11.11, 9.2e-3),
        dispersion.sub_fermi_suppression_partner(10.0, EF_PB, 135.0, 25.5),
    ):
        assert isinstance(val, float)


def test_av_minus_constant():
    """Table I prints the sub-Fermi volume depth as "-8.400A"; the authors
    confirmed it is a constant -8.4 MeV with no mass dependence."""
    idx = mbra.PARAM_NAMES.index("av_minus_0")
    assert mbra.DEFAULT_PARAMS[idx] == -8.4
    assert "av_minus_A" not in mbra.PARAM_NAMES
    assert mbra.resolve_coefficients(40).AV_minus == -8.4
    assert mbra.resolve_coefficients(208).AV_minus == -8.4


def test_calculate_params_vectorized_matches_scalar():
    """Array-Elab depths must equal the per-scalar loop; scalar input must
    keep returning Python complex."""
    Elab = np.array([2.0, 14.1, 100.0, 200.0])
    nl, loc, so = mbra.calculate_params((1, 0), (208, 82), Elab, EF_PB)
    VV, VS, R, a, beta = nl
    UV, _, _ = loc
    VSO, _, _, _ = so
    for depth in (VS, UV, VSO):
        assert depth.shape == Elab.shape
        assert depth.dtype == np.complex128
    for i, E in enumerate(Elab):
        nl_s, loc_s, so_s = mbra.calculate_params((1, 0), (208, 82), float(E), EF_PB)
        assert (VV, R, a, beta) == (nl_s[0], nl_s[2], nl_s[3], nl_s[4])
        np.testing.assert_allclose(VS[i], nl_s[1], rtol=1e-14)
        np.testing.assert_allclose(UV[i], loc_s[0], rtol=1e-14)
        np.testing.assert_allclose(VSO[i], so_s[0], rtol=1e-14)
        assert isinstance(nl_s[1], complex)
        assert isinstance(loc_s[0], complex)
        assert isinstance(so_s[0], complex)


def test_builders_vectorized_match_scalar_stack():
    """Array-depth builders (with and without a shared precomputed kernel)
    must equal np.stack of scalar builds."""
    r = np.linspace(0.1, 12.0, 8)
    ls = np.arange(3)
    VV, R, a, beta = -72.0, 7.1, 0.58, 0.915
    VS = np.array([-9.0 - 2.0j, -8.5 - 4.0j])
    UV = np.array([-1.0 - 0.5j, -2.0 - 3.0j])
    VSO = np.array([-11.0 + 0.1j, -10.5 - 0.2j])
    k = mbra.perey_buck_kernel(r, ls, beta)

    K_nl = mbra.central_nonlocal(r, ls, VV, VS, R, a, beta)
    assert K_nl.shape == (3, 2, 8, 8)
    stack = np.stack(
        [mbra.central_nonlocal(r, ls, VV, vs, R, a, beta) for vs in VS], axis=1
    )
    np.testing.assert_allclose(K_nl, stack, rtol=1e-14)
    np.testing.assert_allclose(
        mbra.central_nonlocal(r, ls, VV, VS, R, a, beta, kernel=k), K_nl, rtol=0
    )

    K_so = mbra.spin_orbit_nonlocal(r, ls, VSO, R, a, beta)
    assert K_so.shape == (3, 2, 8, 8)
    stack = np.stack(
        [mbra.spin_orbit_nonlocal(r, ls, vso, R, a, beta) for vso in VSO], axis=1
    )
    np.testing.assert_allclose(K_so, stack, rtol=1e-14)
    np.testing.assert_allclose(
        mbra.spin_orbit_nonlocal(r, ls, VSO, R, a, beta, kernel=k), K_so, rtol=0
    )
    np.testing.assert_allclose(
        mbra.spin_orbit_nonlocal(r, ls, VSO[0], R, a, beta, kernel=k),
        mbra.spin_orbit_nonlocal(r, ls, VSO[0], R, a, beta),
        rtol=0,
    )

    W_loc = mbra.central_local(r, UV, R, a)
    assert W_loc.shape == (2, 8)
    stack = np.stack([mbra.central_local(r, uv, R, a) for uv in UV], axis=0)
    np.testing.assert_allclose(W_loc, stack, rtol=1e-14)


def test_assemble_terms_matches_scalar_assembly():
    """assemble_terms must equal the old per-energy scalar loop."""
    r = np.linspace(0.1, 12.0, 8)
    ls = np.arange(3)
    Elab = np.array([14.1, 100.0])
    K_nl, W_loc, K_so = mbra.assemble_terms(r, ls, (1, 0), (208, 82), Elab, EF_PB)
    assert K_nl.shape == (3, 2, 8, 8)
    assert W_loc.shape == (2, 8)
    assert K_so.shape == (3, 2, 8, 8)
    Knl_s, Wloc_s, Kso_s = [], [], []
    for E in Elab:
        nl_p, loc_p, so_p = mbra.calculate_params((1, 0), (208, 82), float(E), EF_PB)
        Knl_s.append(mbra.central_nonlocal(r, ls, *nl_p))
        Wloc_s.append(mbra.central_local(r, *loc_p))
        Kso_s.append(mbra.spin_orbit_nonlocal(r, ls, *so_p))
    np.testing.assert_allclose(K_nl, np.stack(Knl_s, axis=1), rtol=1e-13)
    np.testing.assert_allclose(W_loc, np.stack(Wloc_s, axis=0), rtol=1e-13)
    np.testing.assert_allclose(K_so, np.stack(Kso_s, axis=1), rtol=1e-13)


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

    Knl, Wloc, Kso = mbra.assemble_terms(
        rgrid, ls, neutron, target, kin.Elab, reaction.Ef
    )
    V = (
        ws.central(Knl, l_dependent=True, energy_dependent=True)
        + ws.central(Wloc, energy_dependent=True)
        + ws.spin_orbit(Kso, l_dependent=True, energy_dependent=True)
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
    # regression pins (analytic-dispersion default; the numerical-quadrature
    # values were [5.1962, 4.6455, 2.8129], the -48.40 MeV values
    # [5.1695, 4.6695, 2.8508])
    np.testing.assert_allclose(sig_t, [5.1956, 4.6469, 2.8156], rtol=2e-3)


@requires_lax
def test_spin_orbit_coupling_factor():
    """Pin the <l.sigma> = {l, -(l+1)} split between mbra.spin_orbit_nonlocal
    (which passes the unscaled form factor) and the workspace (which applies
    the coupling): in the Born limit of a weak real SO term, the phase-shift
    ratio between the j = l + 1/2 and j = l - 1/2 channels is -l/(l+1)."""
    import jitr

    target = (208, 82)
    neutron = (1, 0)
    reaction = jitr.reactions.Reaction(target=target, projectile=neutron, process="EL")
    kin = reaction.kinematics(np.array([20.0]))
    lmax = 6
    ws = jitr.xs.elastic.IntegralWorkspace(
        reaction=reaction, kinematics=kin, channel_radius_fm=15.0, nbasis=40, lmax=lmax
    )
    rgrid = ws.radial_grid()
    ls = np.arange(lmax + 1)
    c = mbra.resolve_coefficients(208)
    Kso = mbra.spin_orbit_nonlocal(rgrid, ls, 0.1 + 0.0j, c.R, c.a, c.beta)
    V = ws.spin_orbit(
        np.transpose(np.array([Kso]), (1, 0, 2, 3)),
        l_dependent=True,
        energy_dependent=True,
    )
    splus, sminus = ws.smatrix(V)
    # splus rows are l = 0..lmax; sminus rows are l = 1..lmax (no j = l - 1/2
    # channel exists for l = 0)
    delta_plus = np.angle(np.asarray(splus)[:, 0]) / 2.0
    delta_minus = np.angle(np.asarray(sminus)[:, 0]) / 2.0
    # l = 0: <l.sigma> = 0 for j = 1/2 in the plus branch -> no phase shift
    assert abs(np.asarray(splus)[0, 0] - 1.0) < 1e-9
    for l in range(1, lmax + 1):
        np.testing.assert_allclose(
            delta_plus[l] / delta_minus[l - 1], -l / (l + 1), rtol=1e-2
        )
