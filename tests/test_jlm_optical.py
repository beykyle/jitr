"""
Tests for the JLM folding integration under :mod:`jitr.folding`.

Run with::

    uv run pytest tests/test_jlm_optical.py -v

The tests cover:

* density utilities (two-parameter Fermi normalization, tabulated wrapper)
* the 3-D Gaussian folder (analytic identity on Gaussian inputs,
  bulk-constant limit, R = 0 limit, volume conservation)
* JLM self-energy (isoscalar/isovector decomposition, vanishing at
  zero density, Lane sign convention)
* full ``JLMPotential`` (smooth Woods-Saxon-like shape, sensible
  literature volume integrals for ``40Ca + p @ 30 MeV``, isospin
  symmetry on N = Z targets without Coulomb shift, decreasing real
  depth with increasing energy)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import jitr
import jitr.folding as folding
from jitr.folding import (
    RHO_SAT,
    NMSelfEnergy,
    TabulatedNMSelfEnergy,
    TwoParameterFermiDensity,
    density_from_array,
    gaussian_fold,
    two_parameter_fermi,
)
from jitr.folding.jlm import (
    JLMBLambdaModelParameters,
    JLMPotential,
    JLMSelfEnergy,
    JLMSelfEnergyModelParameters,
    JLMV0Parameters,
    coulomb_potential_center,
    jlmb_lambda_factors,
    make_jlmb_parameters,
)


class LinearSelfEnergy(NMSelfEnergy):
    def V0(self, E: float, rho) -> np.ndarray:
        return np.asarray(rho, dtype=float) + E

    def W0(self, E: float, rho) -> np.ndarray:
        return -np.asarray(rho, dtype=float)

    def V1(self, E: float, rho) -> np.ndarray:
        return np.full_like(np.asarray(rho, dtype=float), 2.0)

    def W1(self, E: float, rho) -> np.ndarray:
        return np.full_like(np.asarray(rho, dtype=float), -3.0)


CA40_DENSITY = TwoParameterFermiDensity(R=3.51, a=0.563, N=20)
PB208_NEUTRON_DENSITY = TwoParameterFermiDensity(R=6.7, a=0.55, N=126)
PB208_PROTON_DENSITY = TwoParameterFermiDensity(R=6.654, a=0.547, N=82)


# ============================================================================
#  Density utilities
# ============================================================================
class TestDensities:
    def test_two_parameter_fermi_density_callable(self):
        rho = TwoParameterFermiDensity(R=5.0, a=0.5, N=40)
        values = rho(np.array([0.0, 5.0, 10.0]))
        assert values.shape == (3,)
        assert values[0] > values[1] > values[2]

    def test_two_parameter_fermi_normalization(self):
        """Integral of 4 pi r^2 rho dr matches the requested particle number."""
        for N in [8.0, 20.0, 82.0, 126.0]:
            R, a = 1.2 * N ** (1 / 3), 0.55
            r = np.linspace(0.0, 20.0, 4001)
            rho = two_parameter_fermi(r, R=R, a=a, N=N)
            N_int = 4.0 * np.pi * np.trapezoid(r**2 * rho, r)
            assert N_int == pytest.approx(N, rel=1e-3)

    def test_two_parameter_fermi_central_value(self):
        rho_0 = 0.17
        r = np.array([0.0])
        v = two_parameter_fermi(r, R=5.0, a=0.5, rho0=rho_0)
        # 1 / (1 + exp(-10)) is extremely close to 1
        assert v[0] == pytest.approx(rho_0, rel=1e-4)

    def test_two_parameter_fermi_needs_one_norm(self):
        with pytest.raises(ValueError):
            two_parameter_fermi(np.array([0.0]), R=1.0, a=0.5)

    def test_density_from_array_roundtrip(self):
        """Wrap a Fermi density through the tabulated interface; check fidelity."""
        r_tab = np.linspace(0.0, 12.0, 121)
        rho_tab = two_parameter_fermi(r_tab, R=5.0, a=0.55, N=40)
        rho_callable = density_from_array(r_tab, rho_tab)

        r_test = np.linspace(0.5, 10.0, 50)
        rho_truth = two_parameter_fermi(r_test, R=5.0, a=0.55, N=40)
        assert np.max(np.abs(rho_callable(r_test) - rho_truth)) < 1e-4

    def test_density_from_array_zero_outside(self):
        r_tab = np.linspace(0.0, 8.0, 81)
        rho_tab = two_parameter_fermi(r_tab, R=4.0, a=0.5, N=20)
        rho = density_from_array(r_tab, rho_tab)
        # Outside the tabulated range, density is zero
        assert rho(np.array([12.0]))[0] == 0.0
        assert rho.r_max == pytest.approx(8.0)


# ============================================================================
#  Gaussian fold
# ============================================================================
class TestGaussianFold:
    def test_fold_constant_in_bulk(self):
        """Folding a constant function returns the constant in the bulk."""
        M0 = -45.0
        R_inner = 3.0

        # constant inside R_inner, zero outside (sharp sphere)
        def M(r):
            return np.where(r < R_inner, M0, 0.0)

        t = 0.5  # smaller than R_inner so bulk is well-resolved
        R = np.array([0.0])
        U = gaussian_fold(R, M, t, r_max=10.0, n_quad=400)
        # In the bulk (R=0, far inside the constant region) folding returns ~M0
        assert U[0] == pytest.approx(M0, rel=2e-3)

    def test_fold_gaussian_with_gaussian_analytic(self):
        """
        Fold a 3D Gaussian ``rho_0 exp(-r^2/a^2)`` with a Gaussian smearing of
        width ``t``.

        The exact result is another Gaussian of width ``sqrt(t^2 + a^2)`` and
        amplitude ``rho_0 (a^2 / (t^2 + a^2))^(3/2)``.
        """
        rho_0, a, t = 1.0, 1.8, 1.2

        def M(r):
            return rho_0 * np.exp(-((r / a) ** 2))

        R = np.linspace(0.0, 6.0, 31)
        U_num = gaussian_fold(R, M, t, r_max=15.0, n_quad=500)

        sigma2 = t**2 + a**2
        amp = rho_0 * (a**2 / sigma2) ** 1.5
        U_exact = amp * np.exp(-(R**2) / sigma2)

        assert np.max(np.abs(U_num - U_exact)) < 1e-4

    def test_fold_R0_limit_consistent(self):
        """The R=0 branch matches a tiny-R evaluation from the general formula."""

        def M(r):
            return np.exp(-((r / 2.0) ** 2)) * (1.0 + 0.5 * r)

        t = 1.2
        U0 = gaussian_fold(np.array([0.0]), M, t)[0]
        U_eps = gaussian_fold(np.array([1e-5]), M, t)[0]
        assert U0 == pytest.approx(U_eps, rel=1e-5)

    def test_fold_preserves_volume_integral(self):
        """Gaussian folding preserves the total volume integral."""

        # Smooth, compact M
        def M(r):
            return -50.0 * np.exp(-((r / 3.0) ** 2))

        # Total integral of M over 3-D space:
        # int_0^inf 4 pi r^2 M(r) dr (analytic for a Gaussian)
        # = -50 * pi^(3/2) * 3^3
        I_M_exact = -50.0 * np.pi**1.5 * 27.0

        t = 1.5
        R = np.linspace(0.0, 15.0, 601)
        U = gaussian_fold(R, M, t, r_max=20.0, n_quad=500)
        I_U = 4.0 * np.pi * np.trapezoid(R**2 * U, R)
        assert I_U == pytest.approx(I_M_exact, rel=1e-3)

    def test_fold_invalid_t(self):
        with pytest.raises(ValueError):
            gaussian_fold(np.array([1.0]), lambda r: r, t=-1.0)

    def test_fold_invalid_rmax(self):
        with pytest.raises(ValueError):
            gaussian_fold(np.array([1.0]), lambda r: r, t=1.0, r_max=0.0)


# ============================================================================
#  JLM self-energy
# ============================================================================
class TestJLMSelfEnergy:
    def test_generic_base_combines_lane_components(self):
        rho = np.array([0.1, 0.2])
        beta = 0.25
        V, W = LinearSelfEnergy("n")(5.0, rho, beta)
        np.testing.assert_allclose(V, rho + 5.0 + 0.5)
        np.testing.assert_allclose(W, -rho - 0.75)

    def test_zero_density_zero_potential(self):
        se = JLMSelfEnergy("n")
        rho = np.zeros(5)
        beta = np.zeros(5)
        V, W = se(20.0, rho, beta)
        assert np.allclose(V, 0.0)
        assert np.allclose(W, 0.0)

    def test_isoscalar_only_at_zero_beta(self):
        """At beta=0, projectile species should not matter."""
        rho = np.array([0.16])
        beta = np.zeros_like(rho)
        Vn, Wn = JLMSelfEnergy("n")(30.0, rho, beta)
        Vp, Wp = JLMSelfEnergy("p")(30.0, rho, beta)
        assert Vn[0] == pytest.approx(Vp[0])
        assert Wn[0] == pytest.approx(Wp[0])

    def test_lane_sign_convention(self):
        """In an n-rich nucleus (beta > 0): V_n less attractive than V_p,
        |W_n| greater than |W_p| (more absorption for n in n-rich)."""
        rho = np.array([0.16])
        beta = np.array([0.2])
        Vn, Wn = JLMSelfEnergy("n")(30.0, rho, beta)
        Vp, Wp = JLMSelfEnergy("p")(30.0, rho, beta)
        # n is shallower (less negative)
        assert Vn[0] > Vp[0]
        # n absorbs more (more negative W)
        assert Wn[0] < Wp[0]

    def test_real_depth_decreases_with_energy(self):
        """|V_0| decreases monotonically with E over the JLM-fit range."""
        rho = np.array([0.16])
        se = JLMSelfEnergy("n")
        depths = [-se.V0(E, rho)[0] for E in [5.0, 20.0, 50.0, 100.0, 150.0]]
        assert all(d1 > d2 for d1, d2 in zip(depths[:-1], depths[1:], strict=False))

    def test_imag_absorptive_at_all_energies(self):
        """W_0 stays absorptive (<=0) for energies that include
        proton-Coulomb-shifted negative values."""
        rho = np.array([0.16])
        se = JLMSelfEnergy("n")
        for E in [-15.0, -5.0, 0.0, 5.0, 30.0, 60.0, 100.0]:
            W = se.W0(E, rho)[0]
            assert W <= 0.0, f"W0 became positive at E = {E} MeV (W = {W})"

    def test_invalid_projectile(self):
        with pytest.raises(ValueError):
            JLMSelfEnergy("d")

    def test_density_dependence_smooth(self):
        """V_0 is smooth across density grid (no NaNs / spikes)."""
        rho_grid = np.linspace(0.0, 0.20, 41)
        V = JLMSelfEnergy("n").V0(30.0, rho_grid)
        assert np.isfinite(V).all()

    def test_custom_model_parameters_change_self_energy(self):
        base = JLMSelfEnergy("n")
        tuned = JLMSelfEnergy(
            "n",
            model_parameters=replace(
                JLMSelfEnergyModelParameters(),
                V0=replace(JLMV0Parameters(), energy_constant=125.0),
            ),
        )
        rho = np.array([0.16])
        beta = np.array([0.0])
        V_base, _ = base(30.0, rho, beta)
        V_tuned, _ = tuned(30.0, rho, beta)
        assert V_tuned[0] < V_base[0]


# ============================================================================
#  Auxiliary
# ============================================================================
class TestAuxiliary:
    def test_coulomb_potential_center_pb208(self):
        VC = coulomb_potential_center(Z=82, A=208)
        # 1.5 * 82 * 1.44 / (1.25 * 208^(1/3))
        # = 177.13 / (1.25 * 5.925) = 177.13 / 7.406 = ~23.9 MeV
        assert VC == pytest.approx(23.9, abs=0.5)

    def test_coulomb_potential_center_ca40(self):
        VC = coulomb_potential_center(Z=20, A=40)
        assert VC == pytest.approx(10.1, abs=0.5)

    def test_jlmb_lambda_factors_finite(self):
        for E in [1.0, 14.0, 30.0, 100.0]:
            vals = jlmb_lambda_factors(E)
            assert all(np.isfinite(v) and v > 0 for v in vals)

    def test_jlmb_lambda_model_parameters_are_configurable(self):
        default = jlmb_lambda_factors(30.0)
        tuned = jlmb_lambda_factors(
            30.0,
            model_parameters=replace(JLMBLambdaModelParameters(), lambda_W1_value=2.1),
        )
        assert tuned[3] == pytest.approx(2.1)
        assert tuned[3] != default[3]

    def test_make_jlmb_parameters_overrides(self):
        p = make_jlmb_parameters(E=30.0, t_R=2.0, n_quad=300)
        assert p.t_R == 2.0
        assert p.n_quad == 300

    def test_make_jlmb_parameters_accepts_custom_lambda_model(self):
        p = make_jlmb_parameters(
            E=30.0,
            lambda_model_parameters=replace(
                JLMBLambdaModelParameters(), lambda_V1_value=1.3
            ),
        )
        assert p.lambda_V1 == pytest.approx(1.3)


# ============================================================================
#  Tabulated self-energy
# ============================================================================
class TestTabulatedSelfEnergy:
    def test_roundtrip_against_analytical(self):
        """Sampling the analytical self-energy and interpolating reproduces it."""
        se_a = JLMSelfEnergy("n")
        se_t = TabulatedNMSelfEnergy.from_analytical(
            se_a,
            E_grid=np.linspace(1.0, 200.0, 80),
            rho_grid=np.linspace(0.0, 0.20, 51),
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            E = float(rng.uniform(5.0, 150.0))
            rho = np.array([float(rng.uniform(0.001, 0.19))])
            beta = float(rng.uniform(-0.2, 0.3))
            Va, Wa = se_a(E, rho, np.array([beta]))
            Vt, Wt = se_t(E, rho, np.array([beta]))
            assert Va[0] == pytest.approx(Vt[0], abs=0.05)
            assert Wa[0] == pytest.approx(Wt[0], abs=0.05)

    def test_inside_potential(self):
        """JLMPotential gives same volume integrals with tabulated vs analytic."""
        se_a = JLMSelfEnergy("p")
        se_t = TabulatedNMSelfEnergy.from_analytical(se_a)
        rho_dens = CA40_DENSITY
        params = make_jlmb_parameters(E=30.0)
        pot_a = JLMPotential(rho_dens, rho_dens, se_a, parameters=params)
        pot_t = JLMPotential(rho_dens, rho_dens, se_t, parameters=params)
        JV_a, JW_a = pot_a.volume_integrals(30.0, "p", 40, 20, 40)
        JV_t, JW_t = pot_t.volume_integrals(30.0, "p", 40, 20, 40)
        assert JV_a == pytest.approx(JV_t, rel=5e-3)
        assert JW_a == pytest.approx(JW_t, rel=5e-3)

    def test_text_io_roundtrip(self, tmp_path):
        """save_text_file + from_text_file reproduces the tables exactly."""
        se_a = JLMSelfEnergy("n")
        se_t = TabulatedNMSelfEnergy.from_analytical(se_a)
        path = tmp_path / "se.txt"
        se_t.save_text_file(str(path))
        se_t2 = TabulatedNMSelfEnergy.from_text_file(str(path), projectile="n")
        np.testing.assert_allclose(se_t.V0_table, se_t2.V0_table)
        np.testing.assert_allclose(se_t.W0_table, se_t2.W0_table)
        np.testing.assert_allclose(se_t.V1_table, se_t2.V1_table)
        np.testing.assert_allclose(se_t.W1_table, se_t2.W1_table)

    def test_lane_sign_in_tabulated(self):
        """Lane sign convention preserved through tabulation."""
        se_t_n = TabulatedNMSelfEnergy.from_analytical(
            JLMSelfEnergy("n"), projectile="n"
        )
        se_t_p = TabulatedNMSelfEnergy.from_analytical(
            JLMSelfEnergy("n"), projectile="p"
        )
        rho = np.array([0.16])
        beta = np.array([0.2])
        Vn, Wn = se_t_n(30.0, rho, beta)
        Vp, Wp = se_t_p(30.0, rho, beta)
        # For n-rich (beta > 0), V_p more attractive, W_n more absorptive
        assert Vp[0] < Vn[0]
        assert Wn[0] < Wp[0]

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            TabulatedNMSelfEnergy(
                E_grid=np.linspace(1.0, 100.0, 5),
                rho_grid=np.linspace(0.0, 0.2, 3),
                V0_table=np.zeros((5, 4)),  # wrong shape
                W0_table=np.zeros((5, 3)),
            )

    def test_invalid_grids(self):
        # Not strictly increasing
        with pytest.raises(ValueError):
            TabulatedNMSelfEnergy(
                E_grid=np.array([1.0, 1.0, 2.0]),
                rho_grid=np.linspace(0.0, 0.2, 3),
                V0_table=np.zeros((3, 3)),
                W0_table=np.zeros((3, 3)),
            )

    def test_clamps_out_of_range_inputs_when_bounds_error_disabled(self):
        se = TabulatedNMSelfEnergy.from_analytical(
            JLMSelfEnergy("n"),
            E_grid=np.linspace(10.0, 20.0, 4),
            rho_grid=np.linspace(0.0, 0.2, 5),
        )
        low = se.V0(1.0, np.array([-1.0]))
        edge = se.V0(10.0, np.array([0.0]))
        np.testing.assert_allclose(low, edge)


# ============================================================================
#  Full optical potential
# ============================================================================
def _ca40_pot(E: float, projectile: str = "p"):
    """Helper: 40Ca + p|n potential at energy E."""
    se = JLMSelfEnergy(projectile)
    return JLMPotential(
        CA40_DENSITY,
        CA40_DENSITY,
        se,
        parameters=make_jlmb_parameters(E=E),
    )


def _pb208_pot(E: float, projectile: str = "n"):
    se = JLMSelfEnergy(projectile)
    return JLMPotential(
        PB208_NEUTRON_DENSITY,
        PB208_PROTON_DENSITY,
        se,
        parameters=make_jlmb_parameters(E=E),
    )


class TestJLMPotentialShape:
    def test_ca40_p_30_shape(self):
        """V(R), W(R) for 40Ca + p @ 30 MeV: smooth, peaked at R=0, fall to 0."""
        pot = _ca40_pot(30.0, "p")
        R = np.linspace(0.0, 12.0, 121)
        V, W = pot.compute(R, E=30.0, projectile="p", Z=20, A=40)

        # central is most attractive / absorptive
        assert V[0] == V.min()
        assert W[0] == W.min()
        # central depth in physical range
        assert -50.0 < V[0] < -30.0
        assert -10.0 < W[0] < 0.0
        # falls to zero by 12 fm
        assert abs(V[-1]) < 1.0
        assert abs(W[-1]) < 0.5
        # monotone decreasing |V|, |W| past the central plateau
        assert np.all(np.diff(np.abs(V[20:])) <= 1e-6)

    def test_isospin_symmetry_without_coulomb(self):
        """For N = Z target without Coulomb shift, V(n) == V(p)."""
        # Use a fictitious symmetric 40Ca-like target: rho_n = rho_p
        params = make_jlmb_parameters(E=30.0, apply_coulomb_shift=False)
        pot_n = JLMPotential(
            CA40_DENSITY, CA40_DENSITY, JLMSelfEnergy("n"), parameters=params
        )
        pot_p = JLMPotential(
            CA40_DENSITY, CA40_DENSITY, JLMSelfEnergy("p"), parameters=params
        )
        R = np.linspace(0.0, 10.0, 51)
        Vn, Wn = pot_n.compute(R, E=30.0, projectile="n")
        Vp, Wp = pot_p.compute(R, E=30.0, projectile="p")
        assert np.allclose(Vn, Vp, atol=1e-8)
        assert np.allclose(Wn, Wp, atol=1e-8)

    def test_higher_energy_shallower_real(self):
        """|V(R=0)| at high E < |V(R=0)| at low E (real depth decreases)."""
        pot_low = _ca40_pot(10.0, "n")
        pot_high = _ca40_pot(100.0, "n")
        Vlow, _ = pot_low.compute(np.array([0.0]), 10.0, "n")
        Vhigh, _ = pot_high.compute(np.array([0.0]), 100.0, "n")
        assert abs(Vhigh[0]) < abs(Vlow[0])

    def test_higher_energy_deeper_imag(self):
        """|W(R=0)| at high E > |W(R=0)| at low E (more absorption at high E)."""
        pot_low = _ca40_pot(10.0, "n")
        pot_high = _ca40_pot(100.0, "n")
        _, Wlow = pot_low.compute(np.array([0.0]), 10.0, "n")
        _, Whigh = pot_high.compute(np.array([0.0]), 100.0, "n")
        assert abs(Whigh[0]) > abs(Wlow[0])

    def test_lane_sign_pb208(self):
        """In 208Pb (n-rich), proton sees deeper real V than neutron."""
        # Disable Coulomb shift to isolate Lane physics.
        params = make_jlmb_parameters(E=30.0, apply_coulomb_shift=False)
        pot_n = JLMPotential(
            PB208_NEUTRON_DENSITY,
            PB208_PROTON_DENSITY,
            JLMSelfEnergy("n"),
            parameters=params,
        )
        pot_p = JLMPotential(
            PB208_NEUTRON_DENSITY,
            PB208_PROTON_DENSITY,
            JLMSelfEnergy("p"),
            parameters=params,
        )
        Vn, _ = pot_n.compute(np.array([0.0]), 30.0, "n")
        Vp, _ = pot_p.compute(np.array([0.0]), 30.0, "p")
        # V_p deeper (more negative) than V_n in n-rich nucleus
        assert Vp[0] < Vn[0]


class TestJLMPotentialVolumeIntegrals:
    def test_ca40_p_30_J_V(self):
        """J_V/A for 40Ca + p @ 30 MeV in 400 +/- 30 MeV*fm^3."""
        pot = _ca40_pot(30.0, "p")
        JV, _ = pot.volume_integrals(E=30.0, projectile="p", A_target=40, Z=20, A=40)
        assert 370 < JV < 440

    def test_ca40_p_30_J_W(self):
        """J_W/A for 40Ca + p @ 30 MeV in 70 +/- 25 MeV*fm^3."""
        pot = _ca40_pot(30.0, "p")
        _, JW = pot.volume_integrals(E=30.0, projectile="p", A_target=40, Z=20, A=40)
        assert 45 < JW < 95

    def test_pb208_n_14_reasonable(self):
        """208Pb + n @ 14 MeV gives J_V/A in 300-500 MeV*fm^3, J_W/A > 20."""
        pot = _pb208_pot(14.0, "n")
        JV, JW = pot.volume_integrals(E=14.0, projectile="n", A_target=208)
        assert 300 < JV < 500
        assert 20 < JW < 90

    def test_pb208_lane_difference(self):
        """J_V(p) - J_V(n) > 0 for 208Pb at 14 MeV (Lane convention)."""
        pot_n = _pb208_pot(14.0, "n")
        JVn, _ = pot_n.volume_integrals(E=14.0, projectile="n", A_target=208)
        # Disable Coulomb shift for fair Lane comparison (otherwise the
        # proton case is dominated by V0 shift, not Lane).
        params = make_jlmb_parameters(E=14.0, apply_coulomb_shift=False)
        pot_p_nocoul = JLMPotential(
            PB208_NEUTRON_DENSITY,
            PB208_PROTON_DENSITY,
            JLMSelfEnergy("p"),
            parameters=params,
        )
        pot_n_nocoul = JLMPotential(
            PB208_NEUTRON_DENSITY,
            PB208_PROTON_DENSITY,
            JLMSelfEnergy("n"),
            parameters=params,
        )
        JVn2, _ = pot_n_nocoul.volume_integrals(14.0, "n", A_target=208)
        JVp2, _ = pot_p_nocoul.volume_integrals(14.0, "p", A_target=208)
        assert JVp2 > JVn2  # Lane: proton deeper

    def test_volume_integrals_match_direct_compute(self):
        """volume_integrals reproduces a manual quadrature."""
        pot = _ca40_pot(30.0, "p")
        R = np.linspace(0.0, 15.0, 600)
        V, W = pot.compute(R, 30.0, "p", Z=20, A=40)
        JV_manual = -4.0 * np.pi * np.trapezoid(R**2 * V, R) / 40.0
        JW_manual = -4.0 * np.pi * np.trapezoid(R**2 * W, R) / 40.0
        JV, JW = pot.volume_integrals(30.0, "p", 40, 20, 40)
        assert JV == pytest.approx(JV_manual, rel=1e-6)
        assert JW == pytest.approx(JW_manual, rel=1e-6)


class TestJLMPotentialInputs:
    def test_invalid_projectile(self):
        pot = _ca40_pot(30.0, "p")
        with pytest.raises(ValueError):
            pot.compute(np.array([0.0]), 30.0, "d")

    def test_warns_proton_without_Z_A(self):
        pot = _ca40_pot(30.0, "p")
        with pytest.warns(UserWarning):
            pot.compute(np.array([0.0]), 30.0, "p")


class TestNamespaceExports:
    def test_jitr_exports_folding_namespace(self):
        assert jitr.folding is folding

    def test_folding_exports_generic_surfaces(self):
        assert folding.NMSelfEnergy is NMSelfEnergy
        assert folding.RHO_SAT == RHO_SAT
        assert folding.TabulatedNMSelfEnergy is TabulatedNMSelfEnergy
        assert folding.jlm.JLMSelfEnergy is JLMSelfEnergy
