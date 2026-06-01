"""Tests for the current folding and JLM public API."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from jitr.folding import jlm
from jitr.utils.constants import ALPHA, HBARC
from jitr.utils.density import (
    TwoParameterFermiDensity,
    density_from_array,
    two_parameter_fermi,
)

ENERGY_DELTA_MEV = 1e-14
W0_FERMI_TOLERANCE = 1e-20
W1_FERMI_TOLERANCE = 1e-6


class TestFoldingDensities:
    def test_two_parameter_fermi_normalization_matches_requested_particle_number(self):
        r_grid = np.linspace(0.0, 20.0, 4001)
        density = two_parameter_fermi(r_grid, R=4.5, a=0.55, N=40)
        particle_number = 4.0 * np.pi * np.trapezoid(r_grid**2 * density, r_grid)

        assert particle_number == pytest.approx(40.0, rel=1e-3)

    def test_two_parameter_fermi_density_callable_uses_same_helper(self):
        density = TwoParameterFermiDensity(R=5.0, a=0.5, N=40)
        r_grid = np.array([0.0, 2.5, 5.0])

        np.testing.assert_allclose(
            density(r_grid),
            two_parameter_fermi(r_grid, R=5.0, a=0.5, N=40),
        )

    def test_density_from_array_prepends_origin_and_clips_negative_values(self):
        radii = np.array([0.5, 1.0, 1.5, 2.0])
        densities = np.array([0.2, 0.1, -0.05, -0.2])
        density = density_from_array(radii, densities, clip_negative=True)

        values = density(np.array([0.0, 0.5, 1.5, 3.0]))

        assert values[0] == pytest.approx(densities[0])
        assert values[1] == pytest.approx(densities[0])
        assert values[2] == 0.0
        assert values[3] == 0.0
        assert density.r_max == pytest.approx(2.0)

    def test_density_from_array_requires_strictly_increasing_grid(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            density_from_array(np.array([0.0, 0.5, 0.5]), np.array([0.1, 0.2, 0.3]))


class TestJLMHelpers:
    def test_fermi_energy_matches_notebook_reference_value(self):
        assert jlm.fermi_energy_MeV(0.16) == pytest.approx(-24.8448)

    def test_fermi_energy_talys_low_energy_blend_matches_reference_formula(self):
        rho = np.array([0.02, 0.08, 0.16])
        energy = 10.0
        xfr1 = 1.0 / (1.0 + np.exp((energy - 9.0) / 2.0))
        eps1 = -22.0 - rho * (298.52 - 3760.23 * rho + 12435.82 * rho**2)
        eps2 = rho * (-510.8 + 3222.0 * rho - 6250.0 * rho**2)
        expected = eps1 * xfr1 + eps2 * (1.0 - xfr1)

        np.testing.assert_allclose(
            jlm.fermi_energy_MeV(
                rho,
                projectile_energy_MeV=energy,
                low_energy_offset=jlm.EPS_F_LOW_OFFSET,
                low_energy_coeffs=jlm.EPS_F_LOW_coeffs,
                blend_energy=jlm.EPS_F_BLEND_ENERGY,
                blend_width=jlm.EPS_F_BLEND_WIDTH,
            ),
            expected,
        )

    def test_effective_mass_matches_finite_difference_of_v0(self):
        rho = np.array([0.04, 0.10, 0.16])
        energy = 30.0
        step = 1e-5

        finite_difference = (
            jlm.V0(rho, energy + step) - jlm.V0(rho, energy - step)
        ) / (2.0 * step)

        np.testing.assert_allclose(
            jlm.eff_mass(rho, energy),
            1.0 - finite_difference,
            rtol=1e-7,
            atol=1e-9,
        )

    def test_delta_c_linear_matches_documented_expression(self):
        rho = np.array([0.05, 0.16])
        energy = 30.0
        v_c = 8.0

        np.testing.assert_allclose(
            jlm.Delta_C(rho, energy, v_c, linear=True),
            (jlm.eff_mass(rho, energy) - 1.0) * v_c,
        )

    def test_imaginary_terms_are_finite_at_fermi_energy(self):
        rho = np.array([0.04, 0.10, 0.16])
        e_fermi = jlm.fermi_energy_MeV(rho)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w0 = jlm.W0(rho, e_fermi, e_fermi)
            w1 = jlm.W1(rho, e_fermi, e_fermi)
            w0_hi = jlm.W0(rho, e_fermi + ENERGY_DELTA_MEV, e_fermi)
            w0_lo = jlm.W0(rho, e_fermi - ENERGY_DELTA_MEV, e_fermi)
            w1_hi = jlm.W1(rho, e_fermi + ENERGY_DELTA_MEV, e_fermi)
            w1_lo = jlm.W1(rho, e_fermi - ENERGY_DELTA_MEV, e_fermi)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings
        assert np.all(np.isfinite(w0))
        assert np.all(np.isfinite(w1))
        assert np.all(np.isfinite(w0_hi))
        assert np.all(np.isfinite(w0_lo))
        assert np.all(np.isfinite(w1_hi))
        assert np.all(np.isfinite(w1_lo))
        # W0 scales with 1/(E-E_F)^2 damping and is much more suppressed than W1.
        assert np.all(np.abs(w0) < W0_FERMI_TOLERANCE)
        assert np.all(np.abs(w1) < W1_FERMI_TOLERANCE)


class TestJLMPotentials:
    def test_potential_jlm_neutron_branch_matches_manual_formula(self):
        rho = np.array([0.03, 0.08, 0.16])
        energy = 30.0
        target = (48, 20)
        alpha = (target[0] - 2 * target[1]) / target[0]
        e_fermi = jlm.fermi_energy_MeV(rho)
        d_damping = 450.0
        f_damping = 0.7

        expected_v = jlm.V0(rho, energy) + alpha * jlm.V1(rho, energy, e_fermi)
        expected_w = jlm.W0(
            rho,
            energy,
            e_fermi,
            damping=d_damping,
        ) + alpha * jlm.W1(
            rho,
            energy,
            e_fermi,
            damping=f_damping,
        )

        actual_v, actual_w = jlm.potential_JLM(
            np.linspace(0.0, 6.0, len(rho)),
            rho,
            (1, 0),
            target,
            energy,
            D_damping=d_damping,
            F_damping=f_damping,
        )

        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_w, expected_w, rtol=1e-12, atol=1e-12)

    def test_potential_jlm_proton_branch_uses_coulomb_shift_and_lane_sign(self):
        rho = np.array([0.03, 0.08, 0.16])
        energy = 30.0
        target = (48, 20)
        alpha = (target[0] - 2 * target[1]) / target[0]
        radius_c = 1.2 * target[0] ** (1 / 3)
        v_c = 6.0 * target[1] * ALPHA * HBARC / (5 * radius_c)
        v_c_grid = np.full_like(rho, v_c)
        e_eff = energy - v_c
        e_fermi = jlm.fermi_energy_MeV(rho)
        delta_c = jlm.Delta_C(rho, energy, v_c_grid, linear=True)
        d_damping = 450.0
        f_damping = 0.7

        expected_v = (
            jlm.V0(rho, energy)
            + delta_c
            - alpha
            * jlm.V1(
                rho,
                e_eff,
                e_fermi,
            )
        )
        expected_w = jlm.W0(
            rho,
            e_eff,
            e_fermi,
            damping=d_damping,
        ) - alpha * jlm.W1(
            rho,
            e_eff,
            e_fermi,
            damping=f_damping,
        )

        actual_v, actual_w = jlm.potential_JLM(
            np.linspace(0.0, 6.0, len(rho)),
            rho,
            (1, 1),
            target,
            energy,
            V_C=v_c_grid,
            D_damping=d_damping,
            F_damping=f_damping,
        )

        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_w, expected_w, rtol=1e-12, atol=1e-12)

    def test_potential_jlm_talys_parameterization_matches_manual_formula(self):
        rgrid = np.linspace(0.0, 8.0, 5)
        rho_n = np.array([0.085, 0.080, 0.055, 0.015, 0.002])
        rho_p = np.array([0.075, 0.070, 0.045, 0.012, 0.0015])
        rho = rho_n + rho_p
        target = (208, 82)
        energy = 10.0
        alpha = (target[0] - 2 * target[1]) / target[0]

        from jitr.folding import ILDAFolder

        folder = ILDAFolder(r_max=float(rgrid.max()), n_quad=200)
        v_c = folder.V_coulomb(
            folder.interp_to_quad(rgrid, rho_p), mode="density", r_out=rgrid
        )
        e_eff = energy - v_c
        e_fermi = jlm.fermi_energy_MeV(
            rho,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_E_F,
            projectile_energy_MeV=energy,
            low_energy_offset=jlm.TALYS_PARAMETERIZATION.low_energy_E_F_offset,
            low_energy_coeffs=jlm.TALYS_PARAMETERIZATION.low_energy_E_F_coeffs,
            blend_energy=jlm.TALYS_PARAMETERIZATION.E_F_blend_energy,
            blend_width=jlm.TALYS_PARAMETERIZATION.E_F_blend_width,
        )

        expected_v = jlm.V0(
            rho,
            e_eff,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_A,
        ) - alpha * jlm.V1(
            rho,
            e_eff,
            e_fermi,
            coeffs_B=jlm.TALYS_PARAMETERIZATION.coeffs_B,
            coeffs_C=jlm.TALYS_PARAMETERIZATION.coeffs_C,
        )
        expected_w = jlm.W0(
            rho,
            e_eff,
            e_fermi,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_D,
            damping=jlm.TALYS_PARAMETERIZATION.D_damping,
        ) - alpha * jlm.W1(
            rho,
            e_eff,
            e_fermi,
            coeffs_F=jlm.TALYS_PARAMETERIZATION.coeffs_F,
            coeffs_A=jlm.TALYS_PARAMETERIZATION.coeffs_A,
            coeffs_C=jlm.TALYS_PARAMETERIZATION.coeffs_C,
            damping=jlm.TALYS_PARAMETERIZATION.F_damping,
        )

        actual_v, actual_w = jlm.potential_JLM(
            rgrid,
            rho,
            (1, 1),
            target,
            energy,
            V_C=v_c,
            parameterization="talys",
        )

        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_w, expected_w, rtol=1e-12, atol=1e-12)

    def test_potential_jlm_proton_requires_v_c(self):
        with pytest.raises(ValueError, match="V_C is required"):
            jlm.potential_JLM(
                np.linspace(0.0, 6.0, 3),
                np.array([0.03, 0.08, 0.16]),
                (1, 1),
                (208, 82),
                10.0,
            )

    def test_talys_parameterization_moves_surface_w1_denominator_away_from_zero(self):
        rgrid = np.linspace(0.0, 15.0, 1501)
        rho_n = two_parameter_fermi(rgrid, R=6.8, a=0.55, N=126)
        rho_p = two_parameter_fermi(rgrid, R=6.7, a=0.55, N=82)
        rho = rho_n + rho_p
        energy = 10.0
        target = (208, 82)
        radius_c = 1.2 * target[0] ** (1 / 3)
        v_c_const = 6.0 * target[1] * ALPHA * HBARC / (5 * radius_c)

        current_delta = energy - v_c_const - jlm.fermi_energy_MeV(rho)
        current_denom = 1.0 + jlm.F_DAMPING / current_delta

        from jitr.folding import ILDAFolder

        folder = ILDAFolder(r_max=float(rgrid.max()), n_quad=400)
        v_c_local = folder.V_coulomb(
            folder.interp_to_quad(rgrid, rho_p),
            mode="density",
            r_out=rgrid,
        )
        talys_delta = energy - v_c_local
        talys_delta = talys_delta - jlm.fermi_energy_MeV(
            rho,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_E_F,
            projectile_energy_MeV=energy,
            low_energy_offset=jlm.TALYS_PARAMETERIZATION.low_energy_E_F_offset,
            low_energy_coeffs=jlm.TALYS_PARAMETERIZATION.low_energy_E_F_coeffs,
            blend_energy=jlm.TALYS_PARAMETERIZATION.E_F_blend_energy,
            blend_width=jlm.TALYS_PARAMETERIZATION.E_F_blend_width,
        )
        talys_denom = 1.0 + jlm.TALYS_PARAMETERIZATION.F_damping / talys_delta

        surface = (rgrid > 6.5) & (rgrid < 9.0)
        assert np.min(np.abs(current_denom[surface])) < 0.25
        assert np.min(np.abs(talys_denom[surface])) > 0.5

    def test_potential_jlmb_matches_manual_folded_expression_for_neutrons(self):
        from jitr.folding import ILDAFolder

        folder = ILDAFolder(r_max=14.0, n_quad=250)
        rho_n_q = TwoParameterFermiDensity(R=3.7, a=0.52, N=28)(folder.r_q)
        rho_p_q = TwoParameterFermiDensity(R=3.5, a=0.50, N=20)(folder.r_q)
        rho_q = rho_n_q + rho_p_q
        alpha_q = (rho_n_q - rho_p_q) / rho_q
        energy = 30.0
        e_fermi = jlm.fermi_energy_MeV(rho_q)

        expected_vnm = jlm.V0(rho_q, energy) + alpha_q * jlm.V1(rho_q, energy, e_fermi)
        expected_wnm = jlm.W0(rho_q, energy, e_fermi) + alpha_q * jlm.W1(
            rho_q,
            energy,
            e_fermi,
        )
        expected_v = folder.gaussian_fold(expected_vnm, t=1.25, r_out=folder.r_q)
        expected_w = folder.gaussian_fold(expected_wnm, t=1.35, r_out=folder.r_q)

        actual_v, actual_w = jlm.potential_JLMB(
            folder,
            rho_n_q,
            rho_p_q,
            (1, 0),
            (48, 20),
            energy,
            r_out=folder.r_q,
        )

        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_w, expected_w, rtol=1e-12, atol=1e-12)

    def test_potential_jlmb_talys_parameterization_matches_manual_formula(self):
        from jitr.folding import ILDAFolder

        folder = ILDAFolder(r_max=14.0, n_quad=250)
        rho_n_q = TwoParameterFermiDensity(R=6.8, a=0.55, N=126)(folder.r_q)
        rho_p_q = TwoParameterFermiDensity(R=6.7, a=0.55, N=82)(folder.r_q)
        rho_q = rho_n_q + rho_p_q
        alpha_q = (rho_n_q - rho_p_q) / rho_q
        energy = 10.0
        e_fermi = jlm.fermi_energy_MeV(
            rho_q,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_E_F,
            projectile_energy_MeV=energy,
            low_energy_offset=jlm.TALYS_PARAMETERIZATION.low_energy_E_F_offset,
            low_energy_coeffs=jlm.TALYS_PARAMETERIZATION.low_energy_E_F_coeffs,
            blend_energy=jlm.TALYS_PARAMETERIZATION.E_F_blend_energy,
            blend_width=jlm.TALYS_PARAMETERIZATION.E_F_blend_width,
        )
        v_c_q = folder.V_coulomb(rho_p_q, mode="density")
        e_eff_q = energy - v_c_q

        expected_vnm = jlm.V0(
            rho_q,
            e_eff_q,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_A,
        ) - alpha_q * jlm.V1(
            rho_q,
            e_eff_q,
            e_fermi,
            coeffs_B=jlm.TALYS_PARAMETERIZATION.coeffs_B,
            coeffs_C=jlm.TALYS_PARAMETERIZATION.coeffs_C,
        )
        expected_wnm = jlm.W0(
            rho_q,
            e_eff_q,
            e_fermi,
            coeffs=jlm.TALYS_PARAMETERIZATION.coeffs_D,
            damping=jlm.TALYS_PARAMETERIZATION.D_damping,
        ) - alpha_q * jlm.W1(
            rho_q,
            e_eff_q,
            e_fermi,
            coeffs_F=jlm.TALYS_PARAMETERIZATION.coeffs_F,
            coeffs_A=jlm.TALYS_PARAMETERIZATION.coeffs_A,
            coeffs_C=jlm.TALYS_PARAMETERIZATION.coeffs_C,
            damping=jlm.TALYS_PARAMETERIZATION.F_damping,
        )
        expected_v = folder.gaussian_fold(expected_vnm, t=1.25, r_out=folder.r_q)
        expected_w = folder.gaussian_fold(expected_wnm, t=1.35, r_out=folder.r_q)

        actual_v, actual_w = jlm.potential_JLMB(
            folder,
            rho_n_q,
            rho_p_q,
            (1, 1),
            (208, 82),
            energy,
            V_C=v_c_q,
            parameterization="talys",
            r_out=folder.r_q,
        )

        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_w, expected_w, rtol=1e-12, atol=1e-12)

    def test_potential_jlmb_proton_requires_v_c(self):
        from jitr.folding import ILDAFolder

        folder = ILDAFolder(r_max=14.0, n_quad=100)
        rho_n_q = np.full_like(folder.r_q, 0.08)
        rho_p_q = np.full_like(folder.r_q, 0.07)

        with pytest.raises(ValueError, match="V_C is required"):
            jlm.potential_JLMB(
                folder,
                rho_n_q,
                rho_p_q,
                (1, 1),
                (208, 82),
                10.0,
            )

    def test_potential_jlm_invalid_projectile_raises(self):
        with pytest.raises(ValueError, match="Projectile must be"):
            jlm.potential_JLM(
                np.array([0.0]),
                np.array([0.16]),
                (2, 1),
                (40, 20),
                30.0,
            )
