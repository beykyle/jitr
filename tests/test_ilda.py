"""Tests for :mod:`jitr.folding.folding`."""

from __future__ import annotations

import numpy as np
import pytest

from jitr.folding import ILDAFolder


class TestILDAFolder:
    def test_integrate_z_and_rms_radius_for_gaussian_density(self):
        folder = ILDAFolder(r_max=18.0, n_quad=400)
        rho0 = 0.11
        a = 2.1
        rho_q = rho0 * np.exp(-((folder.r_q / a) ** 2))

        expected_z = rho0 * np.pi**1.5 * a**3
        expected_rms = np.sqrt(1.5) * a

        assert folder.integrate(np.ones_like(folder.r_q)) == pytest.approx(folder.r_max)
        assert folder.Z_from_density(rho_q) == pytest.approx(expected_z, rel=1e-6)
        assert folder.rms_radius(rho_q) == pytest.approx(expected_rms, rel=1e-6)

    def test_rms_radius_requires_positive_norm(self):
        folder = ILDAFolder()
        with pytest.raises(ValueError, match="density integrates"):
            folder.rms_radius(np.zeros_like(folder.r_q))

    def test_v_coulomb_uniform_sphere_matches_analytic_formula(self):
        folder = ILDAFolder(r_max=12.0, n_quad=300)
        radius = 4.0
        rho_q = np.where(folder.r_q <= radius, 0.08, 0.0)
        r_out = np.array([0.0, 2.0, 6.0])

        potential = folder.V_coulomb(
            rho_q,
            mode="uniform_sphere",
            R_C=radius,
            r_out=r_out,
        )
        z_value = folder.Z_from_density(rho_q)
        ze2 = z_value * folder.e2
        expected = np.array(
            [
                3.0 * ze2 / (2.0 * radius),
                (ze2 / (2.0 * radius)) * (3.0 - (2.0 / radius) ** 2),
                ze2 / 6.0,
            ]
        )

        np.testing.assert_allclose(potential, expected, rtol=1e-12, atol=1e-12)

    def test_v_coulomb_exchange_lowers_potential(self):
        folder = ILDAFolder(r_max=12.0, n_quad=300)
        rho_q = 0.07 * np.exp(-((folder.r_q / 3.0) ** 2))

        without_exchange = folder.V_coulomb(rho_q, mode="density")
        with_exchange = folder.V_coulomb(
            rho_q,
            mode="density",
            include_exchange=True,
        )

        assert np.all(with_exchange < without_exchange)

    def test_gaussian_fold_matches_gaussian_analytic_result(self):
        folder = ILDAFolder(r_max=18.0, n_quad=500)
        rho0 = -40.0
        a = 1.8
        t = 1.2
        u_q = rho0 * np.exp(-((folder.r_q / a) ** 2))
        r_out = np.linspace(0.0, 6.0, 31)

        folded = folder.gaussian_fold(u_q, t=t, r_out=r_out)
        sigma2 = t**2 + a**2
        expected = rho0 * (a**2 / sigma2) ** 1.5 * np.exp(-(r_out**2) / sigma2)

        np.testing.assert_allclose(folded, expected, rtol=2e-5, atol=1e-6)

    def test_gaussian_fold_invalid_t_raises(self):
        folder = ILDAFolder()
        with pytest.raises(ValueError, match="positive"):
            folder.gaussian_fold(np.ones_like(folder.r_q), t=0.0)
