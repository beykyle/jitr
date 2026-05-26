"""Tests for the packaged tabulated density helpers under :mod:`jitr.utils`."""

from __future__ import annotations

import numpy as np
import pytest

from jitr.utils import density


class TestDensityUtilities:
    def test_density_models(self):
        assert set(density.density_models()) == {"bskg3", "d1m"}

    def test_density_targets_parse_multiple_blocks(self):
        oxygen_targets = density.density_targets(model="bskg3", Z=8)
        assert (16, 8) in oxygen_targets
        assert (17, 8) in oxygen_targets

    def test_density_table_metadata_and_normalization(self):
        table = density.density_table(16, 8, model="bskg3")
        proton_number = (
            4.0
            * np.pi
            * np.trapezoid(
                table.radial_grid**2 * table.proton_density_grid,
                table.radial_grid,
            )
        )
        neutron_number = (
            4.0
            * np.pi
            * np.trapezoid(
                table.radial_grid**2 * table.neutron_density_grid,
                table.radial_grid,
            )
        )

        assert table.symbol == "O"
        assert table.N == 8
        assert proton_number == pytest.approx(8.0, rel=1e-4)
        assert neutron_number == pytest.approx(8.0, rel=1e-4)

    def test_density_interpolation_matches_tabulated_values(self):
        table = density.density_table(16, 8, model="d1m")
        proton, neutron = density.densities(16, 8, table.radial_grid, model="d1m")

        np.testing.assert_allclose(proton, table.proton_density_grid)
        np.testing.assert_allclose(neutron, table.neutron_density_grid)
        np.testing.assert_allclose(
            density.matter_density(16, 8, table.radial_grid, model="d1m"),
            proton + neutron,
        )

    def test_density_zero_outside_tabulated_range(self):
        table = density.density_table(16, 8, model="bskg3")
        beyond = np.array([table.radial_grid[-1] + table.dr])

        assert table.proton_density(beyond)[0] == 0.0
        assert table.neutron_density(beyond)[0] == 0.0

    def test_density_missing_target_or_model_raises(self):
        with pytest.raises(KeyError):
            density.density_table(999, 8, model="bskg3")

        with pytest.raises(KeyError):
            density.density_table(16, 8, model="not-a-model")
