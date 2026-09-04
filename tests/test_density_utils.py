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

    def test_density_interpolators_are_built_lazily(self):
        radial = np.array([0.0, 0.5, 1.0], dtype=float)
        proton_grid = np.array([0.1, 0.08, 0.04], dtype=float)
        neutron_grid = np.array([0.12, 0.09, 0.05], dtype=float)
        table = density.DensityTable(
            A=16,
            Z=8,
            model="test",
            symbol="O",
            dr=0.5,
            radial_grid=radial,
            proton_density_grid=proton_grid,
            neutron_density_grid=neutron_grid,
        )
        assert table._proton_density_interp is None
        assert table._neutron_density_interp is None
        _ = table.proton_density(np.array([0.0]))
        assert table._proton_density_interp is not None
        assert table._neutron_density_interp is None
        _ = table.neutron_density(np.array([0.0]))
        assert table._neutron_density_interp is not None

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

    def test_rad_to_npz_round_trip(self, tmp_path):
        rad_dir = tmp_path / "rad"
        rad_dir.mkdir()
        rng = np.random.default_rng(0)

        def block(Z, A, n, dr):
            lines = [f"{Z} {A} {n} {dr:.3f}"]
            for i in range(n):
                row = rng.uniform(0.0, 0.2, size=11)
                row[0] = i * dr
                lines.append(" ".join(f"{v:.5E}" for v in row))
            return lines, row

        lines_16, _ = block(8, 16, 12, 0.1)
        lines_17, _ = block(8, 17, 15, 0.1)
        (rad_dir / "O.rad").write_text("\n".join(lines_16 + lines_17) + "\n")

        tables = density.read_rad_file(rad_dir / "O.rad", model="toy")
        assert [(t.A, t.Z) for t in tables] == [(16, 8), (17, 8)]

        npz_path = tmp_path / "toy.npz"
        density.write_density_npz(tables, npz_path)
        loaded = density._load_density_npz(npz_path, model="toy")

        assert set(loaded) == {(16, 8), (17, 8)}
        for table in tables:
            got = loaded[(table.A, table.Z)]
            assert got.symbol == "O"
            assert got.dr == table.dr
            assert got.radial_grid.dtype == np.float64
            assert got.proton_density_grid.dtype == np.float64
            np.testing.assert_allclose(got.radial_grid, table.radial_grid, atol=1e-12)
            np.testing.assert_allclose(
                got.proton_density_grid, table.proton_density_grid, rtol=1e-6
            )
            np.testing.assert_allclose(
                got.neutron_density_grid, table.neutron_density_grid, rtol=1e-6
            )

    def test_write_density_npz_rejects_nonuniform_grid(self, tmp_path):
        table = density.DensityTable(
            A=16,
            Z=8,
            model="toy",
            symbol="O",
            dr=0.1,
            radial_grid=np.array([0.0, 0.1, 0.3]),
            proton_density_grid=np.ones(3),
            neutron_density_grid=np.ones(3),
        )
        with pytest.raises(ValueError, match="uniform"):
            density.write_density_npz([table], tmp_path / "bad.npz")
