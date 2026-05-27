"""Tabulated proton and neutron density accessors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import periodictable
from numpy.typing import NDArray

from ..folding.density import density_from_array

DensityResult = tuple[np.ndarray, np.ndarray]

DEFAULT_DENSITY_MODEL = "d1m"

# Density database initialized at import time from ``utils.__init__``.
__DENSITY_MODELS__: list[str] = []
__DENSITY_DB__: dict[str, dict[tuple[int, int], DensityTable]] | None = None
__DENSITY_DIR__ = (
    Path(__file__).parent.resolve() / Path("./../../data/densities/")
).resolve()


@dataclass(frozen=True)
class DensityTable:
    """Tabulated proton and neutron densities for a single nuclide.

    Args:
        A: Atomic mass number.
        Z: Atomic number.
        model: Density-model identifier.
        symbol: Chemical symbol used by the packaged table.
        dr: Nominal radial-grid spacing in fm.
        radial_grid: Tabulated radial grid in fm.
        proton_density_grid: Proton density sampled on ``radial_grid``.
        neutron_density_grid: Neutron density sampled on ``radial_grid``.
    """

    A: int
    Z: int
    model: str
    symbol: str
    dr: float
    radial_grid: NDArray[np.float64]
    proton_density_grid: NDArray[np.float64]
    neutron_density_grid: NDArray[np.float64]
    _proton_density_interp: Callable[[np.ndarray], np.ndarray] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _neutron_density_interp: Callable[[np.ndarray], np.ndarray] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def _proton_interpolator(self) -> Callable[[np.ndarray], np.ndarray]:
        interp = self._proton_density_interp
        if interp is None:
            interp = density_from_array(self.radial_grid, self.proton_density_grid)
            object.__setattr__(self, "_proton_density_interp", interp)
        return cast(Callable[[np.ndarray], np.ndarray], interp)

    def _neutron_interpolator(self) -> Callable[[np.ndarray], np.ndarray]:
        interp = self._neutron_density_interp
        if interp is None:
            interp = density_from_array(self.radial_grid, self.neutron_density_grid)
            object.__setattr__(self, "_neutron_density_interp", interp)
        return cast(Callable[[np.ndarray], np.ndarray], interp)

    @property
    def N(self) -> int:
        """Return the neutron number."""
        return self.A - self.Z

    def densities(self, r: float | np.ndarray) -> DensityResult:
        """Return proton and neutron densities on the requested radial grid."""
        return self.proton_density(r), self.neutron_density(r)

    def proton_density(self, r: float | np.ndarray) -> np.ndarray:
        """Return the proton density on the requested radial grid."""
        return self._proton_interpolator()(np.asarray(r, dtype=float))

    def neutron_density(self, r: float | np.ndarray) -> np.ndarray:
        """Return the neutron density on the requested radial grid."""
        return self._neutron_interpolator()(np.asarray(r, dtype=float))

    def matter_density(self, r: float | np.ndarray) -> np.ndarray:
        """Return the total matter density on the requested radial grid."""
        return self.proton_density(r) + self.neutron_density(r)


def _parse_density_file(path: Path, model: str) -> list[DensityTable]:
    """Parse all nuclide blocks from a packaged density table."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    symbol = path.stem
    tables: list[DensityTable] = []
    i = 0

    while i < len(lines):
        header = lines[i].split()
        if len(header) != 4:
            raise ValueError(f"Malformed density header in {path}: {lines[i]!r}")

        Z = int(header[0])
        A = int(header[1])
        n_points = int(header[2])
        dr = float(header[3])
        i += 1

        block_lines = lines[i : i + n_points]
        if len(block_lines) != n_points:
            raise ValueError(f"Density block for A={A}, Z={Z} in {path} is truncated.")

        block = np.array(
            [[float(value) for value in line.split()] for line in block_lines],
            dtype=float,
        )
        if block.shape[1] < 7:
            raise ValueError(f"Density block for A={A}, Z={Z} in {path} is malformed.")

        radial_grid = np.ascontiguousarray(block[:, 0], dtype=float)
        proton_density_grid = np.ascontiguousarray(block[:, 1], dtype=float)
        neutron_density_grid = np.ascontiguousarray(block[:, 6], dtype=float)
        expected_symbol = str(periodictable.elements[Z])
        if expected_symbol != symbol:
            raise ValueError(
                f"Density table {path} encodes Z={Z}, "
                f"which maps to {expected_symbol!r}, not {symbol!r}."
            )
        if not np.all(np.diff(radial_grid) > 0.0):
            raise ValueError(f"Radial grid for A={A}, Z={Z} in {path} must increase.")

        tables.append(
            DensityTable(
                A=A,
                Z=Z,
                model=model,
                symbol=symbol,
                dr=dr,
                radial_grid=radial_grid,
                proton_density_grid=proton_density_grid,
                neutron_density_grid=neutron_density_grid,
            )
        )
        i += n_points

    return tables


def init_density_db() -> None:
    """Load the packaged density tables into memory if needed."""
    global __DENSITY_MODELS__
    global __DENSITY_DIR__
    global __DENSITY_DB__

    if __DENSITY_DIR__ is None:
        __DENSITY_DIR__ = (
            Path(__file__).parent.resolve() / Path("./../../data/densities/")
        ).resolve()
        assert __DENSITY_DIR__.is_dir()

    if __DENSITY_DB__ is None:
        density_db: dict[str, dict[tuple[int, int], DensityTable]] = {}
        __DENSITY_MODELS__ = []

        model_dirs = sorted(path for path in __DENSITY_DIR__.iterdir() if path.is_dir())
        for model_dir in model_dirs:
            __DENSITY_MODELS__.append(model_dir.name)
            tables: dict[tuple[int, int], DensityTable] = {}

            for path in sorted(model_dir.glob("*.rad")):
                for table in _parse_density_file(path, model=model_dir.name):
                    key = (table.A, table.Z)
                    if key in tables:
                        raise ValueError(
                            f"Duplicate density table for A={table.A}, Z={table.Z} "
                            f"in model {model_dir.name!r}."
                        )
                    tables[key] = table

            density_db[model_dir.name] = tables

        __DENSITY_DB__ = density_db


def density_models() -> list[str]:
    """Return the list of available density models."""
    if __DENSITY_DB__ is None:
        init_density_db()
    assert __DENSITY_DB__ is not None
    return __DENSITY_MODELS__.copy()


def density_targets(
    model: str = DEFAULT_DENSITY_MODEL,
    Z: int | None = None,
) -> list[tuple[int, int]]:
    """Return the available ``(A, Z)`` targets for a model."""
    tables = _model_density_tables(model)
    targets = sorted(tables)
    if Z is None:
        return targets
    return [target for target in targets if target[1] == Z]


def density_table(A: int, Z: int, model: str = DEFAULT_DENSITY_MODEL) -> DensityTable:
    """Return the tabulated density table for a nuclide."""
    tables = _model_density_tables(model)
    key = (A, Z)
    if key not in tables:
        raise KeyError(f"No density table for A={A}, Z={Z} in model {model!r}.")
    return tables[key]


def densities(
    A: int,
    Z: int,
    r: float | np.ndarray,
    model: str = DEFAULT_DENSITY_MODEL,
) -> DensityResult:
    """Return proton and neutron densities on the requested radial grid."""
    return density_table(A, Z, model=model).densities(r)


def proton_density(
    A: int,
    Z: int,
    r: float | np.ndarray,
    model: str = DEFAULT_DENSITY_MODEL,
) -> np.ndarray:
    """Return the proton density on the requested radial grid."""
    return density_table(A, Z, model=model).proton_density(r)


def neutron_density(
    A: int,
    Z: int,
    r: float | np.ndarray,
    model: str = DEFAULT_DENSITY_MODEL,
) -> np.ndarray:
    """Return the neutron density on the requested radial grid."""
    return density_table(A, Z, model=model).neutron_density(r)


def matter_density(
    A: int,
    Z: int,
    r: float | np.ndarray,
    model: str = DEFAULT_DENSITY_MODEL,
) -> np.ndarray:
    """Return the total matter density on the requested radial grid."""
    return density_table(A, Z, model=model).matter_density(r)


def _model_density_tables(model: str) -> dict[tuple[int, int], DensityTable]:
    """Return the loaded density tables for a model."""
    if __DENSITY_DB__ is None:
        init_density_db()
    assert __DENSITY_DB__ is not None
    if model not in __DENSITY_DB__:
        raise KeyError(f"Unknown density model {model!r}.")
    return __DENSITY_DB__[model]


__all__ = [
    "DEFAULT_DENSITY_MODEL",
    "DensityResult",
    "DensityTable",
    "densities",
    "density_models",
    "density_table",
    "density_targets",
    "init_density_db",
    "matter_density",
    "neutron_density",
    "proton_density",
]
