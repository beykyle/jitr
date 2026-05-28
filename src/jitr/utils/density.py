"""Analytic and tabulated proton/neutron density utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import periodictable
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

FloatArray: TypeAlias = NDArray[np.float64]
DensityResult: TypeAlias = tuple[FloatArray, FloatArray]
RadialDensity: TypeAlias = Callable[[float | ArrayLike], FloatArray]

DEFAULT_DENSITY_MODEL = "d1m"

# Density database initialized at import time from ``utils.__init__``.
__DENSITY_MODELS__: list[str] = []
__DENSITY_DB__: dict[str, dict[tuple[int, int], DensityTable]] | None = None
__DENSITY_DIR__ = (
    Path(__file__).parent.resolve() / Path("./../../data/densities/")
).resolve()


@dataclass(frozen=True)
class TwoParameterFermiDensity:
    """Callable two-parameter Fermi density profile.

    Args:
        R: Half-density radius in fm.
        a: Surface diffuseness in fm.
        rho0: Central density normalization. Mutually exclusive with ``N``.
        N: Integrated particle number. Mutually exclusive with ``rho0``.

    Raises:
        ValueError: If neither ``rho0`` nor ``N`` is provided.
    """

    R: float
    a: float
    rho0: float | None = None
    N: float | None = None

    def __post_init__(self) -> None:
        """Validate that at least one normalization is supplied."""
        if self.rho0 is None and self.N is None:
            raise ValueError("Must provide either `rho0` or `N`.")

    def __call__(self, r: float | ArrayLike) -> FloatArray:
        """Evaluate the density on a radial grid.

        Args:
            r: Radial coordinate or coordinates in fm.

        Returns:
            Density values sampled at ``r``.
        """

        return two_parameter_fermi(r, R=self.R, a=self.a, rho0=self.rho0, N=self.N)


def two_parameter_fermi(
    r: float | ArrayLike,
    R: float,
    a: float,
    rho0: float | None = None,
    N: float | None = None,
) -> FloatArray:
    """Return a two-parameter Fermi density.

    Args:
        r: Radial coordinate or coordinates in fm.
        R: Half-density radius in fm.
        a: Surface diffuseness in fm.
        rho0: Central density normalization. Mutually exclusive with ``N``.
        N: Integrated particle number used to normalize the profile.

    Returns:
        Density values sampled at ``r``.

    Raises:
        ValueError: If neither ``rho0`` nor ``N`` is provided.
    """

    r_array = np.asarray(r, dtype=float)
    shape = 1.0 / (1.0 + np.exp((r_array - R) / a))
    if rho0 is None:
        if N is None:
            raise ValueError("Must provide either `rho0` or `N`.")
        rmax = max(R + 20.0 * a, 25.0)
        radial_grid = np.linspace(0.0, rmax, 8001)
        shape_grid = 1.0 / (1.0 + np.exp((radial_grid - R) / a))
        norm = 4.0 * np.pi * np.trapezoid(radial_grid**2 * shape_grid, radial_grid)
        normalization = float(N / norm)
    else:
        normalization = float(rho0)
    return np.asarray(normalization * shape, dtype=float)


def density_from_array(
    r_array: float | ArrayLike,
    rho_array: float | ArrayLike,
    clip_negative: bool = True,
) -> RadialDensity:
    """Wrap a tabulated radial density as a callable interpolator.

    Args:
        r_array: Strictly increasing tabulated radii in fm.
        rho_array: Density values sampled on ``r_array``.
        clip_negative: Whether to clip interpolated negative values to zero.

    Returns:
        A callable density interpolator that returns zero outside the tabulated
        range and exposes ``r_max`` as the maximum tabulated radius.

    Raises:
        ValueError: If the tabulated arrays have different shapes or the radial
            grid is not strictly increasing.
    """

    radii = np.asarray(r_array, dtype=float)
    densities = np.asarray(rho_array, dtype=float)
    if radii.shape != densities.shape:
        raise ValueError("r_array and rho_array must have the same shape.")
    if not np.all(np.diff(radii) > 0):
        raise ValueError("r_array must be strictly increasing.")
    if radii[0] > 1e-12:
        radii = np.concatenate(([0.0], radii))
        densities = np.concatenate(([densities[0]], densities))

    interpolator = PchipInterpolator(radii, densities, extrapolate=False)
    r_max = float(radii[-1])

    def rho(r: float | ArrayLike) -> FloatArray:
        """Evaluate the interpolated density on a radial grid."""

        r_values = np.asarray(r, dtype=float)
        out = np.zeros_like(r_values)
        mask = (r_values >= 0.0) & (r_values <= r_max)
        if mask.any():
            out[mask] = interpolator(r_values[mask])
        if clip_negative:
            out = np.where(out < 0.0, 0.0, out)
        return np.asarray(out, dtype=float)

    rho.r_max = r_max  # type: ignore[attr-defined]
    return rho


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
    radial_grid: FloatArray
    proton_density_grid: FloatArray
    neutron_density_grid: FloatArray
    _proton_density_interp: RadialDensity | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _neutron_density_interp: RadialDensity | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def _proton_interpolator(self) -> RadialDensity:
        """Build the proton-density interpolator on first use."""

        interp = self._proton_density_interp
        if interp is None:
            interp = density_from_array(self.radial_grid, self.proton_density_grid)
            object.__setattr__(self, "_proton_density_interp", interp)
        return cast(RadialDensity, interp)

    def _neutron_interpolator(self) -> RadialDensity:
        """Build the neutron-density interpolator on first use."""

        interp = self._neutron_density_interp
        if interp is None:
            interp = density_from_array(self.radial_grid, self.neutron_density_grid)
            object.__setattr__(self, "_neutron_density_interp", interp)
        return cast(RadialDensity, interp)

    @property
    def N(self) -> int:
        """Return the neutron number."""

        return self.A - self.Z

    def densities(self, r: float | ArrayLike) -> DensityResult:
        """Return proton and neutron densities on the requested radial grid.

        Args:
            r: Radial coordinate or coordinates in fm.

        Returns:
            Tuple of proton and neutron density arrays.
        """

        return self.proton_density(r), self.neutron_density(r)

    def proton_density(self, r: float | ArrayLike) -> FloatArray:
        """Return the proton density on the requested radial grid.

        Args:
            r: Radial coordinate or coordinates in fm.

        Returns:
            Proton density values sampled at ``r``.
        """

        return self._proton_interpolator()(np.asarray(r, dtype=float))

    def neutron_density(self, r: float | ArrayLike) -> FloatArray:
        """Return the neutron density on the requested radial grid.

        Args:
            r: Radial coordinate or coordinates in fm.

        Returns:
            Neutron density values sampled at ``r``.
        """

        return self._neutron_interpolator()(np.asarray(r, dtype=float))

    def matter_density(self, r: float | ArrayLike) -> FloatArray:
        """Return the total matter density on the requested radial grid.

        Args:
            r: Radial coordinate or coordinates in fm.

        Returns:
            Matter density values sampled at ``r``.
        """

        return self.proton_density(r) + self.neutron_density(r)


def _parse_density_file(path: Path, model: str) -> list[DensityTable]:
    """Parse all nuclide blocks from a packaged density table.

    Args:
        path: Path to the packaged ``.rad`` file.
        model: Density-model identifier for the parsed tables.

    Returns:
        Parsed density-table records from ``path``.

    Raises:
        ValueError: If the file contents are malformed.
    """

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
    """Return the available nuclides for a density model.

    Args:
        model: Density-model identifier.
        Z: Optional atomic-number filter.

    Returns:
        Sorted ``(A, Z)`` target tuples available for ``model``.
    """

    tables = _model_density_tables(model)
    targets = sorted(tables)
    if Z is None:
        return targets
    return [target for target in targets if target[1] == Z]


def density_table(A: int, Z: int, model: str = DEFAULT_DENSITY_MODEL) -> DensityTable:
    """Return the tabulated density table for a nuclide.

    Args:
        A: Atomic mass number.
        Z: Atomic number.
        model: Density-model identifier.

    Returns:
        The packaged density table for ``(A, Z)``.

    Raises:
        KeyError: If no table exists for the requested nuclide/model.
    """

    tables = _model_density_tables(model)
    key = (A, Z)
    if key not in tables:
        raise KeyError(f"No density table for A={A}, Z={Z} in model {model!r}.")
    return tables[key]


def densities(
    A: int,
    Z: int,
    r: float | ArrayLike,
    model: str = DEFAULT_DENSITY_MODEL,
) -> DensityResult:
    """Return proton and neutron densities on the requested radial grid.

    Args:
        A: Atomic mass number.
        Z: Atomic number.
        r: Radial coordinate or coordinates in fm.
        model: Density-model identifier.

    Returns:
        Tuple of proton and neutron density arrays.
    """

    return density_table(A, Z, model=model).densities(r)


def proton_density(
    A: int,
    Z: int,
    r: float | ArrayLike,
    model: str = DEFAULT_DENSITY_MODEL,
) -> FloatArray:
    """Return the proton density on the requested radial grid.

    Args:
        A: Atomic mass number.
        Z: Atomic number.
        r: Radial coordinate or coordinates in fm.
        model: Density-model identifier.

    Returns:
        Proton density values sampled at ``r``.
    """

    return density_table(A, Z, model=model).proton_density(r)


def neutron_density(
    A: int,
    Z: int,
    r: float | ArrayLike,
    model: str = DEFAULT_DENSITY_MODEL,
) -> FloatArray:
    """Return the neutron density on the requested radial grid.

    Args:
        A: Atomic mass number.
        Z: Atomic number.
        r: Radial coordinate or coordinates in fm.
        model: Density-model identifier.

    Returns:
        Neutron density values sampled at ``r``.
    """

    return density_table(A, Z, model=model).neutron_density(r)


def matter_density(
    A: int,
    Z: int,
    r: float | ArrayLike,
    model: str = DEFAULT_DENSITY_MODEL,
) -> FloatArray:
    """Return the total matter density on the requested radial grid.

    Args:
        A: Atomic mass number.
        Z: Atomic number.
        r: Radial coordinate or coordinates in fm.
        model: Density-model identifier.

    Returns:
        Matter density values sampled at ``r``.
    """

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
    "TwoParameterFermiDensity",
    "densities",
    "density_from_array",
    "density_models",
    "density_table",
    "density_targets",
    "init_density_db",
    "matter_density",
    "neutron_density",
    "proton_density",
    "two_parameter_fermi",
]
