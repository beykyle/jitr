"""Mass-table accessors and simple derived nuclear-mass observables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AMU, MASS_E, MASS_N, MASS_P

MassResult = tuple[float, float | None]

# Mass table database initialized at import time from ``utils.__init__``.
__MASS_MODELS__: list[str] = []
__MASS_DB__: pd.DataFrame | None = None
__MASS_DIR__ = (
    Path(__file__).parent.resolve() / Path("./../../data/mass_data/")
).resolve()


def init_mass_db() -> None:
    """Load the packaged mass tables into memory if needed."""
    global __MASS_MODELS__
    global __MASS_DIR__
    global __MASS_DB__

    if __MASS_DIR__ is None:
        __MASS_DIR__ = (
            Path(__file__).parent.resolve() / Path("./../../data/mass_data/")
        ).resolve()
        assert __MASS_DIR__.is_dir()

    if __MASS_DB__ is None:
        __MASS_MODELS__ = ["BMA"]

        mass_excess_table = pd.read_csv(
            __MASS_DIR__ / "mass_BMA.dat",
            header=None,
            names=["Z", "A", "BMA", "err_BMA"],
            dtype={
                "A": "int",
                "Z": "int",
                "BMA": "float",
                "Uncertainty_BMA": "float",
            },
            sep=r"\s+",
        )

        for file in __MASS_DIR__.iterdir():
            if file.is_file() and file.suffix == ".dat" and "BMA" not in file.name:
                name = str(file.with_suffix("").name).removeprefix("mass_")
                __MASS_MODELS__.append(name)
                mass_excess_table = pd.merge(
                    pd.read_csv(
                        file,
                        header=None,
                        names=["Z", "A", f"{name}", f"err_{name}"],
                        dtype={
                            "A": "int",
                            "Z": "int",
                            f"{name}": "float",
                            f"err_{name}": "float",
                        },
                        sep=r"\s+",
                    ),
                    mass_excess_table,
                    on=["A", "Z"],
                    how="outer",
                )

        __MASS_DB__ = mass_excess_table


def mass_db_row(A: int, Z: int) -> pd.DataFrame:
    """Return the matching mass-table row for a nuclide."""
    if __MASS_DB__ is None:
        init_mass_db()
    assert __MASS_DB__ is not None
    return __MASS_DB__[(__MASS_DB__["A"] == A) & (__MASS_DB__["Z"] == Z)]


def mass_models() -> list[str]:
    """Return the list of available mass models."""
    if __MASS_MODELS__ is None:
        init_mass_db()
    assert __MASS_MODELS__ is not None
    return __MASS_MODELS__


def mass_excess(A: int, Z: int, model: str = "ame2020") -> MassResult:
    """Return the mass excess and uncertainty in MeV.

    The uncertainty is ``None`` when the mass table does not include an
    uncertainty for the requested model, or when the nuclide is not found.
    """
    row = mass_db_row(A, Z)
    if row.empty:
        return np.nan, None
    value = float(row[model].iloc[0])
    raw_err = row[f"err_{model}"].iloc[0]
    err: float | None = (
        None if (raw_err is None or np.isnan(raw_err)) else float(raw_err)
    )
    return value, err


def mass(A: int, Z: int, **kwargs: str) -> MassResult:
    """Return the nuclear mass and uncertainty in MeV/c^2."""
    excess = mass_excess(A, Z, **kwargs)
    return A * AMU + excess[0] - Z * MASS_E, excess[1]


def binding_energy(A: int, Z: int, **kwargs: str) -> MassResult:
    """Return the total binding energy and uncertainty in MeV."""
    nuclear_mass = mass(A, Z, **kwargs)
    return -nuclear_mass[0] + Z * MASS_P + (A - Z) * MASS_N, nuclear_mass[1]


def _scale_or_none(value: float | None, scale: float) -> float | None:
    """Multiply ``value`` by ``scale``, or return ``None`` if ``value`` is ``None``."""
    return None if value is None else value * scale


def _combine_uncertainties(a: float | None, b: float | None) -> float | None:
    """Return quadrature sum of two uncertainties, or ``None`` if either is missing."""
    if a is None or b is None:
        return None
    return float(np.sqrt(a**2 + b**2))


def neutron_separation_energy(A: int, Z: int, **kwargs: str) -> MassResult:
    """Return the one-neutron separation energy and uncertainty in MeV."""
    mf = mass(A - 1, Z, **kwargs)
    m0 = mass(A, Z, **kwargs)
    return mf[0] + MASS_N - m0[0], _combine_uncertainties(mf[1], m0[1])


def proton_separation_energy(A: int, Z: int, **kwargs: str) -> MassResult:
    """Return the one-proton separation energy and uncertainty in MeV."""
    mf = mass(A - 1, Z - 1, **kwargs)
    m0 = mass(A, Z, **kwargs)
    return mf[0] + MASS_P - m0[0], _combine_uncertainties(mf[1], m0[1])


def neutron_fermi_energy(A: int, Z: int, **kwargs: str) -> MassResult:
    """Return the neutron Fermi-energy estimate and uncertainty in MeV."""
    separation_a = neutron_separation_energy(A, Z, **kwargs)
    separation_a1 = neutron_separation_energy(A + 1, Z, **kwargs)
    return -0.5 * (separation_a[0] + separation_a1[0]), _combine_uncertainties(
        _scale_or_none(separation_a[1], 0.5),
        _scale_or_none(separation_a1[1], 0.5),
    )


def proton_fermi_energy(A: int, Z: int, **kwargs: str) -> MassResult:
    """Return the proton Fermi-energy estimate and uncertainty in MeV."""
    separation_a = proton_separation_energy(A, Z, **kwargs)
    separation_a1 = proton_separation_energy(A + 1, Z + 1, **kwargs)
    return -0.5 * (separation_a[0] + separation_a1[0]), _combine_uncertainties(
        _scale_or_none(separation_a[1], 0.5),
        _scale_or_none(separation_a1[1], 0.5),
    )
