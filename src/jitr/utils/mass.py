import numpy as np
import pandas as pd
from pathlib import Path

from .constants import MASS_N, MASS_P, MASS_E, AMU

# mass table DB initialized at import
__MASS_MODELS__ = []
__MASS_DB__ = None
__MASS_DIR__ = (
    Path(__file__).parent.resolve() / Path("./../../data/mass_data/")
).resolve()


def init_mass_db():
    r"""
    Should be called once during import to load the AME mass table into memory
    """
    global __MASS_MODELS__
    global __MASS_DIR__
    global __MASS_DB__

    if __MASS_DIR__ is None:
        __MASS_DIR__ = (
            Path(__file__).parent.resolve() / Path("./../../data/mass_data/")
        ).resolve()
        assert __MASS_DIR__.is_file()

    if __MASS_DB__ is None:
        __MASS_MODELS__ = ["BMA"]

        # Read BMA file first
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

        # sequentially merge each other file along A and Z
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


def mass_db_row(A, Z):
    return __MASS_DB__[(__MASS_DB__["A"] == A) & (__MASS_DB__["Z"] == Z)]


def mass_excess(A, Z, model="ame2020"):
    row = mass_db_row(A, Z)
    if row.empty:
        return np.nan, np.nan
    else:
        return float(row[model].iloc[0]), float(row[f"err_{model}"].iloc[0])


def mass(A, Z, **kwargs):
    excess = mass_excess(A, Z, **kwargs)
    return A * AMU + excess[0] - Z * MASS_E, excess[1]


def binding_energy(A, Z, **kwargs):
    m = mass(A, Z, **kwargs)
    return -m[0] + Z * MASS_P + (A - Z) * MASS_N, m[1]


def neutron_separation_energy(A, Z, **kwargs):
    mf = mass(A - 1, Z, **kwargs)
    m0 = mass(A, Z, **kwargs)
    return mf[0] + MASS_N - m0[0], np.sqrt(mf[1] ** 2 + m0[1] ** 2)


def proton_separation_energy(A, Z, **kwargs):
    mf = mass(A - 1, Z - 1, **kwargs)
    m0 = mass(A, Z, **kwargs)
    return mf[0] + MASS_P - m0[0], np.sqrt(mf[1] ** 2 + m0[1] ** 2)


def neutron_fermi_energy(A, Z, **kwargs):
    SnA = neutron_separation_energy(A, Z, **kwargs)
    SnA1 = neutron_separation_energy(A + 1, Z, **kwargs)
    return -0.5 * (SnA[0] + SnA1[0]), 0.5 * np.sqrt(SnA[1] ** 2 + SnA1[1] ** 2)


def proton_fermi_energy(A, Z, **kwargs):
    SpA = proton_separation_energy(A, Z, **kwargs)
    SpA1 = proton_separation_energy(A + 1, Z + 1, **kwargs)
    return -0.5 * (SpA[0] + SpA1[0]), 0.5 * np.sqrt(SpA[1] ** 2 + SpA1[1] ** 2)
