from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from numba import njit

from .constants import ALPHA, HBARC, MASS_N, MASS_P

# AME mass table DB initialized at import
__AME_DB__ = None
__AME_PATH__ = (
    Path(__file__).parent.resolve() / Path("../data/mass_1.mas20.txt")
).resolve()


def init_AME_db():
    r"""
    Should be called once during import to load the AME mass table into memory
    """
    global __AME_PATH__
    global __AME_DB__
    if __AME_PATH__ is None:
        __AME_PATH__ = (
            Path(__file__).parent.resolve() / Path("./../../data/mass_1.mas20.txt")
        ).resolve()
        assert __AME_PATH__.is_file()
    if __AME_DB__ is None:
        __AME_DB__ = pd.read_csv(__AME_PATH__, sep="\s+")


def get_AME_binding_energy(A, Z):
    r"""Calculates binding in MeV/c^2 given mass number, A, proton number, Z, by AME2020 lookup"""
    # look up nuclide in AME2020 table
    global __AME_DB__
    df = __AME_DB__
    mask = (df["A"] == A) & (df["Z"] == Z)
    if mask.any():
        # use AME if data exists
        # format is Eb/A [keV/nucleon]
        return float(df[mask]["BINDING_ENERGY/A"].iloc[0]) * A / 1e3
    return None


@njit
def semiempirical_binding_energy(A, Z):
    r"""Calculates binding in MeV/c^2 given mass number, A, proton number, Z, by semi-empriical
    mass formula"""
    N = A - Z
    delta = 0
    if N % 2 == 0 and Z % 2 == 0:
        delta = 12.0 / np.sqrt(A)
    elif N % 2 != 0 and Z % 2 != 0:
        delta = -12.0 / np.sqrt(A)

    Eb = (
        15.8 * A
        - 18.3 * A ** (2 / 3)
        - 0.714 * Z * (Z - 1) / (A ** (1 / 3))
        - 23.2 * (N - Z) ** 2 / A
        + delta
    )
    return Eb


def get_binding_energy(A, Z):
    r"""Calculates binding in MeV/c^2 given mass number, A, proton number, Z, by AME2020 lookup
    if possible or semi-empriical mass fomrula if not
    """
    Eb = get_AME_binding_energy(A, Z)
    if Eb is None:
        Eb = semiempirical_binding_energy(A, Z)
    return Eb


@njit
def mass(A, Z, Eb):
    r"""Calculates rest mass in MeV/c^2 given mass number, A, proton number, Z, and binding energy
    in MeV/c^2"""
    N = A - Z
    return Z * MASS_P + N * MASS_N - Eb


def kinematics(
    target: tuple,
    projectile: tuple,
    E_lab: float = None,
    E_com: float = None,
    binding_model: Callable[[int, int], float] = get_binding_energy,
):
    r"""Calculates the reduced mass, COM frame kinetic energy and wavenumber for a projectile (A,Z)
    scattering on a target nuclide (A,Z), with binding energies from binding_model, which defaults
    to lookup in AME2020 mass table. Uses relatavistic approximation of Ingemarsson, 1974:
    https://doi.org/10.1088/0031-8949/9/3/004
    Parameters:
        t : target (A,Z)
        p : projectile (A,Z)
        E_lab: bombarding energy in the lab frame [MeV]. Either E_lab or E_com must be provided,
        not both.
        E_com: bombarding energy in the com frame [MeV]. Either E_lab or E_com must be provided,
        not both.
        binding_model : optional callable taking in (A,Z) and returning binding energy in [MeV/c^2],
                        defaults to lookup in AME2020, and semi-empirical mass formula if not
                        available there
    Returns:
        mu (float) : reduced mass in MeV/c^2
        E_com (float) : center-of-mass frame energy in MeV
        k (float) : center-of-mass frame wavenumber in fm^-1
    """
    Eb_target = binding_model(*target)
    Eb_projectile = binding_model(*projectile)
    m_t = mass(*target, Eb_target)
    m_p = mass(*projectile, Eb_projectile)

    if E_lab is None:
        return_Elab = True
        assert E_com is not None
        E_com = np.fabs(E_com)
        E_lab = (m_t + m_p) / m_t * E_com
    else:
        return_Elab = False
        assert E_com is None
        E_lab = np.fabs(E_lab)
        E_com = m_t / (m_t + m_p) * E_lab

    Ep = E_com + m_p

    # relativisitic correction from A. Ingemarsson 1974, Eqs. 17 & 20
    k = (
        m_t
        * np.sqrt(E_lab * (E_lab + 2 * m_p))
        / np.sqrt((m_t + m_p) ** 2 + 2 * m_t * E_lab)
        / HBARC
    )
    mu = k**2 * Ep / (Ep**2 - m_p * m_p) * HBARC**2
    k_C = ALPHA * projectile[1] * target[1] * mu / HBARC
    eta = k_C / k

    if return_Elab:
        return mu, E_lab, k, eta
    else:
        return mu, E_com, k, eta


@njit
def classical_kinematics(mass_target, mass_projectile, E_lab, Q, Zz):
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    E_com = mass_target / (mass_target + mass_projectile) * E_lab
    k = np.sqrt(2 * (E_com + Q) * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return mu, E_com, k, eta
