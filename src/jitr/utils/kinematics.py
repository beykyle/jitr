from collections.abc import Callable

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

from .constants import ALPHA, HBARC, MASS_N, MASS_P

# AME mass table DB initialized at import
__AME_DB__ = None
__AME_PATH__ = (
    Path(__file__).parent.resolve() / Path("./../../data/mass_1.mas20.txt")
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


def mass(A, Z, Eb=None):
    r"""Calculates rest mass in MeV/c^2 given mass number, A, proton number, Z, and binding energy
    in MeV/c^2"""
    if Eb is None:
        Eb = get_binding_energy(A, Z)
    N = A - Z
    return Z * MASS_P + N * MASS_N - Eb


def semi_relativistic_kinematics(
    mass_target,
    mass_projectile,
    Elab,
    Zz=0,
    Q=0,
):
    r"""Calculates the CM frame kinetic energy and wavenumber for a projectile scattering on a
    target nuclide using the relatavistic approximation of Ingemarsson, 1974:
    https://doi.org/10.1088/0031-8949/9/3/004
    Parameters:
        t : target (A,Z)
        p : projectile (A,Z)
        Elab: bombarding energy in the lab frame [MeV]. Either Elab or Ecm must be provided,
        not both.
        Ecm: bombarding energy in the com frame [MeV]. Either Elab or Ecm must be provided,
        not both.
        binding_model : optional callable taking in (A,Z) and returning binding energy in [MeV/c^2],
                        defaults to lookup in AME2020, and semi-empirical mass formula if not
                        available there
    Returns:
        mu (float) : reduced mass in MeV/c^2
        Ecm (float) : center-of-mass frame energy in MeV
        k (float) : center-of-mass frame wavenumber in fm^-1
    """
    m_t = mass_target - Q
    m_p = mass_projectile

    Ecm = m_t / (m_t + m_p) * Elab + Q
    Ep = Ecm + m_p

    # relativisitic correction from A. Ingemarsson 1974, Eqs. 17 & 20
    k = (
        m_t
        * np.sqrt(Elab * (Elab + 2 * m_p))
        / np.sqrt((m_t + m_p) ** 2 + 2 * m_t * Elab)
        / HBARC
    )
    mu = k**2 * Ep / (Ep**2 - m_p * m_p) * HBARC**2
    k_C = ALPHA * Zz * mu / HBARC
    eta = k_C / k

    return mu, Ecm, k, eta


@njit
def classical_kinematics(mass_target, mass_projectile, Elab, Zz=0, Q=0):
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Ecm = mass_target / (mass_target + mass_projectile) * Elab + Q
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return mu, Ecm, k, eta
