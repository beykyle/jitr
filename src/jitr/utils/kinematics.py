from dataclasses import astuple, dataclass

import numpy as np

from .constants import ALPHA, HBARC


@dataclass
class ChannelKinematics:
    Ecm: np.float64
    mu: np.float64
    k: np.float64
    eta: np.float64

    def __iter__(self):
        return iter(astuple(self))


def semi_relativistic_kinematics(
    mass_target,
    mass_projectile,
    Elab,
    Zz=0,
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
    m_t = mass_target
    m_p = mass_projectile

    Ecm = m_t / (m_t + m_p) * Elab
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

    return ChannelKinematics(Ecm, mu, k, eta)


def classical_kinematics(mass_target, mass_projectile, Elab, Zz=0):
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Ecm = mass_target / (mass_target + mass_projectile) * Elab
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return ChannelKinematics(Ecm, mu, k, eta)


def classical_kinematics_cm(mass_target, mass_projectile, Ecm, Zz=0):
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Elab = (mass_target + mass_projectile) / mass_target * Ecm
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return Elab, ChannelKinematics(Ecm, mu, k, eta)
