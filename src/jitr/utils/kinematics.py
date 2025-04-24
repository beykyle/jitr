from dataclasses import astuple, dataclass
import numpy as np
from .constants import ALPHA, HBARC


@dataclass
class ChannelKinematics:
    """
    A class to represent the kinematics of a channel.

    Attributes
    ----------
    Elab : float
        The energy in the laboratory frame.
    Ecm : float
        The energy in the center of mass frame.
    mu : float
        The reduced mass.
    k : float
        The wave number.
    eta : float
        The Sommerfeld parameter.
    """

    Elab: float
    Ecm: float
    mu: float
    k: float
    eta: float

    def __iter__(self):
        """
        Returns
        -------
        iterator
            An iterator over the attributes of the class.
        """
        return iter(astuple(self))


def semi_relativistic_kinematics(mass_target, mass_projectile, Elab, Zz=0):
    """
    Calculate the CM frame kinetic energy and wavenumber for a projectile
    scattering on a target using the relatavistic approximation of
    Ingemarsson, 1974:
    https://doi.org/10.1088/0031-8949/9/3/004

    Parameters:
    mass_target (float): Mass of the target nuclide.
    mass_projectile (float): Mass of the projectile.
    Elab (float): Laboratory frame energy.
    Zz (int, optional): Charge product of the interacting particles.
        Default is 0.

    Returns:
    ChannelKinematics: A dataclass containing Elab, Ecm, mu, k, and eta.
    """
    m_t = mass_target
    m_p = mass_projectile

    Ecm = m_t / (m_t + m_p) * Elab
    Ep = Ecm + m_p

    k = (
        m_t
        * np.sqrt(Elab * (Elab + 2 * m_p))
        / np.sqrt((m_t + m_p) ** 2 + 2 * m_t * Elab)
        / HBARC
    )
    mu = k**2 * Ep / (Ep**2 - m_p * m_p) * HBARC**2
    k_C = ALPHA * Zz * mu / HBARC
    eta = k_C / k

    return ChannelKinematics(Elab, Ecm, mu, k, eta)


def classical_kinematics(mass_target, mass_projectile, Elab, Zz=0):
    """
    Calculate classical kinematics for a projectile scattering on a target.

    Parameters:
    mass_target (float): Mass of the target nuclide.
    mass_projectile (float): Mass of the projectile.
    Elab (float): Laboratory frame energy.
    Zz (int, optional): Charge product of the interacting particles.
        Default is 0.

    Returns:
    ChannelKinematics: A dataclass containing Elab, Ecm, mu, k, and eta.
    """
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Ecm = mass_target / (mass_target + mass_projectile) * Elab
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return ChannelKinematics(Elab, Ecm, mu, k, eta)


def classical_kinematics_cm(mass_target, mass_projectile, Ecm, Zz=0):
    """
    Calculate classical kinematics for projectile scattering on a target
        with center-of-mass frame energy provided

    Parameters:
    mass_target (float): Mass of the target nuclide.
    mass_projectile (float): Mass of the projectile.
    Ecm (float): Center of mass frame energy.
    Zz (int, optional): Charge product of the interacting particles.
        Default is 0.

    Returns:
    ChannelKinematics: A dataclass containing Elab, Ecm, mu, k, and eta.
    """
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Elab = (mass_target + mass_projectile) / mass_target * Ecm
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return ChannelKinematics(Elab, Ecm, mu, k, eta)
