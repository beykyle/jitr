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


def _rho(ma: float, mb: float, mc: float, md: float, E: float, Q: float) -> float:
    rho = np.sqrt(ma * mc / (mb * md) * E / (E + Q))
    if rho >= 1:
        raise ValueError("rho must be < 1 for valid angle conversion.")
    return rho


def _validate_angles(angles, var_name: str):
    angles = np.asarray(angles, dtype=float)
    if np.any(angles < 0.0) or np.any(angles > 180.0):
        raise ValueError(f"{var_name} must be in the range [0, 180] degrees.")


def cm_to_lab_frame(
    angles_cm_deg: np.ndarray,
    ma: float,
    mb: float,
    mc: float,
    md: float,
    E: float,
    Q: float,
):
    """
    Convert angles from the center of mass frame to the laboratory frame
    for a two-body reaction a + b -> c + d.

    Parameters
    ----------
    angles_cm_deg : np.ndarray
        An array of angles in the center of mass frame (in degrees).
    ma : float
        Mass of particle a.
    mb : float
        Mass of particle b.
    mc : float
        Mass of particle c.
    md : float
        Mass of particle d.
    E : float
        Energy in the laboratory frame.
    Q : float
        Q-value of the reaction.

    Returns
    -------
    np.ndarray
        An array of angles in the laboratory frame (in degrees).
    """
    _validate_angles(angles_cm_deg, "angles_cm_deg")
    rho = _rho(ma, mb, mc, md, E, Q)
    theta_cm = np.deg2rad(np.asarray(angles_cm_deg, dtype=float))

    theta_lab = np.arctan2(np.sin(theta_cm), rho + np.cos(theta_cm))
    theta_lab = np.where(theta_lab < 0.0, theta_lab + 2.0 * np.pi, theta_lab)

    return np.rad2deg(theta_lab)


def lab_to_cm_frame(
    angles_lab_deg: np.ndarray,
    ma: float,
    mb: float,
    mc: float,
    md: float,
    E: float,
    Q: float,
):
    """
    Convert angles from the laboratory frame to the center of mass frame
    for a two-body reaction a + b -> c + d.

    Parameters
    ----------
    angles_lab_deg : np.ndarray
        An array of angles in the laboratory frame (in degrees).
    ma : float
        Mass of particle a.
    mb : float
        Mass of particle b.
    mc : float
        Mass of particle c.
    md : float
        Mass of particle d.
    E : float
        Energy in the laboratory frame.
    Q : float
        Q-value of the reaction.

    Returns
    -------
    np.ndarray
        An array of angles in the center of mass frame (in degrees).
    """
    _validate_angles(angles_lab_deg, "angles_lab_deg")
    rho = _rho(ma, mb, mc, md, E, Q)
    theta_lab = np.deg2rad(np.asarray(angles_lab_deg, dtype=float))

    sin_lab = np.sin(theta_lab)
    cos_lab = np.cos(theta_lab)

    under_sqrt = 1.0 - (rho * sin_lab) ** 2
    under_sqrt = np.clip(under_sqrt, 0.0, None)

    k = rho * cos_lab + np.sqrt(under_sqrt)

    sin_cm = k * sin_lab
    cos_cm = k * cos_lab - rho

    sin_cm = np.clip(sin_cm, -1.0, 1.0)
    cos_cm = np.clip(cos_cm, -1.0, 1.0)

    theta_cm = np.arctan2(sin_cm, cos_cm)
    theta_cm = np.where(theta_cm < 0.0, theta_cm + 2.0 * np.pi, theta_cm)

    return np.rad2deg(theta_cm)
