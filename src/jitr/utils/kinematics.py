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


def cm_to_lab_frame(
    angles_cm_deg: np.ndarray,
    ma: float,
    mb: float,
    mc: float,
    md: float,
    E: float,
    Q: float,
):
    r"""
    Convert angles from center of mass to lab frame for a 2-body A + B -> C + D reaction,
    using tan(theta_lab) = sin(theta_cm) / (rho + cos(theta_cm)) [Thompson & Nunes Eq 2.3.14]
    where rho = sqrt(ma * mc / (mb * md) * E / (E + Q)) [Thompson & Nunes Eq. 2.3.16].

    Parameters:
    --------
    angles_cm_deg : np.ndarray
        Angles in degrees in the center of mass frame.
    ma : float
        Mass of particle A in MeV/c^2.
    mb : float
        Mass of particle B in MeV/c^2.
    mc : float
        Mass of particle C in MeV/c^2.
    md : float
        Mass of particle D in MeV/c^2.
    E : float
        Total energy in the center of mass frame in MeV.
    Q : float
        Q-value of the reaction in MeV.
    Returns:
    --------
    np.ndarray
        Angles in degrees in the lab frame.
    """
    rho = np.sqrt(ma * mc / (mb * md) * E / (E + Q))
    if rho >= 1:
        raise ValueError("rho must be >= 1 for valid angle conversion.")
    angles_cm_rad = np.deg2rad(angles_cm_deg)
    angles_cm_rad[np.isclose(angles_cm_rad, np.pi)] = 0
    y = np.sin(angles_cm_rad)
    x = rho + np.cos(angles_cm_rad)

    theta_lab_rad = np.arctan2(y, x)
    return np.rad2deg(theta_lab_rad)


def lab_to_cm_frame(
    angles_lab_deg: np.ndarray,
    ma: float,
    mb: float,
    mc: float,
    md: float,
    E: float,
    Q: float,
):
    r"""
    Convert angles from lab to center of mass frame for a 2-body A + B -> C + D reaction,
    using tan(theta_lab) = sin(theta_cm) / (rho + cos(theta_cm))
    [Thompson & Nunes Eq 2.3.14]cwhere rho = sqrt(ma * mc / (mb * md) * E / (E + Q))
    [Thompson & Nunes Eq. 2.3.16].

    Parameters:
    --------
    angles_lab_deg : np.ndarray
        Angles in degrees in the center of mass frame.
    ma : float
        Mass of particle A in MeV/c^2.
    mb : float
        Mass of particle B in MeV/c^2.
    mc : float
        Mass of particle C in MeV/c^2.
    md : float
        Mass of particle D in MeV/c^2.
    E : float
        Total energy in the center of mass frame in MeV.
    Q : float
        Q-value of the reaction in MeV.
    Returns:
    --------
    np.ndarray
        Angles in degrees in the center of mass frame.
    """
    rho = np.sqrt(ma * mc / (mb * md) * E / (E + Q))
    if rho >= 1:
        raise ValueError("rho must be >= 1 for valid angle conversion.")
    t = np.tan(np.deg2rad(angles_lab_deg))

    s = (rho * t + np.sqrt((1 - rho**2) * t**4 + t**2)) / (t**2 + 1)
    peak_index = np.argmax(s)  # angle cm is within [0,pi]
    theta_cm_rad = np.arcsin(s)
    offset = 0 if s.size % 2 == 0 else 1
    theta_cm_rad[peak_index + offset :] = np.pi - theta_cm_rad[peak_index + offset :]
    return np.rad2deg(theta_cm_rad)
