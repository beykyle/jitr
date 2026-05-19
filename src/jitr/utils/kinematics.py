"""Kinematic helpers for entrance, exit, and frame-conversion calculations."""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import numpy.typing as npt

from .constants import ALPHA, HBARC

FloatArray = npt.NDArray[np.float64]


@dataclass
class ChannelKinematics:
    """Kinematic quantities for a single reaction channel.

    Attributes:
        Elab: Laboratory-frame kinetic energy in MeV.
        Ecm: Center-of-mass kinetic energy in MeV.
        mu: Reduced mass in MeV/c^2.
        k: Center-of-mass wavenumber in fm^-1.
        eta: Sommerfeld parameter.
    """

    Elab: float
    Ecm: float
    mu: float
    k: float
    eta: float

    def __iter__(self):
        """Iterate over the stored kinematic values in field order."""
        return iter(astuple(self))


def semi_relativistic_kinematics(
    mass_target: float,
    mass_projectile: float,
    Elab: float,
    Zz: int = 0,
) -> ChannelKinematics:
    """Compute semi-relativistic entrance-channel kinematics.

    Uses the approximation from Ingemarsson (1974) for a projectile scattering
    from a fixed target.

    Args:
        mass_target: Target rest mass in MeV/c^2.
        mass_projectile: Projectile rest mass in MeV/c^2.
        Elab: Laboratory-frame kinetic energy in MeV.
        Zz: Product of projectile and target charges.

    Returns:
        A populated :class:`ChannelKinematics` instance.
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
    eta = (ALPHA * Zz * mu / HBARC) / k

    return ChannelKinematics(Elab, Ecm, mu, k, eta)


def classical_kinematics(
    mass_target: float,
    mass_projectile: float,
    Elab: float,
    Zz: int = 0,
) -> ChannelKinematics:
    """Compute non-relativistic kinematics from a laboratory energy.

    Args:
        mass_target: Target rest mass in MeV/c^2.
        mass_projectile: Projectile rest mass in MeV/c^2.
        Elab: Laboratory-frame kinetic energy in MeV.
        Zz: Product of projectile and target charges.

    Returns:
        A populated :class:`ChannelKinematics` instance.
    """
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Ecm = mass_target / (mass_target + mass_projectile) * Elab
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return ChannelKinematics(Elab, Ecm, mu, k, eta)


def classical_kinematics_cm(
    mass_target: float,
    mass_projectile: float,
    Ecm: float,
    Zz: int = 0,
) -> ChannelKinematics:
    """Compute non-relativistic kinematics from a center-of-mass energy.

    Args:
        mass_target: Target rest mass in MeV/c^2.
        mass_projectile: Projectile rest mass in MeV/c^2.
        Ecm: Center-of-mass kinetic energy in MeV.
        Zz: Product of projectile and target charges.

    Returns:
        A populated :class:`ChannelKinematics` instance.
    """
    mu = mass_target * mass_projectile / (mass_target + mass_projectile)
    Elab = (mass_target + mass_projectile) / mass_target * Ecm
    k = np.sqrt(2 * Ecm * mu) / HBARC
    eta = (ALPHA * Zz) * mu / (HBARC * k)
    return ChannelKinematics(Elab, Ecm, mu, k, eta)


def _rho(ma: float, mb: float, mc: float, md: float, E: float, Q: float) -> float:
    """Return the standard two-body frame-conversion factor."""
    rho = np.sqrt(ma * mc / (mb * md) * E / (E + Q))
    if rho >= 1:
        raise ValueError("rho must be < 1 for valid angle conversion.")
    return float(rho)


def _validate_angles(angles: npt.ArrayLike, var_name: str) -> None:
    """Validate that an angle array is defined on ``[0, 180]`` degrees."""
    angle_array = np.asarray(angles, dtype=float)
    if np.any(angle_array < 0.0) or np.any(angle_array > 180.0):
        raise ValueError(f"{var_name} must be in the range [0, 180] degrees.")


def cm_to_lab_frame(
    angles_cm_deg: npt.ArrayLike,
    ma: float,
    mb: float,
    mc: float,
    md: float,
    E: float,
    Q: float,
) -> FloatArray:
    """Convert center-of-mass angles to laboratory angles.

    Args:
        angles_cm_deg: Input angles in degrees.
        ma: Mass of projectile ``a``.
        mb: Mass of target ``b``.
        mc: Mass of ejectile ``c``.
        md: Mass of recoil ``d``.
        E: Laboratory-frame energy in MeV.
        Q: Reaction Q-value in MeV.

    Returns:
        A NumPy array of laboratory-frame angles in degrees.
    """
    _validate_angles(angles_cm_deg, "angles_cm_deg")
    rho = _rho(ma, mb, mc, md, E, Q)
    theta_cm = np.deg2rad(np.asarray(angles_cm_deg, dtype=float))

    theta_lab = np.arctan2(np.sin(theta_cm), rho + np.cos(theta_cm))
    theta_lab = np.where(theta_lab < 0.0, theta_lab + 2.0 * np.pi, theta_lab)

    return np.rad2deg(theta_lab)


def lab_to_cm_frame(
    angles_lab_deg: npt.ArrayLike,
    ma: float,
    mb: float,
    mc: float,
    md: float,
    E: float,
    Q: float,
) -> FloatArray:
    """Convert laboratory angles to center-of-mass angles.

    Args:
        angles_lab_deg: Input angles in degrees.
        ma: Mass of projectile ``a``.
        mb: Mass of target ``b``.
        mc: Mass of ejectile ``c``.
        md: Mass of recoil ``d``.
        E: Laboratory-frame energy in MeV.
        Q: Reaction Q-value in MeV.

    Returns:
        A NumPy array of center-of-mass angles in degrees.
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
