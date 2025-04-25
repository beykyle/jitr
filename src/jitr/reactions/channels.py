from fractions import Fraction

from ..structure.structure import Level, Parity

from ..utils.kinematics import kinematics_dtype
from ..utils.free_solutions import H_plus, H_minus, H_plus_prime, H_minus_prime
from ..utils.angular_momentum import RacahW, triangle_rule

import numpy as np

channel_sbasis_dtype = np.dtype(
    [
        ("l", int),  # Orbital angular momentum quantum number
        ("s", Fraction),  # Spin quantum number S = Ip + It
        ("Ex_p", float),  # Excitation energy of the projectile
        ("Ip", Fraction),  # Projectile spin
        ("pi_p", Parity),  # Projectile parity
        ("Ex_t", float),  # target excitation energy
        ("It", Fraction),  # target spin
        ("pi_t", Parity),  # target parity
    ]
)

channel_jbasis_dtype = np.dtype(
    [
        ("l", int),  # Orbital angular momentum quantum number
        ("j", Fraction),  # J = L + Ip
        ("Ex_p", float),  # Excitation energy of the projectile
        ("Ip", Fraction),  # Projectile spin
        ("pi_p", Parity),  # Projectile parity
        ("Ex_t", float),  # target excitation energy
        ("It", Fraction),  # target spin
        ("pi_t", Parity),  # target parity
    ]
)

asymptotics_dtype = np.dtype(
    [
        ("Hm", complex),  # outgoing Coulomb-Hankel function H^+_l(kr,eta)
        ("Hp", complex),  # incoming Coulomb-Hankel function H^-_l(kr,eta)
        ("Hpp", complex),  # d/ds H^+_l(s,eta)|_{s=kr}
        ("Hmp", complex),  # d/ds H^-_l(s,eta)|_{s=kr}
    ]
)


def spin_orbit_coupling(ch: np.ndarray) -> np.ndarray:
    """
    Calculate the diagonal elements of the spin-orbit coupling in the
        J = l dot Jt basis

    Parameters:
        ch (np.ndarray): A structured array with channel_jbasis_dtype

    Returns:
        np.ndarray: An array of diagonal elements of <l dot Jp>.

    """
    if ch.dtype != channel_jbasis_dtype:
        raise ValueError(
            f"ch must have dtype=channel_jbasis_dtype, but has {ch.dtype} instead"
        )
    Ip = ch["Ip"]
    j = ch["j"]
    l = ch["l"]
    return np.array(j * (j + 1) - l * (l + 1) - Ip * (Ip + 1), dtype=float)


def compute_channel_asymptotics(
    kinematics: np.ndarray,
    ch: np.ndarray,
    channel_radius_fm: float,
):
    """
    Compute the asymptotic behavior in a set of channels.

    Parameters
    ----------
    kinematics : np.ndarray
        A 1-dimensional array of kinematic variables with dtype
        kinematics_dtype
    ch : np.ndarray
        A 1-dimensional array in either the j or s basis
    channel_radius_fm : float
        The channel radius in femtometers.

    Returns
    -------
    np.ndarray
        A 2-dimensional array containing the computed asymptotic values,
        where the first dimension indexes along kinematic variables and the
        second along channel quantum numbers

    """
    if ch.ndim != 1:
        raise ValueError(f"ch must be 1D, but was {ch.ndim}D")
    if ch.dtype not in [channel_jbasis_dtype, channel_sbasis_dtype]:
        raise ValueError(
            f"ch must be have either channel_jbasis_dtype or "
            f"channel_sbasis_dtype, not{ch.dtype}"
        )
    if kinematics.ndim != 1:
        raise ValueError(f"kinematics must be 1D, but was {kinematics.ndim}D")
    if kinematics.dtype != kinematics_dtype:
        raise ValueError(
            f"kinematics must be have kinematics_dtype, not{kinematics.dtype}"
        )

    s = channel_radius_fm * kinematics["k"]
    eta = kinematics["eta"]

    asymptotics = np.zeros((kinematics.size, ch.size), dtype=asymptotics_dtype)

    for i, (k, eta) in enumerate(kinematics[["k", "eta"]]):
        s = k * channel_radius_fm
        for l in ch["l"]:
            asymptotics[i, l] = (
                H_plus(s, l, eta),
                H_minus(s, l, eta),
                H_plus_prime(s, l, eta),
                H_minus_prime(s, l, eta),
            )

    return asymptotics


def sbasis_to_jbasis_conversion_matrix(
    Jtot: Fraction, chs: np.ndarray, chj: np.ndarray,
):
    """
    Converts a basis from s-basis to j-basis using Eq. 3.2.4 in
    Thomson & Nunes, 2009

    The two channel arrays, chs and chj must describe the same system,
    e.g. all the fields corresponding the projectile and target
    E, I, and pi must be the same.

    Parameters
    ----------
    chs (np.ndarray): 1D array representing the s-basis channels.
    chj (np.ndarray): 1D array representing the j-basis channels.
    Jtot (Fraction): Total angular momentum.

    Returns
    -------
    np.ndarray: Conversion matrix from s-basis to j-basis.

    Raises
    ValueError if chs and chj are incompatible
    ------
    """
    if (
        chs.shape != chj.shape
        or chs.dtype != channel_sbasis_dtype
        or chj.dtype != channel_jbasis_dtype
        or chs.ndim != 1
    ):
        raise ValueError(
            "Incompatible form for chj and chs; they must be 1-dimensional, "
            "the same size, and have channel_jbasis_dtype and channel_sbasis_dtype "
            "as their respective dtypes"
        )
    assert np.all(chj[["Ip", "It"]] == chs[["Ip", "It"]])
    s = chs["s"].astype(float)
    j = chj["j"].astype(float)
    prefactor = np.sqrt((2 * s + 1) * (2 * j + 1))
    matrix = np.zeros((chs.size, chs.size), dtype=complex)
    for m in range(chs.size):
        Ip, It = chj[["Ip", "It"]][m]  # could use chj or chs here
        for n in range(chj.size):
            l, s = chs[["l", "s"]][m]
            j = chj["j"][n]
            matrix[m, n] = RacahW(l, Ip, Jtot, It, j, s)
    return prefactor * matrix


def build_channels_2body_sbasis(
    Jtot: Fraction,
    pi: Parity,
    level_p: Level,
    level_t: Level,
):
    """
    Get channels coupling level_p in the projectile to level_t in the target
    with total angular momentum and parity jtot and pi in the s-basis.

    Parameters:
        Jtot (Fraction): Total angular momentum.
        pi (Parity): Parity of the system.
        level_p (Level): Level of the projectile.
        level_t (Level): Level of the target.

    Returns:
        np.ndarray: Array of channels with specified properties.
    """
    # Initialize an empty list to store channel data
    channels = []

    # Loop over eigenvalues of S = Ip + It = Jtot - L
    for s in triangle_rule(level_p.I, level_t.I):
        # loop over eigenvalues of L = Jtot - S
        for l in triangle_rule(s, Jtot):
            if level_t.pi * level_p.pi * (-1) ** l == pi:
                # If J is valid and parities match, append the channel
                channels.append((l, s, *level_p, *level_t))

    return np.array(channels, dtype=channel_sbasis_dtype)


def build_channels_2body_jbasis(
    Jtot: Fraction,
    pi: Parity,
    level_p: Level,
    level_t: Level,
):
    """
    Get channels coupling level_p in the projectile to level_t in the target
    with total angular momentum and parity jtot and pi in the j-basis.

    Parameters:
        Jtot (Fraction): Total angular momentum.
        pi (Parity): Parity of the system.
        level_p (Level): Level of the projectile.
        level_t (Level): Level of the target.

    Returns:
        np.ndarray: Array of channels with specified properties.
    """

    # Initialize an empty list to store channel data
    channels = []

    # Loop over eigenvalues of J = Ip + L = Jtot - It
    for j in triangle_rule(Jtot, level_t.I):
        # loop over eigenvalues of L = J - Ip
        for l in triangle_rule(j, level_p.I):
            if level_t.pi * level_p.pi * (-1) ** l == pi:
                # If J is valid and parities match, append the channel
                channels.append((l, j, *level_p, *level_t))

    return np.array(channels, dtype=channel_jbasis_dtype)


def build_all_channels(
    Jtot: Fraction,
    pi: Parity,
    levels_p: list[Level],
    levels_t: list[Level],
    build_channels=build_channels_2body_jbasis,
):
    """
    Build all channels for given total angular momentum and
    parity and list of levels for the projectile and target

    Parameters
    ----------
    Jtot (Fraction): The total angular momentum.
    pi (Parity): The parity.
    levels_p (list[Level]): List of levels for the projectile.
    levels_t (list[Level]): List of levels for the target.
    build_channels (callable, optional): Function to build channels,
        defaults to build_channels_2body_jbasis.

    Returns
    -------
    np.ndarray: A stacked array of all channels.
    """
    channels = []
    for level_p in levels_p:
        for level_t in levels_t:
            channels.append(build_channels(Jtot, pi, level_p, level_t))
    return np.hstack(channels)
