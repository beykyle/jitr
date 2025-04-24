from fractions import Fraction

from . import reaction as rx
from ..data.structure import Level, Parity, triangle_rule
from ..utils.free_solutions import H_plus, H_minus, H_plus_prime, H_minus_prime
from ..utils.angular_momentum import racah

import numpy as np


def spin_orbit_coupling(ch: np.ndarray) -> np.ndarray:
    """J = l dot Jp basis, returns diaginal elements of <l dot Jp>"""
    Jp = ch["Jp"]
    J = ch["J"]
    l = ch["l"]
    return np.array(J * (J + 1) - l * (l + 1) - Jp * (Jp + 1), dtype=float)


asymptotics_dtype = np.dtype(
    [("Hm", complex)("Hp", complex)("Hpp", complex)("Hmp", complex)]
)


def compute_channel_asymptotics(
    ch: np.ndarray,
    channel_radius_fm: float,
    kinematics: np.ndarray,
):
    assert ch.ndim == 1
    assert kinematics.ndim == 1
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
    chs: np.ndarray, chj: np.ndarray, Jtot: Fraction
):
    assert chs.ndim == 1
    assert chj.ndim == 1
    assert chs.shape == chj.shape
    assert chs.dtype == channel_sbasis_dtype
    assert chj.dtype == channel_jbasis_dtype
    prefactor = np.sqrt((2 * chs["s"] + 1) * (2 * chj["j"] + 1))
    matrix = np.zeros((chs.size, chs.size), dtype=complex)
    for m in range(chs.size):
        for n in range(chj.size):
            l, s, Ip, It = chs[["l", "s", "Ip", "It"]][m]
            j = chj["j"][n]
            matrix[m, n] = racah(l, Ip, Jtot, It, j, s)
    return prefactor * matrix


channel_sbasis_dtype = np.dtype(
    [
        ("l", int),
        ("s", Fraction),
        ("Ip", Fraction),
        ("pi_p", Parity),
        ("Ex_p", float),
        ("It", Fraction),
        ("pi_t", Parity),
        ("Ex_t", float),
    ]
)

channel_jbasis_dtype = np.dtype(
    [
        ("l", int),
        ("j", Fraction),
        ("Ip", Fraction),
        ("pi_p", Parity),
        ("Ex_p", float),
        ("It", Fraction),
        ("pi_t", Parity),
        ("Ex_t", float),
    ]
)


def build_channels_2body_sbasis(
    Jtot: Fraction,
    pi: Parity,
    level_p: Level,
    level_t: Level,
):
    """
    get channels coupling level_p in the projectile to level_t in the target
    with total angular momentum and parity jtot, pi in the sbasis

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
    get channels coupling level_p in the projectile to level_t in the target
    with total angular momentum and parity jtot, pi in the jbasis
    """

    # Initialize an empty list to store channel data
    channels = []

    for level_p in levels_projectile:
        for level_t in levels_target:

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
    """ """
    channels = []
    for level_p in levels_p:
        for level_t in levels_t:
            channels.append(build_channels(Jtot, pi, level_p, level_t))
    return np.vstack(channels)
