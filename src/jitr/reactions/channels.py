from ..utils.kinematics import ChannelKinematics

from . import reaction as rx

import numpy as np

from fractions import Fraction
from enum import Enum
from dataclasses import dataclass


# Workspaces call these functions and arrange channel QM's as need be
    # e.g. elastic (Jt=0, Jp=1/2 with central and spin-orbit) would:
        # find set of Jtot, pi values consistent with lmax
        # for each Jtot, pi
            # call build_system
            # pass resulting channel qm np array into spin_orbit_coupling to store <l dot s>
        # re-organize all channels in two np arrays (one for J=l-1/2 and one for J=l+1/2)
        # for each l 0 to lmax:
            # pre-compute Free-kinetic matrices for each of the two (l=J-1/2, l=J+1/2) channels
            # (re-factor free matrix to use numpy arrays)

        # then, when smatrix is called:
            # iterate over partial wave as normal

    # for true CC problem
        # find set of Jtot, pi values consistent with lmax
        # for each Jtot, pi
            # call build_system
            # pass resulting channel qm np array into CC coupling function

        # for each l 0 to lmax:
            # pre-compute Free-kinetic matrices for each of the two (l=J-1/2, l=J+1/2) channels

        # then, when smatrix is called
            # iterate over Jpi, solving each coupled system


# for Nucleus, G.S. spin and parity should be ctor args that default to None,
# and, if None, they grab values from ENSDF/RIPL

class Parity(Enum):
    positive: True
    negative: False


@dataclass
class Level:
    E: float
    I: Fraction
    pi: Parity


channel_dtype = np.dtype(
    [
        ("Ip", Fraction),
        ("It", Fraction),
        ("J", Fraction),
        ("pi_p", Parity),
        ("pi_t", Parity),
        ("Ex_t", float),
        ("Ex_p", float),
        ("l", int),
    ]
)


def spin_orbit_coupling(ch: np.ndarray) -> np.ndarray:
    """ J = l dot Jp basis, returns diaginal elements of <l dot Jp>  """
    Jp = ch['Jp']
    J = ch['J']
    l = ch['l']
    return (J*(J+1) - l*(l+1)) - Jp*(Jp +1)


def build_channels_2body(
    Jtot: Fraction,
    pi: Parity,
    channel_radius_fm: float,
    kinematics: ChannelKinematics,
    rxn: rx.Reaction,
    levels_projectile: list[Level],
    levels_target: list[Level],
):
    """
    calculates channel quantum numbers
    """
    pass

    # for each kinematic grid point
    # taking into account all possible combinations of
    # l, (Jp, pi_p), (Jt, pi_t) that can sum to Jtot, pi:

    # for a set of N possible channels,  (N,)-shaped numpy arrays of channel_dtype holding
        # channel l values
        # channel asymptotic Coulomb wave functions and their derivatives
        # channel Jp values
        # channel Jt values

    # return this numpy array
