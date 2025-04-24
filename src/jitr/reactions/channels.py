from ..utils.kinematics import ChannelKinematics

from . import reaction as rx

import numpy as np

from fractions import Fraction
from enum import Enum
from collections.abc import Callable
from dataclasses import dataclass


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


def spin_orbit_coupling(chi: np.ndarray, chj: np.ndarray) -> np.ndarray:
    """ l s basis, returns diagonal coupling matrix """
    Jp = chi['Jp']
    J = chi['J']
    l = chi['l']
    return np.diag( 0.5*(J*(J+1) - l*(l+1)) - Jp*(Jp +1) )


def build_system(
    Jtot: Fraction,
    pi: Parity,
    channel_radius_fm: float,
    kinematics: ChannelKinematics,
    rxn: rx.Reaction,
    levels_projectile: list[Level],
    levels_target: list[Level],
    coupling_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    """
    calculates channel quantum numbers and couplings
    """
    pass

    # for each kinematic grid point
    # taking into account all possible combinations of
    # l, (Jp, pi_p), (Jt, pi_t) that can sum to Jtot, pi:

    # for a set of N possible channels, set of (N,)-shaped numpy arrays for each of:
    # channel l values
    # channel asymptotic Coulomb wave functions and their derivatives
    # channel Jp values
    # channel Jt values
    # evaluate coupling matrix by calling coupling_function

    # return all of these numpy arrays
