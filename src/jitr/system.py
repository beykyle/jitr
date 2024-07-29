from numba.experimental import jitclass
from numba import float64, int32
import numpy as np

from .channel import ChannelData
from .utils import (
    classical_kinematics,
    H_plus,
    H_minus,
    H_plus_prime,
    H_minus_prime,
)


channel_dtype = [
    ("weight", np.float64),
    ("l", np.int32),
    ("mu", np.float64),
    ("a", np.float64),
    ("E", np.float64),
    ("k", np.float64),
    ("eta", np.float64),
    ("Hp", np.complex128),
    ("Hm", np.complex128),
    ("Hpp", np.complex128),
    ("Hmp", np.complex128),
]


class InteractionMatrix:
    r"""Represents the interaction potentials in each channel as numpy object arrays,
    one for local interactions and one for nonlocal
    """

    def __init__(self, nchannels: np.int32 = 1):
        r"""Initialize the InteractionMatrix

        Parameters:
            - nchannels (int) : the number of channels
        """
        self.nchannels = nchannels
        self.local_args = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_args = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.local_matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_symmetric = np.ones((self.nchannels, self.nchannels), dtype=bool)

        # initialize local interaction to 0's
        for i in range(self.nchannels):
            for j in range(self.nchannels):
                self.local_matrix[i, j] = None
                self.local_args[i, j] = None

    def set_nonlocal_interaction(
        self,
        interaction,
        i: np.int32 = 0,
        j: np.int32 = 0,
        is_symmetric=True,
        args=None,
    ):
        self.nonlocal_matrix[i, j] = interaction
        self.nonlocal_symmetric[i, j] = is_symmetric
        self.nonlocal_args[i, j] = args

    def set_local_interaction(
        self, interaction, i: np.int32 = 0, j: np.int32 = 0, args=None
    ):
        self.local_matrix[i, j] = interaction
        self.local_args[i, j] = args


class ProjectileTargetSystem:
    r"""
    Stores energetics of the system. Calculates useful parameters for each channel.
    """

    def __init__(
        self,
        channel_radii: float64[:],
        l: int32[:],
        mass_target: float64 = 0,
        mass_projectile: float64 = 0,
        Ztarget: float64 = 0,
        Zproj: float64 = 0,
        nchannels: int32 = 1,
        level_energies: float64[:] = None,
        incoming_weights: float64[:] = None,
    ):
        self.channel_radii = channel_radii
        if level_energies is None:
            level_energies = np.zeros(nchannels)

        if incoming_weights is None:
            incoming_weights = np.zeros(nchannels)
            incoming_weights[0] = 1

        self.level_energies = level_energies
        self.incoming_weights = incoming_weights
        self.l = l

        self.mass_target = mass_target
        self.mass_projectile = mass_projectile
        self.Ztarget = Ztarget
        self.Zproj = Zproj
        self.nchannels = nchannels

        assert l.shape == (nchannels,)
        assert channel_radii.shape == (nchannels,)
        assert level_energies.shape == (nchannels,)
        assert incoming_weights.shape == (nchannels,)

    def build_channels_kinematics(self, E_lab):
        Q = -self.level_energies
        Zz = self.Zproj * self.Ztarget
        mu, E_com, k, eta = classical_kinematics(
            self.mass_target, self.mass_projectile, E_lab, Q, Zz
        )
        return self.build_channels(E_com, mu, k, eta)

    def build_channels(self, E_com, mu, k, eta):
        channels = np.zeros(
            self.nchannels,
            dtype=channel_dtype,
        )
        channels["weight"] = self.incoming_weights
        channels["l"] = self.l
        channels["a"] = self.channel_radii
        channels["E"] = E_com
        channels["mu"] = mu
        channels["k"] = k
        channels["eta"] = eta
        channels["Hp"] = np.array(
            [H_plus(ch["a"], ch["l"], ch["eta"]) for ch in channels],
            dtype=np.complex128,
        )
        channels["Hm"] = np.array(
            [H_minus(ch["a"], ch["l"], ch["eta"]) for ch in channels],
            dtype=np.complex128,
        )
        channels["Hpp"] = np.array(
            [H_plus_prime(ch["a"], ch["l"], ch["eta"]) for ch in channels],
            dtype=np.complex128,
        )
        channels["Hmp"] = np.array(
            [H_minus_prime(ch["a"], ch["l"], ch["eta"]) for ch in channels],
            dtype=np.complex128,
        )

        return channels


def make_channel_data(channels: np.array):
    return [
        ChannelData(*channels[["l", "mu", "a", "E", "k", "eta"]][i])
        for i in range(channels.shape[0])
    ]
