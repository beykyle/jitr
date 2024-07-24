from numba.experimental import jitclass
from numba import float64, int32
import numpy as np

from .utils import (
    hbarc,
    c,
    alpha,
    H_plus,
    H_minus,
    H_plus_prime,
    H_minus_prime,
    classical_kinematics,
    classical_kinematics_com,
)
from .channel import ChannelData

channel_dtype = (
    [
        ("weight", np.float64),
        ("l", np.int32),
        ("mu", np.float64),
        ("a", np.float64),
        ("E", np.float64),
        ("k", np.float64),
        ("eta", np.float64),
        ("Hp", np.float64),
        ("Hm", np.float64),
        ("Hpp", np.float64),
        ("Hmp", np.float64),
    ],
)


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
        channel_radii: np.array,
        l: np.array,
        Ztarget: np.float64 = 0,
        Zproj: np.float64 = 0,
        nchannels: np.int32 = 1,
        level_energies: np.array = None,
        incoming_weights: np.array = None,
    ):
        self.channel_radii = channel_radii
        self.l = l
        self.Ztarget = Ztarget
        self.Zproj = Zproj
        self.nchannels = nchannels

        if level_energies is None:
            level_energies = np.zeros(self.nchannels)

        self.level_energies = level_energies

        if incoming_weights is None:
            incoming_weights = np.zeros(self.nchannels)
            incoming_weights[0] = 1

        self.incoming_weights = incoming_weights

        assert channel_radii.shape == (nchannels,)
        assert level_energies.shape == (nchannels,)
        assert incoming_weights.shape == (nchannels,)

    def build_channels_cm_frame(
        self, mass_target, mass_projectile, E_com, kinematics=classical_kinematics_com
    ):
        Q = -self.level_energies
        mu, _, k, eta = kinematics(
            mass_target, mass_projectile, E_com, Q, self.Zproj * self.Ztarget
        )
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
            [H_plus_prime(ch["a"], ch["l"], ch["eta"]) for ch in channels],
            dtype=np.complex128,
        )

        return channels

    def build_channels_lab_frame(
        self, mass_target, mass_projectile, E_lab, kinematics=classical_kinematics
    ):
        Q = -self.level_energies
        mu, E_com, k, eta = kinematics(
            mass_target, mass_projectile, E_lab, Q, self.Zproj * self.Ztarget
        )
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
            [H_plus_prime(ch["a"], ch["l"], ch["eta"]) for ch in channels],
            dtype=np.complex128,
        )

        return channels
