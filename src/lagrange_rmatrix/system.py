from numba.experimental import jitclass
from numba import float64, int64
import numpy as np

from .utils import hbarc, c, alpha, null
from .channel import ChannelData


class InteractionMatrix:
    r"""Represents the interaction potentials in each channel as numpy object arrays,
    one for local interactions and one for nonlocal
    """

    def __init__(self, nchannels: np.int64 = 1):
        r"""Initialize the InteractionMatrix

        Parameters:
            - nchannels (int) : the number of channels
        """
        self.nchannels = nchannels
        self.local_matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_symmetric = np.ones((self.nchannels, self.nchannels), dtype=bool)

        # initialize local interaction to 0's
        for i in range(self.nchannels):
            for j in range(self.nchannels):
                self.local_matrix[i, j] = null

    def set_nonlocal_interaction(
        self, interaction, i: np.int64 = 0, j: np.int64 = 0, is_symmetric=True
    ):
        self.nonlocal_matrix[i, j] = interaction
        self.nonlocal_matrix_symmetric[i, j] = is_symmetric

    def set_local_interaction(self, interaction, i: np.int64 = 0, j: np.int64 = 0):
        self.local_matrix[i, j] = interaction


system_spec = [
    ("reduced_mass", float64[:]),
    ("channel_radii", float64[:]),
    ("l", int64[:]),
    ("Ztarget", float64),
    ("Zproj", float64),
    ("nchannels", int64),
    ("level_energies", float64[:]),
    ("incoming_weights", float64[:]),
]


@jitclass(system_spec)
class ProjectileTargetSystem:
    def __init__(
        self,
        reduced_mass: np.array,
        channel_radii: np.array,
        l: np.array = None,
        Ztarget: np.float64 = 0,
        Zproj: np.float64 = 0,
        nchannels: np.int64 = 1,
        level_energies: np.array = None,
        incoming_weights: np.array = None,
    ):
        self.reduced_mass = reduced_mass
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
        assert l.shape == (nchannels,)
        assert level_energies.shape == (nchannels,)
        assert incoming_weights.shape == (nchannels,)

    def k(self, ecom):
        r"""
        Wavenumber in each channel
        """
        return np.sqrt(2 * (ecom - self.level_energies) / self.reduced_mass) / hbarc

    def velocity(self, ecom):
        return np.sqrt(2 * hbarc * self.k(ecom) / self.reduced_mass) * c

    def eta(self, ecom):
        r"""
        Sommerfield parameter in each channel
        """
        k = self.k(ecom)
        return (alpha * self.Zproj * self.Ztarget) * self.reduced_mass / (hbarc * k)

    def build_channels(self, ecom):
        k = self.k(ecom)
        eta = self.eta(ecom)
        channels = list()
        for i in range(self.nchannels):
            channels.append(
                ChannelData(
                    self.l[i],
                    self.reduced_mass,
                    self.channel_radii[i],
                    ecom - self.level_energies[i],
                    k[i],
                    eta[i],
                )
            )

        return channels
