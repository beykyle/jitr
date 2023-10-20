from numba.experimental import jitclass
from numba import float64, int32
import numpy as np

from .utils import hbarc, c, null
from .channel import ChannelData


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
        self.local_matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.nonlocal_symmetric = np.ones((self.nchannels, self.nchannels), dtype=bool)

        # initialize local interaction to 0's
        for i in range(nchannels):
            for j in range(nchannels):
                self.local_matrix[i, j] = null

    def set_nonlocal_interaction(
        interaction, i: int = 0, j: int = 0, is_symmetric=True
    ):
        self.nonlocal_matrix[i, j] = interaction
        self.nonlocal_matrix_symmetric[i, j] = is_symmetric

    def set_local_interaction(interaction, i: int = 0, j: int = 0):
        self.local_matrix[i, j] = interaction


system_spec = [
    ("incident_energy", float64),
    ("reduced_mass", float64),
    ("channel_radii", float64[:]),
    ("l", int32[:]),
    ("Ztarget", float64),
    ("Zproj", float64),
    ("nchannels", int32),
    ("level_energies", float64[:]),
    ("incoming_weights", float64[:]),
]


@jitclass(system_spec)
class ProjectileTargetSystem:
    def __init__(
        self,
        reduced_mass: np.float64,
        channel_radii: np.array,
        incident_energy: np.float64 = None,
        l: np.array = None,
        Ztarget: np.float64 = 0,
        Zproj: np.float64 = 0,
        nchannels: np.int32 = 1,
        level_energies: np.array = None,
        incoming_weights: np.array = None,
    ):
        self.incident_energy = incident_energy
        self.reduced_mass = reduced_mass
        self.channel_radii = channel_radii
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

        if l is None:
            l = np.empty(self.nchannels)

        self.l = l

        assert channel_radii.shape == (nchannels,)
        assert l.shape == (nchannels,)
        assert level_energies.shape == (nchannels,)
        assert incoming_weights.shape == (nchannels,)

    def k(self):
        r"""
        Wavenumber in each channel
        """
        return (
            np.sqrt(
                2 * (self.incident_energy - self.level_energies) / self.reduced_mass
            )
            / hbarc
        )

    def velocity(self):
        return np.sqrt(2 * hbarc * self.k() / self.reduced_mass) * c

    def eta(self):
        r"""
        Sommerfield parameter in each channel
        """
        k = self.k()
        return (alpha * self.Zproj * self.Ztarget) * self.reduced_mass / (hbarc * k)

    def build_channels(self):
        k = self.k()
        eta = self.eta()
        channels = np.empty((self.num_channels, self.num_channels), dtype=object)
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                ChannelData[i, j] = RadialSEChannel(
                    l[i],
                    self.reduced_mass,
                    self.channel_radius[i],
                    self.incident_energy - self.level_energies[i],
                    k[i],
                    eta[i],
                )

        return channels
