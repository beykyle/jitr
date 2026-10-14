"""A three-level coupled-channels system.

A neutron on a target with three 0+ states at 0, 2.3 and 3.1 MeV, coupled by a
surface-peaked transition potential. With all levels of spin 0 every J^pi
block holds one channel per level, all with l = J. The potentials are real, so
in each block the S-matrix of the open channels is unitary and symmetric.
"""

import numpy as np
from matplotlib import pyplot as plt

from jitr.optical_potentials.potential_forms import (
    surface_peaked_gaussian_potential as spg,
)
from jitr.optical_potentials.potential_forms import woods_saxon_potential as ws
from jitr.reactions import Level
from jitr.utils import constants, mass
from jitr.utils.kinematics import classical_kinematics, level_kinematics
from jitr.xs import coupled


def coupled_channels_example():
    levels = [Level(0), Level(0), Level(0)]
    Ex = [0.0, 2.3, 3.1]
    kin = classical_kinematics(mass.mass(48, 20)[0], constants.MASS_N, 5.0, 0)
    k2, mu, eta = level_kinematics(kin, Ex)
    workspace = coupled.Workspace(
        3 * 2 * np.pi / np.sqrt(k2[0, 0]), 40, levels, 10, k2, mu, eta
    )

    coupling = np.array([[0, 0.8, 0.1], [0.8, 0, 0.1], [0.1, 0.1, 0.0]])
    r = workspace.r
    terms = [
        (np.eye(3), ws(r, -42.0, 0, 4.0, 0.8)),
        (coupling, spg(r, -42.0, 0, 4.0, 0.8)),
    ]
    S, u, _ = workspace.wavefunctions(local=terms)
    S, is_open = np.asarray(S[0]), np.asarray(workspace.open[0])

    for b in range(3):  # J = 0, 1, 2
        n = int(is_open[b].sum())
        Sb = S[b, :n, :n]
        print(
            f"J = {workspace.blocks.J[b]:.0f}: {n} open channels, "
            f"unitary {np.allclose(Sb @ Sb.conj().T, np.eye(n))}, "
            f"symmetric {np.allclose(Sb, Sb.T)}"
        )

    # the J = 0 wavefunctions for an incoming wave in the ground state
    for c in range(3):
        (line,) = plt.plot(r, np.real(u[0, 0, 0, c]), label=rf"level {c}")
        plt.plot(r, np.imag(u[0, 0, 0, c]), ":", color=line.get_color())
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$\tilde u_c(r)$ [a.u.]")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    coupled_channels_example()
