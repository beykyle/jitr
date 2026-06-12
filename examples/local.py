"""Single-channel local-potential examples on the lax-backed workspaces.

Demonstrates the energy-vectorized ``jitr.xs.elastic`` API: one workspace
per (reaction, energy grid, lmax, channel radius), potentials supplied as
arrays on the energy-independent fm grid or as callables.
"""

import numpy as np
from matplotlib import pyplot as plt

from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_potential,
)
from jitr.reactions import ElasticReaction
from jitr.utils import delta, kinematics
from jitr.xs.elastic import DifferentialWorkspace, IntegralWorkspace

# Woods-Saxon potential parameters
V0 = 60  # real potential strength
W0 = 20  # imag potential strength
R0 = 4  # Woods-Saxon potential radius
a0 = 0.5  # Woods-Saxon potential diffuseness


def energy_scan_example():
    r"""Phase shifts for p+48Ca over an energy grid from one workspace."""
    reaction = ElasticReaction((48, 20), (1, 1))
    elab_grid = np.linspace(5.0, 50.0, 30)
    channel_kinematics = kinematics.classical_kinematics(
        reaction.target.m0,
        reaction.projectile.m0,
        elab_grid,
        reaction.projectile.Z * reaction.target.Z,
    )
    workspace = IntegralWorkspace(
        reaction=reaction,
        kinematics=channel_kinematics,
        channel_radius_fm=15.0,
        lmax=4,
        nbasis=40,
    )
    rgrid = workspace.radial_grid()
    central = -woods_saxon_potential(rgrid, V0, W0, R0, a0)
    coulomb = coulomb_charged_sphere(
        rgrid, reaction.projectile.Z * reaction.target.Z, R0
    ).astype(np.complex128)

    splus, _ = workspace.smatrix(central, None, coulomb)
    for ell in range(3):
        deltas = np.array([delta(s)[0] for s in np.asarray(splus)[ell]])
        plt.plot(elab_grid, deltas, label=rf"$l = {ell}$")
    plt.xlabel(r"$E_{\rm lab}$ [MeV]")
    plt.ylabel(r"$\mathfrak{Re}\,\delta_l$ [degrees]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def channel_radius_dependence_test():
    r"""S-matrix stability against the channel radius (n+48Ca, s-wave)."""
    reaction = ElasticReaction((48, 20), (1, 0))
    channel_kinematics = kinematics.classical_kinematics(
        reaction.target.m0, reaction.projectile.m0, 14.1, 0
    )

    a_grid = np.linspace(10, 30, 20)
    delta_grid = np.zeros_like(a_grid, dtype=complex)
    for i, radius in enumerate(a_grid):
        workspace = IntegralWorkspace(
            reaction=reaction,
            kinematics=channel_kinematics,
            channel_radius_fm=float(radius),
            lmax=0,
            nbasis=40,
        )
        central = -woods_saxon_potential(workspace.radial_grid(), V0, W0, R0, a0)
        splus, _ = workspace.smatrix(central)
        real_shift, attenuation = delta(complex(np.asarray(splus)[0, 0]))
        delta_grid[i] = real_shift + 1.0j * attenuation

    plt.plot(a_grid, np.real(delta_grid), label=r"$\mathfrak{Re}\,\delta_0$")
    plt.plot(a_grid, np.imag(delta_grid), label=r"$\mathfrak{Im}\,\delta_0$")
    plt.legend()
    plt.xlabel("channel radius [fm]")
    plt.ylabel(r"$\delta_0$ [degrees]")
    plt.show()


def differential_xs_example():
    r"""dσ/dΩ for p+48Ca at several energies from one workspace."""
    reaction = ElasticReaction((48, 20), (1, 1))
    elab_grid = np.array([10.0, 20.0, 35.0])
    channel_kinematics = kinematics.classical_kinematics(
        reaction.target.m0,
        reaction.projectile.m0,
        elab_grid,
        reaction.projectile.Z * reaction.target.Z,
    )
    angles = np.linspace(0.1, np.pi - 0.1, 90)
    workspace = DifferentialWorkspace.build_from_system(
        reaction=reaction,
        kinematics=channel_kinematics,
        channel_radius_fm=15.0,
        lmax=12,
        angles=angles,
        nbasis=40,
    )
    rgrid = workspace.radial_grid()
    central = -woods_saxon_potential(rgrid, V0, W0, R0, a0)
    coulomb = coulomb_charged_sphere(
        rgrid, reaction.projectile.Z * reaction.target.Z, R0
    ).astype(np.complex128)

    result = workspace.xs(central, None, coulomb)
    for i, elab in enumerate(elab_grid):
        ratio = np.asarray(result.dsdo)[i] / np.asarray(workspace.rutherford)[i]
        plt.semilogy(np.rad2deg(angles), ratio, label=rf"{elab:.0f} MeV")
    plt.xlabel(r"$\theta$ [degrees]")
    plt.ylabel(r"$d\sigma/d\Omega \,/\, d\sigma_{\rm Ruth}/d\Omega$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    channel_radius_dependence_test()
    energy_scan_example()
    differential_xs_example()
