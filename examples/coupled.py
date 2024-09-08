import numpy as np
from matplotlib import pyplot as plt

from jitr import rmatrix
from jitr.reactions import ProjectileTargetSystem, wavefunction
from jitr.reactions.potentials import (
    woods_saxon_potential as ws,
    surface_peaked_gaussian_potential as spg,
    coulomb_charged_sphere as coul,
)
from jitr.utils import complex_det, kinematics


def interaction_3level(r, V, W, R0, a0, Zz, coupling_matrix):
    coulomb = coul(r, Zz, R0)
    nuclear = ws(r, V, W, R0, a0)
    zero = np.zeros_like(r)
    diagonal = np.array(
        [
            [nuclear + coulomb, zero, zero],
            [zero, nuclear + coulomb, zero],
            [zero, zero, nuclear + coulomb],
        ]
    )
    off_diag = coupling_matrix[..., np.newaxis] * spg(r, V, W, R0, a0)
    return diagonal + off_diag


def coupled_channels_example():
    """
    3 level system example with local diagonal and transition potentials and neutral
    particles. Potentials are real, so S-matrix is unitary and symmetric
    """

    Elab = 5  # MeV
    nchannels = 3
    nodes_within_radius = 3
    levels = np.array([0, 2.3, 3.1])

    # target (A,Z)
    Ca48 = (28, 20)
    mass_Ca48 = kinematics.mass(*Ca48)

    # projectile (A,Z)
    proton = (1, 1)
    mass_proton = kinematics.mass(*proton)

    # S-wave only
    sys = ProjectileTargetSystem(
        2 * np.pi * nodes_within_radius * np.ones(nchannels),
        np.zeros(nchannels, dtype=np.int64),
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
        nchannels=nchannels,
        level_energies=levels,
    )

    mu, Ecm, k, eta = kinematics.classical_kinematics(
        sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
    )
    channels, asymptotics = sys.coupled(Ecm, mu, k, eta)

    # initialize solver
    solver = rmatrix.Solver(40)

    coupling_matrix = np.array(
        [
            [0, 0.8, 0.1],
            [0, 0, 0.8],
            [0, 0, 0.0],
        ]
    )
    coupling_matrix += coupling_matrix.T

    # get R and S-matrix, and both internal and external soln
    R, S, x, uext_prime_boundary = solver.solve(
        channels,
        asymptotics,
        local_interaction=interaction_3level,
        local_args=(
            -42,
            0,
            4,
            0.8,
            sys.Zproj * sys.Zproj,
            coupling_matrix,
        ),
        wavefunction=True,
    )

    # calculate wavefunctions in s = k_i r space
    u = wavefunction.Wavefunctions(
        solver,
        x,
        S,
        uext_prime_boundary,
        sys.incoming_weights,
        channels,
    ).uint()

    # S must be unitary
    assert np.isclose(complex_det(S), 1.0)

    # plot in s-space
    s_values = np.linspace(0.05, sys.channel_radii[0], 500)

    lines = []
    for i in range(nchannels):
        u_values = u[i](s_values)
        (p1,) = plt.plot(s_values, np.real(u_values), label=r"$n=%d$" % i)
        (p2,) = plt.plot(s_values, np.imag(u_values), ":", color=p1.get_color())
        lines.append([p1, p2])

    legend1 = plt.legend(
        lines[0], [r"$\mathfrak{Re}\, u_n(s) $", r"$\mathfrak{Im}\, u_n(s)$"], loc=3
    )
    plt.legend([l[0] for l in lines], [l[0].get_label() for l in lines], loc=1)
    plt.gca().add_artist(legend1)

    plt.xlabel(r"$s = k_0 r$")
    plt.ylabel(r"$u (s) $ [a.u.]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    coupled_channels_example()
