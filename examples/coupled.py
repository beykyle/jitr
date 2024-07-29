import numpy as np
from numba import njit
from matplotlib import pyplot as plt

from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    RMatrixSolver,
    Wavefunctions,
    woods_saxon_potential,
    surface_peaked_gaussian_potential,
    coulomb_charged_sphere,
    complex_det,
    make_channel_data,
)


@njit
def diagonal_interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, r_c)


def coupled_channels_example(visualize=False):
    """
    3 level system example with local diagonal and transition potentials and neutral
    particles. Potentials are real, so S-matrix is unitary and symmetric
    """

    nchannels = 3
    nodes_within_radius = 5
    levels = np.array([0, 2.3, 3.1])

    # target (A,Z)
    Ca48 = (28, 20)
    mass_Ca48 = 44657.26581995028  # MeV/c^2

    # projectile (A,Z)
    proton = (1, 1)
    mass_proton = 938.271653086152  # MeV/c^2

    # S-wave only
    sys = ProjectileTargetSystem(
        2 * np.pi * nodes_within_radius * np.ones(nchannels),
        np.zeros(nchannels),
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
        nchannels=nchannels,
        level_energies=levels,
    )

    # initialize solver
    solver = RMatrixSolver(40)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 0  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, proton[1] * Ca48[1], RC)

    interaction_matrix = InteractionMatrix(3)

    # diagonal potentials are just Woods-Saxons
    for i in range(nchannels):
        interaction_matrix.set_local_interaction(
            diagonal_interaction, i, i, args=params
        )

    # transition potentials have depths damped by a factor
    # compared to diagonal terms and use surface peaked Gaussian
    # form factors rather than Woods-Saxons
    transition_dampening_factor = 0.2
    Vt = V0 * transition_dampening_factor
    Wt = W0 * transition_dampening_factor
    params_off_diag = (Vt, Wt, R0, a0)

    # off diagonal potential terms
    for i in range(sys.nchannels):
        for j in range(sys.nchannels):
            if i != j:
                interaction_matrix.set_local_interaction(
                    surface_peaked_gaussian_potential,
                    i,
                    j,
                    args=params_off_diag,
                )

    ecom = 35
    channels = sys.build_channels_kinematics(ecom)

    # get R and S-matrix, and both internal and external soln
    R, S, x, uext_prime_boundary = solver.solve(
        interaction_matrix, channels, wavefunction=True
    )

    # calculate wavefunctions in s = k_i r space
    u = Wavefunctions(
        solver,
        x,
        S,
        uext_prime_boundary,
        sys.incoming_weights,
        make_channel_data(channels),
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

    plt.xlabel(r"$s_n = k_n r$")
    plt.ylabel(r"$u (s) $ [a.u.]")
    plt.tight_layout()
    plt.show()

    # plot in r-space
    r_values = s_values / channels["k"][:, np.newaxis]

    print("k_i =  ", *channels["k"])

    lines = []
    for i in range(nchannels):
        u_values = u[i](s_values)
        (p1,) = plt.plot(r_values[i, :], np.real(u_values), label=r"$n=%d$" % i)
        (p2,) = plt.plot(r_values[i, :], np.imag(u_values), ":", color=p1.get_color())
        lines.append([p1, p2])

    legend1 = plt.legend(
        lines[0], [r"$\mathfrak{Re}\, u_n(r) $", r"$\mathfrak{Im}\, u_n(r)$"], loc=3
    )
    plt.legend([l[0] for l in lines], [l[0].get_label() for l in lines], loc=1)
    plt.gca().add_artist(legend1)

    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$u (r) $ [a.u.]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    coupled_channels_example(visualize=False)
