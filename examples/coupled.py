import numpy as np
from matplotlib import pyplot as plt

from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    LagrangeRMatrixSolver,
    Wavefunctions,
    woods_saxon_potential,
    surface_peaked_gaussian_potential,
    complex_det,
    FreeAsymptotics,
    compute_asymptotics,
)


def coupled_channels_example(visualize=False):
    """
    3 level system example with local diagonal and transition potentials and neutral
    particles. Potentials are real, so S-matrix is unitary and symmetric
    """

    # Potential parameters
    V = 60  # real potential strength
    W = 0  # imag potential strength
    R = 4  # Woods-Saxon potential radius
    a = 0.5  # Woods-Saxon potential diffuseness
    params = (V, W, R, a)

    nodes_within_radius = 5

    system = ProjectileTargetSystem(
        reduced_mass=np.array([939.0, 939.0, 939]),
        channel_radii=np.ones(3) * nodes_within_radius * (2 * np.pi),
        nchannels=3,
        level_energies=np.array([0.0, 12.0, 20.0]),
        l=np.array(
            [0, 0, 0]
        ),  # for simplicity, consider the S-wave coupling to 3 different 0+ levels
    )

    interaction_matrix = InteractionMatrix(3)

    # diagonal potentials are just Woods-Saxons
    for i in range(system.nchannels):
        interaction_matrix.set_local_interaction(
            woods_saxon_potential, i, i, args=params
        )

    # transition potentials have depths damped by a factor compared to diagonal terms
    # and use surface peaked Gaussian form factors rather than Woods-Saxons
    transition_dampening_factor = 4
    Vt = V / transition_dampening_factor
    Wt = W / transition_dampening_factor
    params_off_diag = (Vt, Wt, R, a)

    # off diagonal potential terms
    for i in range(system.nchannels):
        for j in range(system.nchannels):
            if i != j:
                interaction_matrix.set_local_interaction(
                    surface_peaked_gaussian_potential,
                    i,
                    j,
                    args=params_off_diag,
                )

    ecom = 35
    channels = system.build_channels(ecom)
    asym = compute_asymptotics(channels)
    solver = LagrangeRMatrixSolver(200, 3, system, ecom=ecom)

    # get R and S-matrix, and both internal and external soln
    R, S, x, uext_prime_boundary = solver.solve(
        interaction_matrix, channels, asym, wavefunction=True
    )
    u = Wavefunctions(
        solver,
        x,
        S,
        uext_prime_boundary,
        system.incoming_weights,
        channels,
        asym=FreeAsymptotics,
    ).uint()

    # S must be unitary
    assert np.isclose(complex_det(S), 1.0)

    s_values = np.linspace(0.05, system.channel_radii[0], 500)

    lines = []
    for i in range(system.nchannels):
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


if __name__ == "__main__":
    coupled_channels_example(visualize=False)
