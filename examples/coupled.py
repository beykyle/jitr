import numpy as np
from matplotlib import pyplot as plt

from lagrange_rmatrix import (
    ProjectileTargetSystem,
    RadialSEChannel,
    LagrangeRMatrix,
    woods_saxon_potential,
    coulomb_potential,
    surface_peaked_gaussian_potential,
    complex_det,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
)


def coupled_channels_example(visualize=False):
    """
    3 level system example with local diagonal and transition potentials and neutral
    particles. Potentials are real, so S-matrix is unitary and symmetric
    """
    mass = 939  # reduced mass of scattering system MeV / c^2

    # Potential parameters
    V = 60  # real potential strength
    W = 0  # imag potential strength
    R = 4  # Woods-Saxon potential radius
    a = 0.5  # Woods-Saxon potential diffuseness
    params = (V, W, R, a)

    nodes_within_radius = 10

    system = ProjectileTargetSystem(
        incident_energy=50,
        reduced_mass=939,
        channel_radius=5 * (2 * np.pi),
        num_channels=3,
        level_energies=[0, 12, 20],
    )

    l = 0

    matrix = np.empty((3, 3), dtype=object)

    # diagonal potentials are just Woods-Saxons
    for i in range(system.num_channels):
        matrix[i, i] = RadialSEChannel(
            l=l,
            system=system,
            interaction=lambda r: woods_saxon_potential(r, params),
            threshold_energy=system.level_energies[i],
        )

    # transition potentials have depths damped by a factor compared to diagonal terms
    # and use surface peaked Gaussian form factors rather than Woods-Saxons
    transition_dampening_factor = 1
    Vt = V / transition_dampening_factor
    Wt = W / transition_dampening_factor

    # off diagonal potential terms
    for i in range(system.num_channels):
        for j in range(system.num_channels):
            if i != j:
                matrix[i, j] = RadialSEChannel(
                    l=l,
                    system=system,
                    interaction=lambda r: surface_peaked_gaussian_potential(
                        r, (Vt, Wt, R, a)
                    ),
                    threshold_energy=system.level_energies[i],
                )

    solver_lm = LagrangeRMatrix(40, system, matrix)

    H = solver_lm.bloch_se_matrix()

    if visualize:
        for i in range(3):
            for j in range(3):
                plt.imshow(np.real(solver_lm.single_channel_bloch_se_matrix(i, j)))
                plt.xlabel("n")
                plt.ylabel("m")
                plt.colorbar()
                plt.title(f"({i}, {j})")
                plt.show()

    # get R and S-matrix, and both internal and external soln
    R, S, uint = solver_lm.solve_wavefunction()

    # S must be unitary
    assert np.isclose(complex_det(S), 1.0)
    # S is symmetric iff the correct factors of momentum are applied
    # assert np.allclose(S, S.T)

    r_values = np.linspace(0.05, 40, 500)
    s_values = np.linspace(0.05, system.channel_radius, 500)

    lines = []
    for i in range(system.num_channels):
        u_values = uint[i].uint(units="s")(s_values)
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
    coupled_channels_example()
