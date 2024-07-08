import numpy as np
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    LagrangeRMatrixSolver,
    yamaguchi_potential,
    yamaguchi_swave_delta,
    hbarc,
)


def nonlocal_interaction_example():
    alpha = 0.2316053  # fm**-1
    beta = 1.3918324  # fm**-1
    W0 = 41.472  # Mev fm**2

    params = (W0, beta, alpha)
    mass = hbarc**2 / (2 * W0)

    ecom = 0.1

    sys = ProjectileTargetSystem(
        reduced_mass=np.array([mass]),
        channel_radii=np.array([30.0]),
        l=np.array([0], dtype=np.int64),
    )
    channels = sys.build_channels(ecom)

    interaction_matrix = InteractionMatrix(1)
    interaction_matrix.set_nonlocal_interaction(yamaguchi_potential, args=params)

    solver = LagrangeRMatrixSolver(20, 1, sys, ecom=ecom)
    _, S, _ = solver.solve(interaction_matrix, channels)

    delta = np.rad2deg(np.real(np.log(S[0, 0]) / 2j))

    print("\nYamaguchi potential test:\n delta:")
    print("Lagrange-Legendre Mesh: {:.6f} [degrees]".format(delta))
    print(
        "Analytic              : {:.6f} [degrees]".format(
            yamaguchi_swave_delta(channels[0].k, *params)
        )
    )


if __name__ == "__main__":
    nonlocal_interaction_example()
