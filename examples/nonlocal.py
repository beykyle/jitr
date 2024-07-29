import numpy as np
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    RMatrixSolver,
    yamaguchi_potential,
    yamaguchi_swave_delta,
    hbarc,
)


def nonlocal_interaction_example():
    alpha = 0.2316053  # fm**-1
    beta = 1.3918324  # fm**-1
    W0 = 41.472  # Mev fm**2

    params = (W0, beta, alpha)

    ecom = 0.1
    mu = hbarc**2 / (2 * W0)
    k = np.sqrt(2 * mu * ecom) / hbarc
    eta = 0

    sys = ProjectileTargetSystem(
        channel_radii=np.array([30.0]),
        l=np.array([0]),
    )
    channels = sys.build_channels(ecom, mu, k, eta)

    im = InteractionMatrix(1)
    im.set_nonlocal_interaction(yamaguchi_potential, args=params)

    solver = RMatrixSolver(20)
    _, S, _ = solver.solve(im, channels)

    delta = np.rad2deg(np.real(np.log(S[0, 0]) / 2j))

    print("\nYamaguchi potential test:\n delta:")
    print("Lagrange-Legendre Mesh: {:.6f} [degrees]".format(delta))
    print(
        "Analytic              : {:.6f} [degrees]".format(
            yamaguchi_swave_delta(channels["k"][0], *params)
        )
    )


if __name__ == "__main__":
    nonlocal_interaction_example()
