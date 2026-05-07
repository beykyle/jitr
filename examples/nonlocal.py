import numpy as np

from jitr import rmatrix
from jitr.optical_potentials.potential_forms import (
    yamaguchi_potential,
    yamaguchi_swave_delta,
)
from jitr.reactions import ProjectileTargetSystem
from jitr.utils.constants import HBARC


def nonlocal_interaction_example():
    alpha = 0.2316053  # fm**-1
    beta = 1.3918324  # fm**-1
    W0 = 41.472  # Mev fm**2

    params = (W0, beta, alpha)

    ecom = 12
    mu = HBARC**2 / (2 * W0)
    k = np.sqrt(2 * mu * ecom) / HBARC
    eta = 0

    sys = ProjectileTargetSystem(
        channel_radius=30.0,
        lmax=5,
    )
    channels, asymptotics = sys.get_partial_wave_channels(ecom, ecom, mu, k, eta)

    l = 0
    solver = rmatrix.Solver(20)
    _, S, _ = solver.solve(
        channels[l],
        asymptotics[l],
        nonlocal_interaction=yamaguchi_potential,
        nonlocal_args=params,
    )

    delta = np.rad2deg(np.real(np.log(S[0, 0]) / 2j))

    print("\nYamaguchi potential test:\n delta:")
    print(f"Lagrange-Legendre Mesh: {delta:.6f} [degrees]")
    analytic_delta = yamaguchi_swave_delta(channels[l].k[0], *params)
    print(f"Analytic              : {analytic_delta:.6f} [degrees]")


if __name__ == "__main__":
    nonlocal_interaction_example()
