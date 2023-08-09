import numpy as np
from matplotlib import pyplot as plt

from lagrange_rmatrix import (
    ProjectileTargetSystem,
    NonlocalRadialSEChannel,
    LagrangeRMatrix,
    yamaguchi_potential,
    yamaguchi_swave_delta,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
)


def nonlocal_interaction_example():
    alpha = 0.2316053  # fm**-1
    beta = 1.3918324  # fm**-1
    W0 = 41.472  # Mev fm**2

    params = (W0, beta, alpha)
    hbarc = 197.3  # MeV fm
    mass = hbarc**2 / (2 * W0)

    sys = ProjectileTargetSystem(
        incident_energy=0.1, reduced_mass=mass, channel_radius=20
    )

    se = NonlocalRadialSEChannel(
        l=0, system=sys, interaction=lambda r, rp: yamaguchi_potential(r, rp, params)
    )

    nbasis = 20
    solver_lm = LagrangeRMatrix(nbasis, sys, se)

    R, S, _ = solver_lm.solve()
    delta = np.rad2deg(np.real(np.log(S) / 2j))

    print("\nYamaguchi potential test:\n delta:")
    print("Lagrange-Legendre Mesh: {:.6f} [degrees]".format(delta))
    print(
        "Analytic              : {:.6f} [degrees]".format(
            yamaguchi_swave_delta(se.k, params)
        )
    )


if __name__ == "__main__":
    nonlocal_interaction_example()
