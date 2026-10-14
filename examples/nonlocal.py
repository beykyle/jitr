"""A nonlocal potential: the s-wave Yamaguchi interaction, whose phase shift is
known analytically, solved on the Lagrange mesh at a few energies at once."""

import numpy as np

from jitr.optical_potentials.potential_forms import (
    yamaguchi_potential,
    yamaguchi_swave_delta,
)
from jitr.reactions import Level
from jitr.utils.constants import HBARC
from jitr.xs import coupled


def nonlocal_interaction_example():
    alpha, beta, W0 = 0.2316053, 1.3918324, 41.472  # fm^-1, fm^-1, MeV fm^2
    params = (W0, beta, alpha)
    mu = HBARC**2 / (2 * W0)
    E = np.array([0.1, 1.0, 12.0])
    k = np.sqrt(2 * mu * E) / HBARC
    ws = coupled.Workspace(
        15.0, 20, [Level(0)], 0, k[:, None] ** 2, np.full((3, 1), mu), np.zeros((3, 1))
    )
    r, rp = np.meshgrid(ws.r, ws.r, indexing="ij")
    S, _ = ws.smatrix(nonlocal_=[(np.eye(1), yamaguchi_potential(r, rp, *params))])
    delta = np.rad2deg(np.angle(np.asarray(S[:, 0, 0, 0])) / 2)
    for e in range(3):
        exact = yamaguchi_swave_delta(k[e], *params)
        print(f"E = {E[e]:5.1f} MeV: mesh {delta[e]:10.6f}, exact {exact:10.6f} deg")


if __name__ == "__main__":
    nonlocal_interaction_example()
