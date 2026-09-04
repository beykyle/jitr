"""The elastic S-matrix must not depend on the channel radius once the nuclear
potential is negligible there. This is a regression test for a bug where the
interaction was scaled by Ecm rather than hbar^2 k^2 / (2 mu); with the
semi-relativistic kinematics these differ, which made the interior Coulomb
potential inconsistent with the asymptotic Sommerfeld parameter and produced an
O(100%) channel-radius dependence at backward angles for alpha scattering."""

import numpy as np

from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)
from jitr.reactions.reaction import Reaction
from jitr.rmatrix import Solver
from jitr.xs import elastic

ANGLES = np.deg2rad(np.arange(10, 180, 2.0))


def _ratio_to_rutherford(reaction, kinematics, channel_radius_fm, nbasis):
    A3 = reaction.target.A ** (1 / 3)
    Zz = reaction.target.Z * reaction.projectile.Z
    workspace = elastic.DifferentialWorkspace.build_from_system(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=channel_radius_fm,
        solver=Solver(nbasis),
        lmax=60,
        angles=ANGLES,
    )
    r = workspace.radial_grid()
    Rv, av, Rw, aw = 1.4 * A3, 0.5, 1.4 * A3, 0.4
    central = (
        -185.0 * woods_saxon_safe(r, Rv, av)
        - 1j * 25.0 * woods_saxon_safe(r, Rw, aw)
        - 1j * 25.0 * (-4 * aw) * woods_saxon_prime_safe(r, Rw, aw)
    )
    xs = workspace.xs(
        central_potential=central,
        coulomb_potential=coulomb_charged_sphere(r, Zz, 1.3 * A3),
    )
    return xs.dsdo / workspace.rutherford


def test_alpha_elastic_independent_of_channel_radius():
    reaction = Reaction(target=(48, 20), projectile=(4, 2), process="El")
    kinematics = reaction.kinematics(29.0)  # semi-relativistic by default
    y_near = _ratio_to_rutherford(reaction, kinematics, 12.5, 120)
    y_far = _ratio_to_rutherford(reaction, kinematics, 17.0, 160)
    rel = np.abs(y_near / y_far - 1)
    forward = ANGLES < np.deg2rad(60)
    assert rel[forward].max() < 1e-3
    assert rel.max() < 1e-2
