"""G4 smoke: jax.grad flows through potential → σ_reaction.

Uses ``method="linear_solve"`` — the eig spectral path is not differentiable
(lax DESIGN.md Appendix C.11).
"""

import numpy as np

from jitr.reactions import ElasticReaction
from jitr.utils.kinematics import classical_kinematics

from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 3
NBASIS = 16
RADIUS = 8.0


def test_grad_through_reaction_xs():
    import jax
    import jax.numpy as jnp

    from jitr.xs.elastic import IntegralWorkspace

    reaction = ElasticReaction((48, 20), (1, 0))  # neutron
    kinematics = classical_kinematics(
        reaction.target.m0, reaction.projectile.m0, np.array([8.0, 20.0]), 0
    )
    workspace = IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=RADIUS,
        lmax=LMAX,
        nbasis=NBASIS,
        method="linear_solve",
    )
    profile = jnp.exp(-((jnp.asarray(workspace.radial_grid()) / 3.5) ** 2))

    def total_reaction_xs(depth):
        central = (depth - 5.0j) * profile
        _, rxn = workspace.xs(central)
        return jnp.sum(rxn)

    value = total_reaction_xs(-40.0)
    gradient = jax.grad(total_reaction_xs)(-40.0)
    assert np.isfinite(float(value)) and float(value) > 0.0
    assert np.isfinite(float(gradient)) and float(gradient) != 0.0

    # finite-difference cross-check
    h = 1e-4
    fd = (total_reaction_xs(-40.0 + h) - total_reaction_xs(-40.0 - h)) / (2 * h)
    np.testing.assert_allclose(float(gradient), float(fd), rtol=5e-5)
