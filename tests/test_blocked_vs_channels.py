"""G3 smoke: blocked workspace ≡ per-l individually compiled solvers.

The exhaustive version lives in lax's own suite; this pins jitr's wiring.
"""

import numpy as np

from jitr.reactions import ElasticReaction

from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 5
NBASIS = 20
RADIUS = 9.0


def test_blocked_matches_channels_compiles():
    import lax

    from jitr.xs.elastic import IntegralWorkspace

    reaction = ElasticReaction((40, 20), (1, 1))
    kinematics = reaction.kinematics(np.array([12.0, 26.0]))
    workspace = IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=RADIUS,
        lmax=LMAX,
        nbasis=NBASIS,
    )
    grid = workspace.engine.grid
    rgrid = workspace.radial_grid()
    central = (-50.0 - 4.0j) * np.exp(-((rgrid / 3.6) ** 2))
    coulomb = (20.0 * 1.44 / np.maximum(rgrid, 1.2)).astype(np.complex128)
    splus, _ = workspace.smatrix(central, None, coulomb)

    # interior rescale (semi-relativistic kinematics) applied identically here
    scaled = (central + coulomb)[None, :] * grid.interior_scale[:, None]
    for ell in range(LMAX + 1):
        solver = lax.compile(
            mesh=lax.MeshSpec("legendre", "x", n=NBASIS, scale=RADIUS),
            channels=[
                lax.ChannelSpec(
                    l=ell, threshold=0.0, mass_factor=float(grid.mass_factors[0])
                )
            ],
            solvers=("spectrum", "smatrix"),
            energies=np.asarray(grid.energies),
            mass_factor_grid=(
                None if grid.uniform_mass_factor else np.asarray(grid.mass_factors)
            ),
            z1z2=(1, 20),
            V_is_complex=True,
            method="eig",
        )
        v = solver.interaction_from_array(local=[scaled], energy_dependent=True)
        s_ref = np.asarray(solver.smatrix_grid(solver.spectrum(v)))[:, 0, 0]
        np.testing.assert_allclose(
            np.asarray(splus)[ell], s_ref, rtol=1e-11, atol=1e-12
        )
