from jitr import reactions, rmatrix
from jitr.utils.kinematics import classical_kinematics
import numpy as np


def potential_scalar(r, depth, mass):
    return depth * np.exp(-r / mass)


def potential_2level(r, depth, mass, coupling):
    diag = potential_scalar(r, depth, mass)
    off_diag = potential_scalar(r, coupling, mass)
    return np.array(
        [[diag, off_diag], [off_diag, diag]],
    )


nbasis = 40
solver = rmatrix.Solver(nbasis)

nchannels = 2
sys = reactions.ProjectileTargetSystem(
    5 * np.pi * np.ones(nchannels),
    np.arange(0, nchannels, dtype=np.int64),
    mass_target=44657,
    mass_projectile=938.3,
    Ztarget=20,
    Zproj=1,
    nchannels=nchannels,
)

Elab = 42.1
mu, Ecm, k, eta = classical_kinematics(
    sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
)

# for solving the un-coupled channels independently
channels_uncoupled, asymptotics_uncoupled = sys.uncoupled(Ecm, mu, k, eta)

# for solving as a single (block-diagonal) system
channels_coupled, asymptotics_coupled = sys.coupled(Ecm, mu, k, eta)

# the 2-level system is block diagonal consisting of 2 copies of the
# un-coupled scalar subsystems
params_2level = (10, 4, 0)
params_scalar = (10, 4)


def test_coupled_vs_single():
    # solve the two un-coupled systems on the block diagonal
    R, S, u = solver.solve(
        channels_uncoupled[0],
        asymptotics_uncoupled[0],
        potential_scalar,
        params_scalar,
    )
    R2, S2, u2 = solver.solve(
        channels_uncoupled[1],
        asymptotics_uncoupled[1],
        potential_scalar,
        params_scalar,
    )

    # solve the full system
    Rm, Sm, xm = solver.solve(
        channels_coupled,
        asymptotics_coupled,
        potential_2level,
        params_2level,
    )

    b = solver.precompute_boundaries(sys.channel_radii[0:1])
    bm = solver.precompute_boundaries(sys.channel_radii)
    np.testing.assert_almost_equal(b, bm[:nbasis])
    np.testing.assert_almost_equal(b, bm[nbasis : 2 * nbasis])

    np.testing.assert_almost_equal(np.linalg.det(Sm.conj().T @ Sm), 1)
    np.testing.assert_almost_equal(np.linalg.det(S.conj().T @ S), 1)
    np.testing.assert_almost_equal(Sm[1, 0], 0)
    np.testing.assert_almost_equal(Sm[0, 1], 0)
    np.testing.assert_almost_equal(Rm[1, 0], 0)
    np.testing.assert_almost_equal(Rm[0, 1], 0)
    np.testing.assert_almost_equal(Sm[1, 1], S2)
    np.testing.assert_almost_equal(Rm[1, 1], R2)
    np.testing.assert_almost_equal(Sm[0, 0], S)
    np.testing.assert_almost_equal(Rm[0, 0], R)

    free = solver.free_matrix(sys.channel_radii, sys.l)
    interaction = solver.interaction_matrix(
        channels_coupled, potential_2level, params_2level
    )

    # test diaginal blocks
    np.testing.assert_almost_equal(
        solver.free_matrix(sys.channel_radii[0:1], sys.l[0:1]),
        solver.get_channel_block(free, 0, 0),
    )
    np.testing.assert_almost_equal(
        solver.interaction_matrix(
            channels_uncoupled[0], potential_scalar, params_scalar
        ),
        solver.get_channel_block(
            interaction,
            0,
            0,
        ),
    )
    np.testing.assert_almost_equal(
        solver.free_matrix(sys.channel_radii[1:], sys.l[1:]),
        solver.get_channel_block(free, 1, 1),
    )
    np.testing.assert_almost_equal(
        solver.interaction_matrix(
            channels_uncoupled[1], potential_scalar, params_scalar
        ),
        solver.get_channel_block(
            interaction,
            1,
            1,
        ),
    )

    # test off diag blocks
    for i in range(nchannels):
        for j in range(nchannels):
            if j != i:
                np.testing.assert_almost_equal(solver.get_channel_block(free, i, j), 0)
                np.testing.assert_almost_equal(
                    solver.get_channel_block(interaction, i, j), 0
                )

    # test full matric
    A = solver.interaction_matrix(
        channels_uncoupled[0], potential_scalar, params_scalar
    ) + solver.free_matrix(sys.channel_radii[:1], sys.l[:1])
    Am = free + interaction
    np.testing.assert_almost_equal(Am[:nbasis, :nbasis], A)
    np.testing.assert_almost_equal(bm[:nbasis], b)
    x = np.linalg.solve(A, b)
    xm = np.linalg.solve(Am, bm)
    np.testing.assert_almost_equal(x, xm[:nbasis])
