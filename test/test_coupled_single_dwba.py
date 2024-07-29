import numpy as np
from numba import njit

import jitr


@njit
def potential(r, depth):
    return depth * np.exp(-r / 4)


nchannels = 2
nbasis = 40

sys = jitr.ProjectileTargetSystem(
    2 * np.pi * 3 * np.ones(nchannels),
    np.arange(0, nchannels, dtype=np.int32),
    mass_target=44657,
    mass_projectile=938.3,
    Ztarget=20,
    Zproj=1,
    nchannels=nchannels,
)
channels = sys.build_channels_kinematics(E_lab=42.1)
solver = jitr.RMatrixSolver(nbasis)
free_matrices = solver.free_matrix(sys.channel_radii, sys.l, full_matrix=False)

# single-channel interaction
interaction_matrix = jitr.InteractionMatrix(1)
interaction_matrix.set_local_interaction(potential, args=(10,))

# multichanel interaction
multi_channel_interaction = jitr.InteractionMatrix(nchannels)
for i in range(nchannels):
    multi_channel_interaction.set_local_interaction(potential, i, i, args=(10,))


def test_coupled_vs_single():
    # solve the two un-coupled systems on the block diagonal
    R, S, u = solver.solve(interaction_matrix, channels[0:1])
    R2, S2, u2 = solver.solve(interaction_matrix, channels[1:2])

    # solve the full system
    Rm, Sm, xm = solver.solve(multi_channel_interaction, channels)

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
    interaction = solver.interaction_matrix(multi_channel_interaction, channels)
    np.testing.assert_almost_equal(
        solver.free_matrix(sys.channel_radii[0:1], sys.l[0:1]),
        solver.get_channel_block(free, 0, 0),
    )
    np.testing.assert_almost_equal(
        solver.interaction_matrix(interaction_matrix, channels[0:1]),
        solver.get_channel_block(
            solver.interaction_matrix(multi_channel_interaction, channels),
            0,
            0,
        ),
    )
    for i in range(nchannels):
        for j in range(nchannels):
            if j != i:
                np.testing.assert_almost_equal(solver.get_channel_block(free, i, j), 0)
                np.testing.assert_almost_equal(
                    solver.get_channel_block(interaction, i, j), 0
                )

    A = solver.interaction_matrix(
        interaction_matrix, channels[0:1]
    ) + solver.free_matrix(channels["a"][0:1], channels["l"][0:1])
    Am = solver.interaction_matrix(
        multi_channel_interaction, channels
    ) + solver.free_matrix(channels["a"], channels["l"])
    np.testing.assert_almost_equal(Am[:nbasis, :nbasis], A)
    np.testing.assert_almost_equal(bm[:nbasis], b)
    x = np.linalg.solve(A, b)
    xm = np.linalg.solve(Am, bm)
    np.testing.assert_almost_equal(x, xm[:nbasis])
