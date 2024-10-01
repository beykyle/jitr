from jitr import reactions, rmatrix
from jitr.utils.kinematics import classical_kinematics
import numpy as np


def potential_scalar(r, depth, mass):
    return -depth * np.exp(-r / mass)


def coupling_2level(l):
    r"""
    Each partial wave has 2 uncoupled channels
    """
    return np.array([[1, 0], [0, 1]])


def potential_2level(r, depth, mass, coupling):
    r"""
    if coupling=0, this 2 level interaction acts on each
    channel independently
    """
    diag = potential_scalar(r, depth, mass)
    off_diag = potential_scalar(r, coupling, mass)
    return np.array(
        [[diag, off_diag], [off_diag, diag]],
    )


nbasis = 40
solver = rmatrix.Solver(nbasis)

nchannels = 2
sys_2level = reactions.ProjectileTargetSystem(
    channel_radius=5 * np.pi,
    lmax=10,
    mass_target=44657,
    mass_projectile=938.3,
    Ztarget=20,
    Zproj=1,
    coupling=coupling_2level,
)

Elab = 42.1
channels, asymptotics = sys_2level.get_partial_wave_channels(
    *classical_kinematics(
        sys_2level.mass_target,
        sys_2level.mass_projectile,
        Elab,
        sys_2level.Zproj * sys_2level.Ztarget,
    )
)

# look at the s-wave
l = 0

# get coupled channels for partial wave l
channels_coupled = channels[l]
asymptotics_coupled = asymptotics[l]

# get un-coupled channels for partial wave l
channels_uncoupled = channels[l].decouple()
asymptotics_uncoupled = asymptotics[l].decouple()

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

    b = solver.precompute_boundaries(sys_2level.channel_radius)
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

    b = solver.precompute_boundaries(sys_2level.channel_radius)
    free = solver.free_matrix(
        channels_coupled.a,
        channels_coupled.l,
        channels_coupled.E,
    )
    interaction = solver.interaction_matrix(
        channels_coupled.k[0],
        channels_coupled.E[0],
        channels_coupled.a,
        channels_coupled.size,
        potential_2level,
        params_2level,
    )

    # test diaginal blocks
    free_0 = solver.free_matrix(
        channels_uncoupled[0].a,
        channels_uncoupled[0].l,
        channels_uncoupled[0].E,
    )
    free_1 = solver.free_matrix(
        channels_uncoupled[1].a,
        channels_uncoupled[1].l,
        channels_uncoupled[1].E,
    )

    np.testing.assert_almost_equal(
        free_0,
        solver.get_channel_block(free, 0, 0),
    )
    np.testing.assert_almost_equal(
        free_1,
        solver.get_channel_block(free, 1, 1),
    )

    np.testing.assert_almost_equal(
        solver.interaction_matrix(
            channels_uncoupled[0].k[0],
            channels_uncoupled[0].E[0],
            channels_uncoupled[0].a,
            channels_uncoupled[0].size,
            potential_scalar,
            params_scalar,
        ),
        solver.get_channel_block(
            interaction,
            0,
            0,
        ),
    )
    np.testing.assert_almost_equal(
        solver.interaction_matrix(
            channels_uncoupled[1].k[0],
            channels_uncoupled[1].E[0],
            channels_uncoupled[1].a,
            channels_uncoupled[1].size,
            potential_scalar,
            params_scalar,
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

    # test full matrix
    A = (
        solver.interaction_matrix(
            channels_uncoupled[0].k[0],
            channels_uncoupled[0].E[0],
            channels_uncoupled[0].a,
            channels_uncoupled[0].size,
            potential_scalar,
            params_scalar,
        )
        + free_0
    )
    Am = free + interaction
    np.testing.assert_almost_equal(Am[:nbasis, :nbasis], A)
    bm = np.hstack([b, b])
    x = np.linalg.solve(A, b)
    xm = np.linalg.solve(Am, bm)
    np.testing.assert_almost_equal(x, xm[:nbasis])
