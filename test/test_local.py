import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    RMatrixSolver,
    woods_saxon_potential,
    coulomb_charged_sphere,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
    make_channel_data,
)


@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, r_c)


def rmse_RK_LM():
    r"""Test with simple Woods-Saxon plus coulomb without spin-orbit coupling"""

    n_partial_waves = 10
    lgrid = np.arange(0, n_partial_waves, dtype=np.int32)
    egrid = np.linspace(0.5, 100, 10)
    nodes_within_radius = 5

    # target (A,Z)
    Ca48 = (28, 20)
    mass_Ca48 = 44657.26581995028  # MeV/c^2

    # projectile (A,z)
    proton = (1, 1)
    mass_proton = 938.271653086152  # MeV/c^2

    sys = ProjectileTargetSystem(
        2 * np.pi * nodes_within_radius * np.ones(n_partial_waves),
        lgrid,
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
        nchannels=n_partial_waves,
    )

    # initialize solver
    solver = RMatrixSolver(40)

    # precompute sub matrices for kinetic energy operator in
    # each partial wave channel
    free_matrices = solver.free_matrix(sys.channel_radii, sys.l, full_matrix=False)

    # precompute values of Lagrange basis functions at channel radius
    # radius is the same for each partial wave channel so just do it once
    basis_boundary = solver.precompute_boundaries(sys.channel_radii[0:1])

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, proton[1] * Ca48[1], RC)

    # use same interaction for all channels (no spin-orbit coupling)
    im = InteractionMatrix(1)
    im.set_local_interaction(interaction, args=params)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for i, e in enumerate(egrid):
        # calculate channel kinematics at this energy
        channels = sys.build_channels_kinematics(e)
        channel_data = make_channel_data(channels)

        for l in lgrid:
            # get view of single channel corresponding to partial wave l
            ch = channels[l : l + 1]

            # Lagrange-Legendre R-Matrix solve for this partial wave
            R_lm, S_lm, uext_boundary = solver.solve(
                im,
                ch,
                basis_boundary=basis_boundary,
                free_matrix=free_matrices[l],
            )

            # Runge-Kutta solve for this partial wave
            domain, init_con = channel_data[l].initial_conditions()
            sol_rk = solve_ivp(
                lambda s, y: schrodinger_eqn_ivp_order1(
                    s, y, channel_data[l], im.local_matrix[0, 0], params
                ),
                domain,
                init_con,
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            a = domain[1]
            R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
            S_rk = smatrix(R_rk, a, l, channel_data[l].eta)

            # comparison between solvers
            delta_lm, atten_lm = delta(S_lm)
            delta_rk, atten_rk = delta(S_rk)

            err = 0 + 0j

            if np.fabs(delta_rk) > 1e-12:
                err += np.fabs(delta_lm - delta_rk)

            if np.fabs(atten_rk) > 1e-12:
                err += 1j * np.fabs(atten_lm - atten_rk)

            error_matrix[l, i] = err

        return np.real(error_matrix), np.imag(error_matrix)


def test_local():
    rtol = 1.0e-2
    real_err, imag_err = rmse_RK_LM()
    assert not np.any(real_err / 180 > rtol)
    assert not np.any(imag_err / 180 > rtol)
