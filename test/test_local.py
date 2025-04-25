import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from jitr import rmatrix
from jitr.reactions import (
    ProjectileTargetSystem,
    make_channel_data,
)
from jitr.optical_potentials.potential_forms import (
    woods_saxon_potential,
    coulomb_charged_sphere,
)
from jitr.utils import delta, smatrix, schrodinger_eqn_ivp_order1, kinematics


def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return -woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(
        r, zz, r_c
    )


def test_local():
    r"""Test with simple Woods-Saxon plus coulomb without spin-orbit coupling"""

    n_partial_waves = 3
    egrid = np.linspace(0.5, 100, 10)
    nodes_within_radius = 5

    # target (A,Z)
    Ca48 = (28, 20)
    mass_Ca48 = 44657.26581995028  # MeV/c^2

    # projectile (A,z)
    proton = (1, 1)
    mass_proton = 938.271653086152  # MeV/c^2

    sys = ProjectileTargetSystem(
        channel_radius=2 * np.pi * nodes_within_radius,
        lmax=n_partial_waves - 1,
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
    )

    # initialize solver
    solver = rmatrix.Solver(70)

    # precompute sub matrices for kinetic energy operator in
    # each partial wave channel
    free_matrices = solver.free_matrix(sys.channel_radius, sys.l, coupled=False)

    # precompute values of Lagrange basis functions at channel radius
    # radius is the same for each partial wave channel so just do it once
    basis_boundary = solver.precompute_boundaries(sys.channel_radius)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, proton[1] * Ca48[1], RC)

    # use same interaction for all channels (no spin-orbit coupling)
    error_matrix = np.zeros((n_partial_waves, len(egrid)), dtype=complex)

    for i, Elab in enumerate(egrid):
        # calculate channel kinematics at this energy
        channels, asymptotics = sys.get_partial_wave_channels(
            *kinematics.classical_kinematics(
                sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
            )
        )

        for l in sys.l:
            # Lagrange-Legendre R-Matrix solve for this partial wave
            R_lm, S_lm, uext_boundary = solver.solve(
                channels[l],
                asymptotics[l],
                local_interaction=interaction,
                local_args=params,
                basis_boundary=basis_boundary,
                free_matrix=free_matrices[l],
            )

            # Runge-Kutta solve for this partial wave
            rk_solver_info = make_channel_data(channels[l])[0]
            domain, init_con = rk_solver_info.initial_conditions()
            sol_rk = solve_ivp(
                lambda s, y: schrodinger_eqn_ivp_order1(
                    s, y, rk_solver_info, interaction, params
                ),
                domain,
                init_con,
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            a = domain[1]
            R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
            S_rk = smatrix(R_rk, a, l, rk_solver_info.eta)

            error_matrix[l, i] = np.absolute(S_rk - S_lm[0, 0]) / np.absolute(S_rk)

    rtol = 1.0e-2  # 1 % max error
    np.testing.assert_array_less(error_matrix, rtol)
