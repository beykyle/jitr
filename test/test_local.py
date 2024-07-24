import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    LagrangeRMatrixSolver,
    woods_saxon_potential,
    coulomb_charged_sphere,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
    compute_asymptotics,
)


@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, r_c)


def rmse_RK_LM():
    r"""Test with simple Woods-Saxon plus coulomb without spin-orbit coupling"""

    lgrid = np.arange(0, 4, 1)
    egrid = np.linspace(0.5, 100, 10)
    nodes_within_radius = 5
    Ztarget = 40
    Zproj = 1

    # channels are the same except for l and uncoupled
    # so just set up a single channel system. We will set
    # incident energy and l later
    systems = [
        ProjectileTargetSystem(
            np.array([939.0]),
            np.array([nodes_within_radius * (2 * np.pi)]),
            l=np.array([l]),
            Ztarget=Ztarget,
            Zproj=Zproj,
            nchannels=1,
        )
        for l in lgrid
    ]

    # Lagrange-Mesh solvers, don't set the energy
    solvers = [LagrangeRMatrixSolver(40, 1, sys, ecom=None) for sys in systems]

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, Zproj * Ztarget, RC)

    # use same interaction for all channels
    interaction_matrix = InteractionMatrix(1)
    interaction_matrix.set_local_interaction(interaction, args=params)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for i, e in enumerate(egrid):
        for l in lgrid:
            channels = systems[l].build_channels(e)
            ch = channels[0]

            domain, init_con = ch.initial_conditions()

            # Runge-Kutta
            sol_rk = solve_ivp(
                lambda s, y,: schrodinger_eqn_ivp_order1(
                    s, y, ch, interaction_matrix.local_matrix[0, 0], params
                ),
                domain,
                init_con,
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            a = domain[1]
            R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
            S_rk = smatrix(R_rk, a, l, ch.eta)

            # Lagrange-Legendre R-Matrix
            asymptotics = compute_asymptotics(channels)
            R_lm, S_lm, uext_boundary = solvers[l].solve(
                interaction_matrix, channels, asymptotics
            )

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
