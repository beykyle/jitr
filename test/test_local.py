import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from lagrange_rmatrix import (
    ProjectileTargetSystem,
    ChannelData,
    LagrangeRMatrixSolver,
    woods_saxon_potential,
    coulomb_charged_sphere,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
)


@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return woods_saxon_potential(r, (V0, W0, R0, a0)) + coulomb_charged_sphere(
        r, zz, r_c
    )


def rmse_RK_LM(nchannels: int = 5):
    r"""Test with simple Woods-Saxon plus coulomb without spin-orbit coupling"""

    lgrid = np.arange(0, nchannels - 1, 1)
    egrid = np.linspace(0.01, 100, 10)
    nodes_within_radius = 5

    # channels are the same except for l and uncoupled
    # so just set up a single channel system. We will set
    # incident energy and l later
    sys = ProjectileTargetSystem(
        939,
        np.ones(nchannels) * nodes_within_radius * (2 * np.pi),
        incident_energy=None,
        l=None,
        Ztarget=40,
        Zproj=1,
        nchannels=nchannels,
    )

    # Lagrange-Mesh
    solver_lm = LagrangeRMatrixSolver(40, 1, sys)

    # use same interaction for all channels
    interaction_matrix = InteractionMatrix(1)
    interaction_matrix.set_local_interaction(i, i, interaction)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, RC)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for i, e in enumerate(egrid):
        sys.incident_energy = e

        for l in lgrid:
            sys.l = np.array([l])
            ch = sys.build_channels()

            # Runge-Kutta
            sol_rk = solve_ivp(
                lambda s, y,: schrodinger_eqn_ivp_order1(
                    s, y, ch, interaction_matrix[l], params
                ),
                ch.domain,
                ch.initial_conditions(),
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            R_rk = sol_rk(se.a)[0] / (se.a * sol_rk(se.a)[1])
            S_rk = smatrix(R_rk, se.a, se.l, se.eta)

            # Lagrange-Legendre R-Matrix
            R_lm, S_lm, x = solver_lm.solve(interaction_matrix, ch, params)
            print(S_lm)
            print(S_rk)

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
