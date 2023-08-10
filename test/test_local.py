import numpy as np
from scipy.integrate import solve_ivp
from lagrange_rmatrix import (
    ProjectileTargetSystem,
    RadialSEChannel,
    LagrangeRMatrix,
    woods_saxon_potential,
    coulomb_potential,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
)


def rmse_RK_LM():
    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    nodes_within_radius = 5

    lgrid = np.arange(0, 2)
    egrid = np.linspace(0.01, 100, 10)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for l in lgrid:
        for i, e in enumerate(egrid):
            sys = ProjectileTargetSystem(
                incident_energy=e,
                reduced_mass=939,
                channel_radius=nodes_within_radius * (2 * np.pi),
                Ztarget=40,
                Zproj=1,
            )

            se = RadialSEChannel(
                l=l,
                system=sys,
                interaction=lambda r: woods_saxon_potential(r, params),
                coulomb_interaction=lambda zz, r: np.vectorize(coulomb_potential)(
                    zz, r, R0
                ),
            )

            # Runge-Kutta
            sol_rk = solve_ivp(
                lambda s, y,: schrodinger_eqn_ivp_order1(s, y, se),
                se.domain,
                se.initial_conditions(),
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            R_rk = sol_rk(se.a)[0] / (se.a * sol_rk(se.a)[1])
            S_rk = smatrix(R_rk, se.a, se.l, se.eta)

            # Lagrange-Mesh
            solver_lm = LagrangeRMatrix(40, sys, se)

            R_lm, S_lm, _ = solver_lm.solve()

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
