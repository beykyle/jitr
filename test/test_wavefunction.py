import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from jitr import rmatrix
from jitr.reactions import (
    ProjectileTargetSystem,
    make_channel_data,
)
from jitr.reactions import potentials, wavefunction
from jitr.utils import smatrix, schrodinger_eqn_ivp_order1, kinematics

Elab = 14.1
nodes_within_radius = 5

# target (A,Z)
Ca48 = (28, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2

# p-wave (l=1)
sys = ProjectileTargetSystem(
    channel_radius=nodes_within_radius * (2 * np.pi),
    lmax=10,
    mass_target=mass_Ca48,
    mass_projectile=mass_proton,
    Ztarget=Ca48[1],
    Zproj=proton[1],
)

# Woods-Saxon potential parameters
V0 = 70  # real potential strength
W0 = 40  # imag potential strength
R0 = 6  # Woods-Saxon potential radius
a0 = 1.2  # Woods-Saxon potential diffuseness
params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, R0)

channel_kinematics = kinematics.classical_kinematics(
    sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
)
channels, asymptotics = sys.get_partial_wave_channels(*channel_kinematics)


def interaction(r, *params):
    (V0, W0, R0, a0, zz, RC) = params
    coulomb = potentials.coulomb_charged_sphere(r, zz, RC)
    nuclear = -potentials.woods_saxon_potential(r, V0, W0, R0, a0)
    return coulomb + nuclear


def test_wavefunction(l=0):
    ch = channels[l]
    asym = asymptotics[l]

    s_values = np.linspace(0.01, sys.channel_radius, 200)

    # Lagrange-Mesh
    solver_lm = rmatrix.Solver(100)
    R_lm, S_lm, x, uext_prime_boundary = solver_lm.solve(
        ch,
        asym,
        wavefunction=True,
        local_interaction=interaction,
        local_args=params,
    )
    u_lm = wavefunction.Wavefunctions(
        solver_lm,
        x,
        S_lm,
        uext_prime_boundary,
        ch,
    ).uint()[0]
    u_lm = u_lm(s_values)

    # Runge-Kutta
    rk_solver_channel_data = make_channel_data(ch)[0]
    domain, init_con = rk_solver_channel_data.initial_conditions()
    sol_rk = solve_ivp(
        lambda s, y: schrodinger_eqn_ivp_order1(
            s, y, rk_solver_channel_data, interaction, params
        ),
        domain,
        init_con,
        dense_output=True,
        atol=1.0e-12,
        rtol=1.0e-12,
    ).sol
    a = domain[1]
    u_rk = sol_rk(s_values)[0]
    R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
    S_rk = smatrix(R_rk, a, rk_solver_channel_data.l, rk_solver_channel_data.eta)

    np.testing.assert_almost_equal(R_rk, R_lm[0, 0], decimal=5)
    np.testing.assert_almost_equal(S_rk, S_lm[0, 0], decimal=5)

    ratio = u_lm[40] / u_rk[40]
    u_rk *= ratio
    np.testing.assert_allclose(
        np.absolute(u_rk - u_lm) / (np.absolute(u_rk)), 0, atol=1e-3
    )
