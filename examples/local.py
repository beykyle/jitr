import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

from jitr import rmatrix
from jitr.reactions import ProjectileTargetSystem, make_channel_data, wavefunction
from jitr.reactions.potentials import (
    woods_saxon_potential,
    coulomb_charged_sphere,
)
from jitr.utils import delta, smatrix, schrodinger_eqn_ivp_order1, kinematics

# target (A,Z)
Ca48 = (48, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2


def interaction(r, *params):
    (V0, W0, R0, a0, zz, RC) = params
    return -woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, RC)


def local_interaction_example():
    r"""
    example of single-channel s-wave S-matrix calculation for p+Ca48
    """
    Elab = 14.1
    nodes_within_radius = 3
    n_partial_waves = 1

    sys = ProjectileTargetSystem(
        channel_radius=2 * np.pi * nodes_within_radius,
        lmax=0,
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
    )

    Ecm, mu, k, eta = kinematics.classical_kinematics(
        sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
    )
    channels, asymptotics = sys.get_partial_wave_channels(Ecm, mu, k, eta)

    l = 0
    channel_data_rk = make_channel_data(channels[l])
    ch = channel_data_rk[0]

    # Lagrange-Mesh
    solver_lm = rmatrix.Solver(100)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, R0)

    s_values = np.linspace(0.01, sys.channel_radius, 200)
    domain, init_con = ch.initial_conditions()

    # Runge-Kutta
    sol_rk = solve_ivp(
        lambda s, y: schrodinger_eqn_ivp_order1(s, y, ch, interaction, params),
        domain,
        init_con,
        dense_output=True,
        atol=1.0e-12,
        rtol=1.0e-12,
    ).sol
    a = domain[1]
    u_rk = sol_rk(s_values)[0]
    R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
    S_rk = smatrix(R_rk, a, ch.l, ch.eta)

    # Lagrange mesh
    R_lm, S_lm, x, uext_prime_boundary = solver_lm.solve(
        channels[l], asymptotics[l], interaction, params, wavefunction=True
    )
    # R_lmp = u_lm(se.a) / (se.a * derivative(u_lm, se.a, dx=1.0e-6))
    u_lm = wavefunction.Wavefunctions(
        solver_lm, x, S_lm, uext_prime_boundary, channels[l]
    ).uint()[0]
    u_lm = u_lm(s_values)

    R_lm = R_lm[0, 0]
    S_lm = S_lm[0, 0]

    delta_lm, atten_lm = delta(S_lm)
    delta_rk, atten_rk = delta(S_rk)

    # normalization and phase matching
    u_rk = u_rk * u_lm[20] / u_rk[20]

    print(f"k: {ch.k}")
    print(f"R-Matrix RK: {R_rk:.3e}")
    print(f"R-Matrix LM: {R_lm:.3e}")
    # print(f"R-Matrix LMp: {R_lmp:.3e}")
    print(f"S-Matrix RK: {S_rk:.3e}")
    print(f"S-Matrix LM: {S_lm:.3e}")
    print(f"real phase shift RK: {delta_rk:.3e} degrees")
    print(f"real phase shift LM: {delta_lm:.3e} degrees")
    print(f"complex phase shift RK: {atten_rk:.3e} degrees")
    print(f"complex phase shift LM: {atten_lm:.3e} degrees")

    plt.plot(s_values, np.real(u_rk), "k", alpha=0.5, label="Runge-Kutta")
    plt.plot(
        s_values,
        np.imag(u_rk),
        ":k",
        alpha=0.5,
    )

    plt.plot(s_values, np.real(u_lm), "r", alpha=0.5, label="Lagrange-Legendre")
    plt.plot(
        s_values,
        np.imag(u_lm),
        ":r",
        alpha=0.5,
    )

    plt.legend()
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$u_{%d} (r) $ [a.u.]" % ch.l)
    plt.tight_layout()
    plt.show()


def channel_radius_dependence_test():
    r"""
    Channel radius dependence of single-channel s-wave S-matrix calculation for p+Ca48
    """

    Elab = 14.1
    sys = ProjectileTargetSystem(
        channel_radius=0,
        lmax=3,
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=0,
        Zproj=0,
    )
    mu, Ecm, k, eta = kinematics.classical_kinematics(
        sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
    )

    # Potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (
        V0,
        W0,
        R0,
        a0,
    )

    a_grid = np.linspace(5, 50, 50, dtype=np.float64)
    delta_grid = np.zeros_like(a_grid, dtype=complex)

    solver = rmatrix.Solver(60)

    # choose a partial wave
    l = 0
    for i, a in enumerate(a_grid):
        sys.channel_radius = a
        channels, asymptotics = sys.get_partial_wave_channels(Ecm, mu, k, eta)
        R, S, _ = solver.solve(
            channels[l], asymptotics[l], woods_saxon_potential, params
        )
        deltaa, attena = delta(S[0, 0])
        delta_grid[i] = deltaa + 1.0j * attena

    plt.plot(a_grid, np.real(delta_grid), label=r"$\mathfrak{Re}\,\delta_l$")
    plt.plot(a_grid, np.imag(delta_grid), label=r"$\mathfrak{Im}\,\delta_l$")
    plt.legend()
    plt.xlabel("channel radius [fm]")
    plt.ylabel(r"$\delta_l$ [degrees]")
    plt.show()


def rmse_RK_LM():
    r"""Test with simple Woods-Saxon plus coulomb without spin-orbit coupling"""

    n_partial_waves = 3
    egrid = np.linspace(0.1, 120, 200)
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
    solver = rmatrix.Solver(50)

    # precompute sub matrices for kinetic energy operator in
    # each partial wave channel
    free_matrices = solver.free_matrix(sys.channel_radius, sys.l, coupled=False)

    # precompute values of Lagrange basis functions at channel radius
    # radius is the same for each partial wave channel so just do it once
    basis_boundary = solver.precompute_boundaries(sys.channel_radius)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 18  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, proton[1] * Ca48[1], RC)

    # use same interaction for all channels (no spin-orbit coupling)
    error_matrix = np.zeros((n_partial_waves, len(egrid)))

    for i, Elab in enumerate(egrid):
        # calculate channel kinematics at this energy
        channels, asymptotics = sys.get_partial_wave_channels(
            *kinematics.classical_kinematics(
                sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
            )
        )

        # since our interaction is l-independent and we're using the same
        # set of parameters for each partial wave, we can actually pre-compute
        # the interaction part of the Lagrange-matrix
        im = solver.interaction_matrix(
            channels[0].k[0],
            channels[0].E[0],
            channels[0].a,
            channels[0].size,
            local_interaction=interaction,
            local_args=params,
        )

        for l in sys.l:
            # Lagrange-Legendre R-Matrix solve for this partial wave
            R_lm, S_lm, uext_boundary = solver.solve(
                channels[l],
                asymptotics[l],
                basis_boundary=basis_boundary,
                free_matrix=free_matrices[l],
                interaction_matrix=im,
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

    lines = []
    for l in sys.l:
        plt.plot(egrid, 100 * error_matrix[l, :], label=r"$l = %d$" % l)

    plt.ylabel(
        r"$ | \mathcal{S}_{l}^{\rm RK} - \mathcal{S}_{l}^{\rm LM} |"
        r" / | \mathcal{S}_{l}^{\rm RK}|$ [%]"
    )
    plt.xlabel(r"$E$ [MeV]")

    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    channel_radius_dependence_test()
    local_interaction_example()
    rmse_RK_LM()
