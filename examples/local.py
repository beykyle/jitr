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
Ca48 = (28, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2


def interaction(r, *params):
    (V0, W0, R0, a0, zz, RC) = params
    return woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, RC)


def local_interaction_example():
    r"""
    example of single-channel s-wave S-matrix calculation for p+Ca48
    """
    Elab = 14.1
    nodes_within_radius = 3
    n_partial_waves = 1
    l = np.array([0])

    sys = ProjectileTargetSystem(
        channel_radii=2 * np.pi * nodes_within_radius * np.ones(n_partial_waves),
        l=l,
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
        nchannels=n_partial_waves,
    )

    mu, Ecm, k, eta = kinematics.classical_kinematics(
        sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
    )
    channels, asymptotics = sys.coupled(Ecm, mu, k, eta)

    channel_data_rk = make_channel_data(channels)
    ch = channel_data_rk[0]

    # Lagrange-Mesh
    solver_lm = rmatrix.Solver(100)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, R0)

    s_values = np.linspace(0.01, sys.channel_radii[0], 200)
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

    R_lm, S_lm, x, uext_prime_boundary = solver_lm.solve(
        channels, asymptotics, interaction, params, wavefunction=True
    )
    # R_lmp = u_lm(se.a) / (se.a * derivative(u_lm, se.a, dx=1.0e-6))
    u_lm = wavefunction.Wavefunctions(
        solver_lm, x, S_lm, uext_prime_boundary, sys.incoming_weights, channels
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
        channel_radii=np.array([0], dtype=np.float64),
        l=np.array([0]),
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        # turn off coulomb for this test
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

    for i, a in enumerate(a_grid):
        sys.channel_radii[:] = a
        channels, asymptotics = sys.coupled(Ecm, mu, k, eta)
        R, S, _ = solver.solve(channels, asymptotics, woods_saxon_potential, params)
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
    n_partial_waves = 10
    lgrid = np.arange(0, n_partial_waves, dtype=np.int32)
    egrid = np.linspace(0.5, 100, 100)
    nodes_within_radius = 5

    # channels are the same except for l and uncoupled
    # so just set up a single channel system. We will set
    # incident energy and l later
    sys = ProjectileTargetSystem(
        channel_radii=2 * np.pi * nodes_within_radius * np.ones(n_partial_waves),
        l=lgrid,
        mass_target=mass_Ca48,
        mass_projectile=mass_proton,
        Ztarget=Ca48[1],
        Zproj=proton[1],
        nchannels=n_partial_waves,
    )

    # initialize solver
    solver = rmatrix.Solver(40)

    # precompute sub matrices for kinetic energy operator in
    # each partial wave channel
    free_matrices = solver.free_matrix(sys.channel_radii, sys.l, coupled=False)

    # precompute values of Lagrange basis functions at channel radius
    # radius is the same for each partial wave channel so just do it once
    basis_boundary = solver.precompute_boundaries(sys.channel_radii[:1])

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, RC)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for i, Elab in enumerate(egrid):
        # calculate channel kinematics at this energy
        mu, Ecm, k, eta = kinematics.classical_kinematics(
            sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
        )
        channels, asymptotics = sys.uncoupled(Ecm, mu, k, eta)

        for l in lgrid:
            # R-Matrix
            R_lm, S_lm, uext_boundary = solver.solve(
                channels[l],
                asymptotics[l],
                interaction,
                params,
                free_matrix=free_matrices[l],
                basis_boundary=basis_boundary,
            )

            # Runge-Kutta
            channel_data_rk = make_channel_data(channels[l])
            domain, init_con = channel_data_rk[0].initial_conditions()
            sol_rk = solve_ivp(
                lambda s, y: schrodinger_eqn_ivp_order1(
                    s, y, channel_data_rk[0], interaction, params
                ),
                domain,
                init_con,
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-9,
            ).sol

            a = domain[1]
            R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
            S_rk = smatrix(R_rk, a, l, channel_data_rk[0].eta)

            # comparison between solvers
            delta_lm, atten_lm = delta(S_lm[0, 0])
            delta_rk, atten_rk = delta(S_rk)

            err = 0 + 0j

            if np.fabs(delta_rk) > 1e-12:
                err += np.fabs(delta_lm - delta_rk)

            if np.fabs(atten_rk) > 1e-12:
                err += 1j * np.fabs(atten_lm - atten_rk)

            error_matrix[l, i] = err

    lines = []
    for l in lgrid:
        (p1,) = plt.plot(egrid, np.real(error_matrix[l, :]), label=r"$l = %d$" % l)
        (p2,) = plt.plot(egrid, np.imag(error_matrix[l, :]), ":", color=p1.get_color())
        lines.append([p1, p2])

    plt.ylabel(r"$\Delta \equiv | \delta^{\rm RK} - \delta^{\rm LM} |$ [degrees]")
    plt.xlabel(r"$E$ [MeV]")

    legend1 = plt.legend(
        lines[0], [r"$\mathfrak{Re}\, \Delta$", r"$\mathfrak{Im}\, \Delta$"], loc=0
    )
    plt.legend([l[0] for l in lines], [l[0].get_label() for l in lines], loc=1)
    plt.yscale("log")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    channel_radius_dependence_test()
    local_interaction_example()
    rmse_RK_LM()
