import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    Wavefunctions,
    RMatrixSolver,
    woods_saxon_potential,
    coulomb_charged_sphere,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
    make_channel_data,
)

# target (A,Z)
Ca48 = (28, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2


@njit
def interaction(r, *params):
    (V0, W0, R0, a0, zz, RC) = params
    return woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, RC)


def local_interaction_example():
    r"""
    example of single-channel s-wave S-matrix calculation for p+Ca48
    """
    E = 14.1
    nodes_within_radius = 5
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

    channels = sys.build_channels_kinematics(E)
    channel_data = make_channel_data(channels)
    ch = channel_data[0]

    # Lagrange-Mesh
    solver_lm = RMatrixSolver(100)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, R0)

    im = InteractionMatrix(1)
    im.set_local_interaction(interaction, args=params)

    s_values = np.linspace(0.01, sys.channel_radii[0], 200)
    domain, init_con = ch.initial_conditions()

    # Runge-Kutta
    sol_rk = solve_ivp(
        lambda s, y,: schrodinger_eqn_ivp_order1(
            s, y, ch, im.local_matrix[0, 0], params
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
    S_rk = smatrix(R_rk, a, ch.l, ch.eta)

    R_lm, S_lm, x, uext_prime_boundary = solver_lm.solve(
        im, channels, wavefunction=True
    )
    # R_lmp = u_lm(se.a) / (se.a * derivative(u_lm, se.a, dx=1.0e-6))
    u_lm = Wavefunctions(
        solver_lm, x, S_lm, uext_prime_boundary, sys.incoming_weights, channel_data
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
    E = 14.1
    l = 0

    # Potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    a_grid = np.linspace(5, 50, 50)
    delta_grid = np.zeros_like(a_grid, dtype=complex)

    im = InteractionMatrix(1)
    im.set_local_interaction(woods_saxon_potential, args=params)

    solver = RMatrixSolver(60)

    for i, a in enumerate(a_grid):
        sys = ProjectileTargetSystem(
            channel_radii=np.array([a]),
            l=np.array([l]),
            mass_target=mass_Ca48,
            mass_projectile=mass_proton,
            # turn off coulomb for this test
            Ztarget=0,
            Zproj=0,
        )
        channels = sys.build_channels_kinematics(E)
        R, S, _ = solver.solve(im, channels)
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

    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, RC)

    # use same interaction for all channels
    im = InteractionMatrix(1)
    im.set_local_interaction(interaction, args=params)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for i, e in enumerate(egrid):
        # calculate channel kinematics at this energy
        channels = sys.build_channels_kinematics(e)
        channel_data = make_channel_data(channels)

        for l in lgrid:
            ch = channels[l : l + 1]

            # Lagrange-Legendre R-Matrix
            R_lm, S_lm, uext_boundary = solver.solve(
                im, ch, basis_boundary=basis_boundary, free_matrix=free_matrices[l]
            )

            # Runge-Kutta
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
