import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    ChannelData,
    Wavefunctions,
    LagrangeRMatrixSolver,
    woods_saxon_potential,
    coulomb_charged_sphere,
    delta,
    smatrix,
    schrodinger_eqn_ivp_order1,
)


@njit
def interaction(r, *params):
    (V0, W0, R0, a0, zz, RC) = params
    return woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, zz, RC)


def channel_radius_dependence_test():
    E = 14.1
    nodes_within_radius = 5

    # Potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    a_grid = np.linspace(10, 30, 50)
    delta_grid = np.zeros_like(a_grid, dtype=complex)

    ints = InteractionMatrix(1)
    ints.set_local_interaction(woods_saxon_potential, args=params)

    for i, a in enumerate(a_grid):
        sys = ProjectileTargetSystem(
            reduced_mass=np.array([939.0], dtype=np.float64),
            channel_radii=np.array([a]),
            l=np.array([0]),
        )
        channels = sys.build_channels(E)
        solver = LagrangeRMatrixSolver(40, 1, sys, E, channels)
        R, S, _ = solver.solve(ints, channels)
        deltaa, attena = delta(S)
        delta_grid[i] = deltaa + 1.0j * attena

    plt.plot(a_grid, np.real(delta_grid), label=r"$\mathfrak{Re}\,\delta_l$")
    plt.plot(a_grid, np.imag(delta_grid), label=r"$\mathfrak{Im}\,\delta_l$")
    plt.legend()
    plt.xlabel("channel radius [fm]")
    plt.ylabel(r"$\delta_l$ [degrees]")
    plt.show()


def local_interaction_example():
    E = 14.1
    nodes_within_radius = 5

    sys = ProjectileTargetSystem(
        reduced_mass=np.array([939.0]),
        channel_radii=np.array([nodes_within_radius * (2 * np.pi)]),
        l=np.array([1]),
        Ztarget=40,
        Zproj=1,
    )

    ch = sys.build_channels(E)

    # Lagrange-Mesh
    solver_lm = LagrangeRMatrixSolver(40, 1, sys, ecom=E)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, R0)

    ints = InteractionMatrix(1)
    ints.set_local_interaction(interaction, args=params)

    s_values = np.linspace(0.01, sys.channel_radii[0], 200)

    # Runge-Kutta
    sol_rk = solve_ivp(
        lambda s, y,: schrodinger_eqn_ivp_order1(
            s, y, ch[0], ints.local_matrix[0, 0], params
        ),
        ch[0].domain,
        ch[0].initial_conditions(),
        dense_output=True,
        atol=1.0e-12,
        rtol=1.0e-12,
    ).sol
    a = ch[0].domain[1]
    u_rk = sol_rk(s_values)[0]
    R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
    S_rk = smatrix(R_rk, a, ch[0].l, ch[0].eta)

    R_lm, S_lm, x, uext_prime_boundary = solver_lm.solve(ints, ch, wavefunction=True)
    # R_lmp = u_lm(se.a) / (se.a * derivative(u_lm, se.a, dx=1.0e-6))
    u_lm = Wavefunctions(
        solver_lm, x, S_lm, uext_prime_boundary, sys.incoming_weights, ch
    ).uint()[0]
    u_lm = u_lm(s_values)

    R_lm = R_lm[0, 0]
    S_lm = S_lm[0, 0]

    delta_lm, atten_lm = delta(S_lm)
    delta_rk, atten_rk = delta(S_rk)

    # normalization and phase matching
    u_rk = u_rk * np.max(np.real(u_lm)) / np.max(np.real(u_rk))

    print(f"k: {ch[0].k}")
    print(f"R-Matrix RK: {R_rk:.3e}")
    print(f"R-Matrix LM: {R_lm:.3e}")
    # print(f"R-Matrix LMp: {R_lmp:.3e}")
    print(f"S-Matrix RK: {S_rk:.3e}")
    print(f"S-Matrix LM: {S_lm:.3e}")
    print(f"real phase shift RK: {delta_rk:.3e} degrees")
    print(f"real phase shift LM: {delta_lm:.3e} degrees")
    print(f"complex phase shift RK: {atten_rk:.3e} degrees")
    print(f"complex phase shift LM: {atten_lm:.3e} degrees")

    plt.plot(s_values, np.real(u_rk), "k", label="Runge-Kutta")
    plt.plot(s_values, np.imag(u_rk), ":k")

    plt.plot(s_values, np.real(u_lm), "r", label="Lagrange-Legendre")
    plt.plot(s_values, np.imag(u_lm), ":r")

    plt.legend()
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$u_{%d} (r) $ [a.u.]" % ch[0].l)
    plt.tight_layout()
    plt.show()


def rmse_RK_LM():
    r"""Test with simple Woods-Saxon plus coulomb without spin-orbit coupling"""

    lgrid = np.arange(0, 6 - 1, 1)
    egrid = np.linspace(0.01, 100, 10)
    nodes_within_radius = 5

    # channels are the same except for l and uncoupled
    # so just set up a single channel system. We will set
    # incident energy and l later
    sys = ProjectileTargetSystem(
        np.array([939.0]),
        np.array([nodes_within_radius * (2 * np.pi)]),
        l=np.array([0]),
        Ztarget=40,
        Zproj=1,
        nchannels=1,
    )

    # Lagrange-Mesh solver, don't set the energy
    solver_lm = LagrangeRMatrixSolver(40, 1, sys, ecom=None)

    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    RC = R0  # Coulomb cutoff

    params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, RC)

    # use same interaction for all channels
    ints = InteractionMatrix(1)
    ints.set_local_interaction(interaction, args=params)

    error_matrix = np.zeros((len(lgrid), len(egrid)), dtype=complex)

    for i, e in enumerate(egrid):
        solver_lm.set_energy(e)
        for l in lgrid:
            sys.l = np.array([l])
            ch = sys.build_channels(e)
            a = ch[0].domain[1]

            # Runge-Kutta
            sol_rk = solve_ivp(
                lambda s, y,: schrodinger_eqn_ivp_order1(
                    s, y, ch[0], ints.local_matrix[0, 0], params
                ),
                ch[0].domain,
                ch[0].initial_conditions(),
                dense_output=True,
                atol=1.0e-12,
                rtol=1.0e-12,
            ).sol

            R_rk = sol_rk(a)[0] / (a * sol_rk(a)[1])
            S_rk = smatrix(R_rk, a, l, ch[0].eta)

            # Lagrange-Legendre R-Matrix
            R_lm, S_lm, uext_boundary = solver_lm.solve(ints, ch, ecom=e)

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
