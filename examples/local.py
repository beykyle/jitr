import numpy as np
from matplotlib import pyplot as plt
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


def channel_radius_dependence_test():
    # Potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    a_grid = np.linspace(10, 30, 50)
    delta_grid = np.zeros_like(a_grid, dtype=complex)

    for i, a in enumerate(a_grid):
        sys = ProjectileTargetSystem(
            incident_energy=50, reduced_mass=939, channel_radius=a, Ztarget=60, Zproj=0
        )

        se = RadialSEChannel(
            l=0,
            system=sys,
            interaction=lambda r: woods_saxon_potential(r, params),
        )

        solver_lm = LagrangeRMatrix(40, sys, se)

        R_lm, S_lm, G = solver_lm.solve()
        delta_lm, atten_lm = delta(S_lm)

        delta_grid[i] = delta_lm + 1.0j * atten_lm

    plt.plot(a_grid, np.real(delta_grid), label=r"$\mathfrak{Re}\,\delta_l$")
    plt.plot(a_grid, np.imag(delta_grid), label=r"$\mathfrak{Im}\,\delta_l$")
    plt.legend()
    plt.xlabel("channel radius [fm]")
    plt.ylabel(r"$\delta_l$ [degrees]")
    plt.show()


def local_interaction_example():
    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    nodes_within_radius = 5

    sys = ProjectileTargetSystem(
        incident_energy=20,
        reduced_mass=939,
        channel_radius=nodes_within_radius * (2 * np.pi),
        Ztarget=40,
        Zproj=1,
    )

    se = RadialSEChannel(
        l=1,
        system=sys,
        interaction=lambda r: woods_saxon_potential(r, params),
        coulomb_interaction=lambda zz, r: np.vectorize(coulomb_potential)(zz, r, R0),
    )

    s_values = np.linspace(0.01, sys.channel_radius, 200)
    r_values = s_values / se.k

    # Runge-Kutta
    sol_rk = solve_ivp(
        lambda s, y,: schrodinger_eqn_ivp_order1(s, y, se),
        se.domain,
        se.initial_conditions(),
        dense_output=True,
        atol=1.0e-12,
        rtol=1.0e-9,
    ).sol

    u_rk = sol_rk(s_values)[0]
    R_rk = sol_rk(se.a)[0] / (se.a * sol_rk(se.a)[1])
    S_rk = smatrix(R_rk, se.a, se.l, se.eta)

    # Lagrange-Mesh
    solver_lm = LagrangeRMatrix(40, sys, se)

    R_lm, S_lm, u_lm = solver_lm.solve_wavefunction()
    # R_lmp = u_lm(se.a) / (se.a * derivative(u_lm, se.a, dx=1.0e-6))
    u_lm = u_lm(r_values)

    delta_lm, atten_lm = delta(S_lm)
    delta_rk, atten_rk = delta(S_rk)

    # normalization and phase matching
    u_rk = u_rk * np.max(np.real(u_lm)) / np.max(np.real(u_rk)) * (-1j)

    print(f"k: {se.k}")
    print(f"R-Matrix RK: {R_rk:.3e}")
    print(f"R-Matrix LM: {R_lm:.3e}")
    # print(f"R-Matrix LMp: {R_lmp:.3e}")
    print(f"S-Matrix RK: {S_rk:.3e}")
    print(f"S-Matrix LM: {S_lm:.3e}")
    print(f"real phase shift RK: {delta_rk:.3e} degrees")
    print(f"real phase shift LM: {delta_lm:.3e} degrees")
    print(f"complex phase shift RK: {atten_rk:.3e} degrees")
    print(f"complex phase shift LM: {atten_lm:.3e} degrees")

    plt.plot(r_values, np.real(u_rk), "k", label="Runge-Kutta")
    plt.plot(r_values, np.imag(u_rk), ":k")

    plt.plot(r_values, np.real(u_lm), "r", label="Lagrange-Legendre")
    plt.plot(r_values, np.imag(u_lm), ":r")

    plt.legend()
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$u_{%d} (r) $ [a.u.]" % se.l)
    plt.tight_layout()
    plt.show()


def rmse_RK_LM():
    # Woods-Saxon potential parameters
    V0 = 60  # real potential strength
    W0 = 20  # imag potential strength
    R0 = 4  # Woods-Saxon potential radius
    a0 = 0.5  # Woods-Saxon potential diffuseness
    params = (V0, W0, R0, a0)

    nodes_within_radius = 5

    lgrid = np.arange(0, 10)
    egrid = np.linspace(0.01, 100, 100)

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
    plt.ylim([0, 1])
    plt.yscale("log")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    channel_radius_dependence_test()
    local_interaction_example()
    # rmse_RK_LM()
