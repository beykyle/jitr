"""Single-channel scattering on a local potential: the Lagrange-mesh R-matrix
solution against a direct (Runge-Kutta) integration of the radial equation.

p + 48Ca with a Woods-Saxon plus Coulomb potential: the s-wave S-matrix and
wavefunction, the (in)dependence on the channel radius, and the relative error
over a grid of 200 energies solved by one vectorized workspace.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_potential,
)
from jitr.reactions import Level
from jitr.utils import delta, radial_equation
from jitr.utils.free_solutions import coulomb_hankel_table
from jitr.utils.kinematics import classical_kinematics, level_kinematics
from jitr.xs import coupled

MASS_CA48, MASS_P, ZZ = 44657.26581995028, 938.271653086152, 20


def potential(r, V0=60.0, W0=20.0, R0=4.0, a0=0.5):
    return -woods_saxon_potential(r, V0, W0, R0, a0) + coulomb_charged_sphere(r, ZZ, R0)


def workspace(Elab, radius=3 * 2 * np.pi / 0.8, nbasis=100, lmax=0):
    """One level, spin 0: every J^pi block is the single partial wave l = J."""
    kin = classical_kinematics(MASS_CA48, MASS_P, np.atleast_1d(Elab), ZZ)
    k2, mu, eta = level_kinematics(kin, [0.0])
    return kin, coupled.Workspace(radius, nbasis, [Level(0)], lmax, k2, mu, eta)


def runge_kutta(l, k, mu, eta, a, r0=1e-3):
    """S from integrating outward to a and matching to Coulomb functions."""
    sol = solve_ivp(
        radial_equation,
        (r0, a),
        np.array([r0 ** (l + 1), (l + 1) * r0**l], complex),
        args=(l, k, mu, potential),
        dense_output=True,
        rtol=1e-11,
        atol=1e-14,
    ).sol
    u, du = sol(a)
    Hp, Hm, Hpp, Hmp = (h[l] for h in coulomb_hankel_table(k * a, eta, l))
    return (du * Hm - k * u * Hmp) / (du * Hp - k * u * Hpp), sol


def local_interaction_example():
    kin, ws = workspace(14.1)
    S, u, _ = ws.wavefunctions(local=[(np.eye(1), potential(ws.r))])
    S_lm, u_lm = complex(S[0, 0, 0, 0]), np.asarray(u[0, 0, 0, 0])
    S_rk, sol = runge_kutta(0, kin.k[0], kin.mu, kin.eta[0], ws.mesh.a)
    u_rk = sol(ws.r)[0]
    u_rk *= u_lm[20] / u_rk[20]  # the integration has arbitrary normalization

    for name, S_ in (("Runge-Kutta", S_rk), ("Lagrange mesh", S_lm)):
        d, att = delta(S_)
        print(
            f"{name:14s} S = {S_:.6f}, phase shift {d:.4f}, attenuation {att:.4f} deg"
        )

    plt.plot(ws.r, u_rk.real, "k", alpha=0.5, label="Runge-Kutta")
    plt.plot(ws.r, u_rk.imag, ":k", alpha=0.5)
    plt.plot(ws.r, u_lm.real, "r.", alpha=0.5, label="Lagrange-Legendre")
    plt.plot(ws.r, u_lm.imag, "r.", alpha=0.3)
    plt.legend()
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$u_0(r)$ [a.u.]")
    plt.tight_layout()
    plt.show()


def channel_radius_dependence():
    """Beyond the range of the nuclear potential, S does not depend on a."""
    radii = np.linspace(8, 30, 23)
    deltas = []
    for a in radii:
        _, ws = workspace(14.1, radius=a, nbasis=int(5 * a), lmax=3)
        S, _ = ws.smatrix(local=[(np.eye(1), potential(ws.r))])
        deltas.append([complex(*delta(S[0, b, 0, 0])) for b in range(4)])
    deltas = np.array(deltas)
    for l in range(4):
        plt.plot(radii, deltas[:, l].real, label=rf"$l={l}$")
    plt.legend()
    plt.xlabel("channel radius [fm]")
    plt.ylabel(r"$\mathfrak{Re}\,\delta_l$ [degrees]")
    plt.show()


def relative_error_over_energy():
    """All energies in one workspace call, against Runge-Kutta energy by energy."""
    energies = np.linspace(0.5, 120, 200)
    kin, ws = workspace(energies, radius=5 * 2 * np.pi / 0.8, nbasis=60, lmax=2)
    S, _ = ws.smatrix(local=[(np.eye(1), potential(ws.r))])
    S = np.asarray(S[:, :, 0, 0])
    error = np.zeros((3, energies.size))
    for e in range(energies.size):
        for l in range(3):
            S_rk, _ = runge_kutta(l, kin.k[e], kin.mu, kin.eta[e], ws.mesh.a)
            error[l, e] = abs(S_rk - S[e, l]) / abs(S_rk)
    for l in range(3):
        plt.plot(energies, 100 * error[l], label=rf"$l = {l}$")
    plt.ylabel(r"$|S^{\rm RK}_l - S^{\rm LM}_l| / |S^{\rm RK}_l|$ [%]")
    plt.xlabel(r"$E_{\rm lab}$ [MeV]")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    local_interaction_example()
    channel_radius_dependence()
    relative_error_over_energy()
