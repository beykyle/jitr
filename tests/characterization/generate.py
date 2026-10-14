"""Generate characterization goldens from the jitr v3 (numba) engine.

The v4 rewrite must reproduce these numbers to round-off. Each golden stores
the *inputs* (potential arrays on the quadrature grid, kinematics) next to the
outputs, so the v4 tests feed identical arrays and never depend on the
optical-potential modules.

This script imports the v3 API (``jitr.rmatrix.Solver``, the v3 workspaces)
and therefore only runs on a v3 checkout. It was run once, at commit e56415c
(jitr main, v3.x); it is kept for provenance.

Conventions that make the comparison exact:

* early-exit tolerances are off (``smatrix_abs_tol = tmatrix_abs_tol = 0``),
  so every partial wave up to ``lmax`` is solved;
* one channel radius in fm and one basis size are shared by every energy, so
  a single v4 workspace sees the same grid as every v3 workspace.

Run as ``python tests/characterization/generate.py``.
"""

from pathlib import Path

import numpy as np

from jitr import reactions, rmatrix
from jitr.optical_potentials import kduq
from jitr.optical_potentials.dispersion import DispersionSolver
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)
from jitr.quadrature import Kernel
from jitr.reactions import Reaction
from jitr.reactions.wavefunction import Wavefunctions
from jitr.xs import elastic, lane_pn, quasielastic_pn

DATA = Path(__file__).with_name("data")


def kinematics_arrays(kins):
    return {
        f"kin_{name}": np.array([getattr(k, name) for k in kins])
        for name in ("Elab", "Ecm", "mu", "k", "eta")
    }


def elastic_p48ca():
    """p + 48Ca with KD03 at four energies, semi-relativistic kinematics."""
    reaction = Reaction(target=(48, 20), projectile=(1, 1), process="El")
    energies = np.array([10.0, 25.0, 50.0, 100.0])
    radius_fm, nbasis, lmax = 15.0, 60, 30
    angles = np.linspace(1e-3, np.pi - 1e-3, 91)
    params = kduq.get_kd03((1, 1))
    model = kduq.KDUQ((1, 1))

    kins, out = [], {k: [] for k in ("central", "spin_orbit", "coulomb")}
    res = {k: [] for k in ("splus", "sminus", "dsdo", "Ay", "Q", "t", "rxn")}
    for E in energies:
        kin = reaction.kinematics(E)
        ws = elastic.DifferentialWorkspace.build_from_system(
            reaction, kin, radius_fm, rmatrix.Solver(nbasis), lmax, angles, 0.0
        )
        r = ws.radial_grid()
        central, spin_orbit, coulomb = model.evaluate(r, reaction, kin, *params)
        splus, sminus = ws.integral_workspace.smatrix(central, spin_orbit, coulomb)
        xs = ws.xs(central, spin_orbit, coulomb)
        kins.append(kin)
        out["central"].append(central)
        out["spin_orbit"].append(spin_orbit)
        out["coulomb"].append(coulomb * np.ones_like(r))
        res["splus"].append(splus)
        res["sminus"].append(sminus)
        for name in ("dsdo", "Ay", "Q", "t", "rxn"):
            res[name].append(getattr(xs, name))
    return dict(
        r=r,
        angles=angles,
        radius_fm=radius_fm,
        nbasis=nbasis,
        lmax=lmax,
        **kinematics_arrays(kins),
        **{k: np.array(v) for k, v in out.items()},
        **{k: np.array(v) for k, v in res.items()},
    )


def elastic_alpha44ca():
    """alpha + 44Ca at 29 MeV with the quickstart Woods-Saxon form."""
    reaction = Reaction(target=(44, 20), projectile=(4, 2), process="El")
    kin = reaction.kinematics(29.0)
    radius_fm, nbasis, lmax = 14.0, 60, 40
    angles = np.linspace(np.deg2rad(5), np.deg2rad(175), 86)
    ws = elastic.DifferentialWorkspace.build_from_system(
        reaction, kin, radius_fm, rmatrix.Solver(nbasis), lmax, angles, 0.0
    )
    r = ws.radial_grid()
    A13 = 44 ** (1 / 3)
    central = (
        -150.0 * woods_saxon_safe(r, 1.25 * A13, 0.7)
        - 1j * 20.0 * woods_saxon_safe(r, 1.4 * A13, 0.6)
        - 1j * 4.0 * (-4 * 0.6) * woods_saxon_prime_safe(r, 1.4 * A13, 0.6)
    )
    coulomb = coulomb_charged_sphere(r, 40, 1.3 * A13)
    splus, sminus = ws.integral_workspace.smatrix(central, None, coulomb)
    xs = ws.xs(central, None, coulomb)
    return dict(
        r=r,
        angles=angles,
        radius_fm=radius_fm,
        nbasis=nbasis,
        lmax=lmax,
        **kinematics_arrays([kin]),
        central=central,
        coulomb=coulomb,
        splus=splus,
        sminus=sminus,
        dsdo=xs.dsdo,
        rutherford=ws.rutherford,
        t=xs.t,
        rxn=xs.rxn,
    )


def _thomas(r, depth, R, a):
    x = np.exp((r - R) / a)
    return -depth * x / (1 + x) ** 2 / (a * r)


def pn_48ca():
    """48Ca(p,n)48Sc(IAS) at 35 MeV: DWBA and coupled-channels Lane."""
    reaction = Reaction((48, 20), (1, 1), (1, 0), (48, 21))
    ke = reaction.kinematics(35.0, relativistic=False)
    kx = reaction.kinematics_exit(ke, 6.67, relativistic=False)
    radius_fm, nbasis, lmax = 14.0, 35, 15
    angles = np.linspace(1e-3, np.pi - 1e-3, 181)
    solver = rmatrix.Solver(nbasis)
    dwba = quasielastic_pn.Workspace(
        reaction, ke, kx, solver, angles, lmax, radius_fm, tmatrix_abs_tol=0
    )
    cc = lane_pn.Workspace(reaction, ke, kx, solver, angles, lmax, radius_fm)
    r = dwba.radial_grid()
    p = {
        "U_p_coulomb": coulomb_charged_sphere(r, 20, 4.7),
        "U_p_central": (-50.0 - 8.0j) * woods_saxon_safe(r, 4.4, 0.65),
        "U_p_spin_orbit": _thomas(r, 6.0, 4.0, 0.6),
        "U_n_central": (-46.0 - 8.0j) * woods_saxon_safe(r, 4.4, 0.65),
        "U_n_spin_orbit": _thomas(r, 5.5, 4.0, 0.6),
    }
    Tpn, Sn, Sp = dwba.tmatrix(**p)
    obs_dwba = dwba.observables(**p)
    R_cc, S_cc = cc.rsmatrix(**p)
    obs_cc = cc.observables(**p)
    return dict(
        r=r,
        angles=angles,
        radius_fm=radius_fm,
        nbasis=nbasis,
        lmax=lmax,
        **{f"p_{k}": v for k, v in kinematics_arrays([ke]).items()},
        **{f"n_{k}": v for k, v in kinematics_arrays([kx]).items()},
        isovector_factor=dwba.isovector_factor,
        **p,
        Tpn=Tpn,
        Sn=Sn,
        Sp=Sp,
        dwba_dsdo=obs_dwba.dsdo,
        dwba_Ay=obs_dwba.Ay,
        dwba_Q=obs_dwba.Q,
        cc_R=R_cc,
        cc_S=S_cc,
        cc_dsdo=obs_cc.dsdo,
        cc_Ay=obs_cc.Ay,
        cc_Q=obs_cc.Q,
        cc_integrated=cc.integrated_xs_from_smatrix(S_cc),
    )


def wavefunction_3ch():
    """Three coupled channels with different k, mu and eta; general weights."""
    nbasis, l = 30, 1
    k = np.array([1.27, 1.13, 0.95])
    mu = np.array([918.96, 920.21, 1870.0])
    eta = np.array([0.5, 0.0, 0.8])
    a = 5 * np.pi  # dimensionless k_0 a
    system = reactions.ProjectileTargetSystem(
        channel_radius=a, lmax=l, coupling=lambda _: np.eye(3)
    )
    channels, asymptotics = system.get_partial_wave_channels(0.0, 0.0, mu, k, eta)
    ch, asym = channels[l], asymptotics[l]
    solver = rmatrix.Solver(nbasis)
    r = solver.radial_grid(ch.a, ch.k[0])
    V = np.zeros((3, 3, nbasis), dtype=np.complex128)
    for i in range(3):
        V[i, i] = (-40.0 - 3.0j) * (1 + 0.1 * i) * np.exp(-r / 4)
    for i in range(2):
        V[i, i + 1] = V[i + 1, i] = -5.0 * np.exp(-r / 4)
    weights = np.array([0.6, 0.8, 0.3])
    R, S, coeffs, uext_prime = solver.solve(
        ch, asym, local_potential=V, weights=weights, wavefunction=True
    )
    wf = Wavefunctions(solver, coeffs, S, uext_prime, ch, weights)
    # interior u(s) = sum_n coeffs[n] f_n(s) / a; at a mesh node f_n is
    # 1/sqrt(w_n) (evaluating f_n there directly is 0/0)
    u_nodes = coeffs / ch.a / np.sqrt(solver.kernel.quadrature.weights)
    u_a = np.array([u(ch.a) for u in wf.uint()])
    return dict(
        r=r,
        radius_fm=ch.a / ch.k[0],
        nbasis=nbasis,
        l=l,
        k=k,
        mu=mu,
        eta=eta,
        V=V,
        weights_raw=weights,
        R_rho=R,
        S=S,
        u_nodes=u_nodes,
        u_a=u_a,
    )


def transforms_and_grid():
    """Mesh, kinetic matrix and Fourier-Bessel transforms on the Legendre mesh."""
    nbasis, radius = 30, 10.0
    kernel = Kernel(nbasis)
    quad = kernel.quadrature
    r = kernel.radial_grid(radius)
    f = np.exp(-((r - 3.0) ** 2)) * (1 + 0.5j)
    g = np.exp(-(np.subtract.outer(r, r) ** 2) - 0.1 * np.add.outer(r, r))
    kgrid = np.linspace(0.05, 3.0, 25)
    return dict(
        nbasis=nbasis,
        radius=radius,
        x=quad.abscissa,
        w=quad.weights,
        F=quad.kinetic_matrix(1.0, 0),
        b=np.array([kernel.f(n, 1.0, 1.0) for n in range(1, nbasis + 1)]).real,
        f=f,
        g=g,
        kgrid=kgrid,
        fb0=kernel.fourier_bessel_transform(0, f, kgrid, radius),
        fb2=kernel.fourier_bessel_transform(2, f, kgrid, radius),
        dfb1=kernel.double_fourier_bessel_transform(1, g, kgrid, radius),
    )


def dom_dispersion():
    """Dispersion correction of a surface DOM imaginary potential."""
    from jitr.optical_potentials import dom

    def W(r, x):
        E2 = x * x
        a = 0.30 + 0.30 * E2 * E2 / (E2 * E2 + 12.0**4)
        return dom.Ws_depth(x, 16.2, 12.5, 0.0214) / np.cosh((r - 4.55) / (2 * a)) ** 2

    r = np.linspace(0.1, 12.0, 40)
    out = {"r": r}
    for E in (-12.3, 5.7, 41.1):
        solver = DispersionSolver(r, E)
        W_grid = np.array([[W(ri, x) for x in solver.x_quad] for ri in r])
        W_at_E = np.array([W(ri, E) for ri in r])
        out[f"dV_{E}"] = solver(W_grid, W_at_E)
    return out


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, make in [
        ("elastic_p48ca", elastic_p48ca),
        ("elastic_alpha44ca", elastic_alpha44ca),
        ("pn_48ca", pn_48ca),
        ("wavefunction_3ch", wavefunction_3ch),
        ("transforms_and_grid", transforms_and_grid),
        ("dom_dispersion", dom_dispersion),
    ]:
        np.savez_compressed(DATA / f"{name}.npz", **make())
        print(f"wrote {name}")
