"""Case construction shared by the golden generator and characterization tests.

Each ``compute_*`` function runs the *current* engine and returns a flat dict
of numpy arrays. ``generate_goldens.py`` saves these dicts to ``data/*.npz``;
the ``test_char_*`` modules recompute them and compare against the saved
goldens. The goldens pin pre-rewrite behavior so the lax-backed engine can be
validated against per-energy slices (design doc §6, Phase 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# --- elastic case: p + 48Ca, KDUQ defaults -------------------------------

ELASTIC_TARGET = (48, 20)
ELASTIC_PROJECTILE = (1, 1)  # proton
ELASTIC_ELAB = np.array([10.0, 25.0, 35.0, 50.0])
ELASTIC_LMAX = 10
ELASTIC_CHANNEL_RADIUS_FM = 12.0
ELASTIC_NBASIS = 40
ELASTIC_ANGLES = np.linspace(0.1, np.pi - 0.1, 19)


def _kduq_default_params(projectile: tuple[int, int]) -> np.ndarray:
    from jitr.optical_potentials import kduq

    return np.array(list(kduq.Global(projectile).params.values()))


def compute_elastic_case() -> dict[str, np.ndarray]:
    """S-matrices and observables per energy for the elastic golden."""
    from jitr.optical_potentials.kduq import KDUQ
    from jitr.reactions import ElasticReaction
    from jitr.rmatrix import Solver
    from jitr.xs.elastic import DifferentialWorkspace, IntegralWorkspace

    reaction = ElasticReaction(ELASTIC_TARGET, ELASTIC_PROJECTILE)
    omp = KDUQ(ELASTIC_PROJECTILE)
    params = _kduq_default_params(ELASTIC_PROJECTILE)
    solver = Solver(ELASTIC_NBASIS)

    n_e = ELASTIC_ELAB.size
    out: dict[str, np.ndarray] = {
        "Elab": ELASTIC_ELAB,
        "angles": ELASTIC_ANGLES,
        "Ecm": np.zeros(n_e),
        "k": np.zeros(n_e),
        "mu": np.zeros(n_e),
        "eta": np.zeros(n_e),
        "Splus": np.zeros((n_e, ELASTIC_LMAX + 1), dtype=np.complex128),
        "Sminus": np.zeros((n_e, ELASTIC_LMAX + 1), dtype=np.complex128),
        "Tplus": np.zeros((n_e, ELASTIC_LMAX + 1)),
        "Tminus": np.zeros((n_e, ELASTIC_LMAX + 1)),
        "dsdo": np.zeros((n_e, ELASTIC_ANGLES.size)),
        "Ay": np.zeros((n_e, ELASTIC_ANGLES.size)),
        "Q": np.zeros((n_e, ELASTIC_ANGLES.size)),
        "t": np.zeros(n_e),
        "rxn": np.zeros(n_e),
    }

    for i, elab in enumerate(ELASTIC_ELAB):
        kinematics = reaction.kinematics(float(elab))
        integral = IntegralWorkspace(
            reaction=reaction,
            kinematics=kinematics,
            channel_radius_fm=ELASTIC_CHANNEL_RADIUS_FM,
            solver=solver,
            lmax=ELASTIC_LMAX,
            smatrix_abs_tol=0.0,  # defeat the per-l early exit: full-length S
        )
        differential = DifferentialWorkspace(integral, ELASTIC_ANGLES)
        rgrid = integral.radial_grid()
        central, spin_orbit, coulomb = omp.evaluate(
            rgrid, reaction, kinematics, *params
        )

        out["Ecm"][i] = kinematics.Ecm
        out["k"][i] = kinematics.k
        out["mu"][i] = kinematics.mu
        out["eta"][i] = kinematics.eta

        splus, sminus = integral.smatrix(central, spin_orbit, coulomb)
        out["Splus"][i] = splus
        out["Sminus"][i] = sminus
        tplus, tminus = integral.transmission_coefficients(central, spin_orbit, coulomb)
        out["Tplus"][i] = tplus
        out["Tminus"][i] = tminus

        xs = differential.xs(central, spin_orbit, coulomb)
        out["dsdo"][i] = xs.dsdo
        out["Ay"][i] = xs.Ay
        out["Q"][i] = xs.Q
        out["t"][i] = xs.t
        out["rxn"][i] = xs.rxn

    return out


# --- quasi-elastic (p,n) case: 48Ca(p,n)48Sc IAS, KDUQ defaults ----------
# Mirrors examples/notebooks/chex_jitr_validation.ipynb (the chex_qepn_xs.txt
# case) and the lax acceptance fixture (Elab=35, lmax=20, a=16 fm, nbasis=35).

QEPN_ELAB = 35.0
QEPN_E_IAS = 6.67
QEPN_LMAX = 20
QEPN_CHANNEL_RADIUS_FM = 16.0
QEPN_ANGLES = np.linspace(0.01, np.pi, 180)


def compute_qepn_case() -> dict[str, np.ndarray]:
    from jitr.optical_potentials.kduq import KDUQ
    from jitr.reactions import Reaction
    from jitr.rmatrix import Solver
    from jitr.utils import suggested_basis_size
    from jitr.xs.quasielastic_pn import Workspace

    proton = (1, 1)
    neutron = (1, 0)
    reaction = Reaction(
        target=(48, 20), projectile=proton, product=neutron, residual=(48, 21)
    )
    reaction_exit = Reaction(
        target=reaction.residual, projectile=reaction.product, process="El"
    )

    kinematics_entrance = reaction.kinematics(QEPN_ELAB)
    kinematics_exit = reaction.kinematics_exit(
        kinematics_entrance, residual_excitation_energy=QEPN_E_IAS
    )
    nbasis = suggested_basis_size(QEPN_CHANNEL_RADIUS_FM * kinematics_entrance.k)

    workspace = Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        Solver(nbasis),
        QEPN_ANGLES,
        QEPN_LMAX,
        QEPN_CHANNEL_RADIUS_FM,
        tmatrix_abs_tol=0.0,  # defeat the per-l early exit
    )

    rgrid = workspace.radial_grid()
    params_p = _kduq_default_params(proton)
    params_n = _kduq_default_params(neutron)
    U_p_central, U_p_spin_orbit, U_p_coulomb = KDUQ(proton).evaluate(
        rgrid, reaction, kinematics_entrance, *params_p
    )
    U_n_central, U_n_spin_orbit, _ = KDUQ(neutron).evaluate(
        rgrid, reaction_exit, kinematics_exit, *params_n
    )

    Tpn, Sn, Sp = workspace.tmatrix(
        U_p_coulomb, U_p_central, U_p_spin_orbit, U_n_central, U_n_spin_orbit
    )
    xs = workspace.xs(
        U_p_coulomb, U_p_central, U_p_spin_orbit, U_n_central, U_n_spin_orbit
    )

    return {
        "Elab": np.array(QEPN_ELAB),
        "E_IAS": np.array(QEPN_E_IAS),
        "nbasis": np.array(nbasis),
        "angles": QEPN_ANGLES,
        "rgrid": rgrid,
        "Ecm_p": np.array(kinematics_entrance.Ecm),
        "k_p": np.array(kinematics_entrance.k),
        "mu_p": np.array(kinematics_entrance.mu),
        "eta_p": np.array(kinematics_entrance.eta),
        "Ecm_n": np.array(kinematics_exit.Ecm),
        "k_n": np.array(kinematics_exit.k),
        "mu_n": np.array(kinematics_exit.mu),
        "Tpn": Tpn,
        "Sn": Sn,
        "Sp": Sp,
        "xs": np.asarray(xs),
    }


# --- folding (ILDAFolder) -------------------------------------------------


def compute_folding_case() -> dict[str, np.ndarray]:
    from jitr.folding import ILDAFolder

    folder = ILDAFolder(r_max=18.0, n_quad=400)
    rho_q = 0.11 * np.exp(-((folder.r_q / 2.1) ** 2))
    u_q = -40.0 * np.exp(-((folder.r_q / 1.8) ** 2))
    r_out = np.linspace(0.0, 10.0, 41)

    return {
        "r_out": r_out,
        "Z": np.array(folder.Z_from_density(rho_q)),
        "rms": np.array(folder.rms_radius(rho_q)),
        "v_coulomb": folder.V_coulomb(rho_q, mode="density", r_out=r_out),
        "v_coulomb_exchange": folder.V_coulomb(
            rho_q, mode="density", include_exchange=True, r_out=r_out
        ),
        "gaussian_fold": folder.gaussian_fold(u_q, t=1.2, r_out=r_out),
    }


# --- quadrature transforms (Fourier-Bessel) -------------------------------

TRANSFORMS_NBASIS = 80
TRANSFORMS_RADIUS = np.pi
TRANSFORMS_KGRID = np.array([0.0, 0.2, 0.35, 0.75, 0.9, 1.3, 1.7])


def compute_transforms_case() -> dict[str, np.ndarray]:
    from jitr import quadrature

    kernel = quadrature.Kernel(TRANSFORMS_NBASIS, basis="Legendre")
    rgrid = kernel.radial_grid(TRANSFORMS_RADIUS)
    gaussian = np.exp(-0.3 * rgrid**2)

    return {
        "k_grid": TRANSFORMS_KGRID,
        "rgrid": rgrid,
        "fb_l0_linear": kernel.fourier_bessel_transform(
            0, rgrid, TRANSFORMS_KGRID, TRANSFORMS_RADIUS
        ),
        "fb_l1_gaussian": kernel.fourier_bessel_transform(
            1, gaussian, TRANSFORMS_KGRID, TRANSFORMS_RADIUS
        ),
        "dfb_l0_separable": kernel.double_fourier_bessel_transform(
            0, np.outer(rgrid, rgrid), TRANSFORMS_KGRID, TRANSFORMS_RADIUS
        ),
    }


# --- dispersive OMP kernel -------------------------------------------------

DISPERSION_E = 8.7
DISPERSION_RGRID = np.linspace(0.1, 12.0, 60)


def _dom_surface_W(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """DOM-style surface imaginary term W_d(x)·sech²((r−R)/2a(x)), (N_r, N_q)."""
    from jitr.optical_potentials import dom

    Wd = dom.Ws_depth(x, 16.2, 12.5, 0.0214)
    x2 = x * x
    a = 0.30 + 0.30 * x2 * x2 / (x2 * x2 + 12.0**4)
    return Wd / np.cosh((r[:, None] - 4.55) / (2.0 * a)) ** 2


def compute_dispersion_case() -> dict[str, np.ndarray]:
    from jitr.optical_potentials.dispersion import DispersionSolver

    solver = DispersionSolver(DISPERSION_RGRID, DISPERSION_E)
    W_grid = _dom_surface_W(DISPERSION_RGRID, solver.x_quad)
    W_at_E = _dom_surface_W(DISPERSION_RGRID, np.array([DISPERSION_E]))[:, 0]

    return {
        "rgrid": DISPERSION_RGRID,
        "E": np.array(DISPERSION_E),
        "W_grid": W_grid,
        "W_at_E": W_at_E,
        "dV": solver(W_grid, W_at_E),
    }


# --- grids + one wavefunction ----------------------------------------------


def compute_grid_case() -> dict[str, np.ndarray]:
    """Public radial grids per elastic energy + one wavefunction.

    The public grid is energy-independent fm (the k-scaling cancels) and
    matches lax's ``MeshSpec("legendre", "x")`` radii bitwise — pinned here.
    The wavefunction is the Phase 6 oracle for the rebuilt
    ``reactions.wavefunction`` (convention conversion per lax DESIGN.md
    Appendix C.12 expected).
    """
    from jitr.optical_potentials import potential_forms as potentials
    from jitr.reactions import ElasticReaction, wavefunction
    from jitr.rmatrix import Solver
    from jitr.xs.elastic import IntegralWorkspace

    reaction = ElasticReaction(ELASTIC_TARGET, ELASTIC_PROJECTILE)
    solver = Solver(ELASTIC_NBASIS)
    out: dict[str, np.ndarray] = {"Elab": ELASTIC_ELAB}
    for i, elab in enumerate(ELASTIC_ELAB):
        kinematics = reaction.kinematics(float(elab))
        workspace = IntegralWorkspace(
            reaction=reaction,
            kinematics=kinematics,
            channel_radius_fm=ELASTIC_CHANNEL_RADIUS_FM,
            solver=solver,
            lmax=ELASTIC_LMAX,
        )
        out[f"rgrid_{i}"] = workspace.radial_grid()

    # one wavefunction: l=1, Elab=35 MeV, complex Woods-Saxon + Coulomb
    from jitr.reactions import ProjectileTargetSystem
    from jitr.utils import kinematics as kin_mod

    sys = ProjectileTargetSystem(
        channel_radius=5 * (2 * np.pi),
        lmax=ELASTIC_LMAX,
        mass_target=reaction.target.m0,
        mass_projectile=reaction.projectile.m0,
        Ztarget=reaction.target.Z,
        Zproj=reaction.projectile.Z,
    )
    channel_kinematics = kin_mod.classical_kinematics(
        sys.mass_target, sys.mass_projectile, 35.0, sys.Zproj * sys.Ztarget
    )
    channels, asymptotics = sys.get_partial_wave_channels(*channel_kinematics)
    l = 1
    ch = channels[l]
    asym = asymptotics[l]
    wf_solver = Solver(60)
    rgrid = wf_solver.radial_grid(ch.a, ch.k[0])
    local = -potentials.woods_saxon_potential(
        rgrid, 70.0, 40.0, 6.0, 1.2
    ) + potentials.coulomb_charged_sphere(rgrid, sys.Zproj * sys.Ztarget, 6.0)
    R, S, x, uext_prime_boundary = wf_solver.solve(
        ch, asym, local_potential=local, wavefunction=True
    )
    s_values = np.linspace(0.05, sys.channel_radius, 200)
    u = wavefunction.Wavefunctions(wf_solver, x, S, uext_prime_boundary, ch).uint()[0](
        s_values
    )

    out["wf_l"] = np.array(l)
    out["wf_Elab"] = np.array(35.0)
    out["wf_s_values"] = s_values
    out["wf_u"] = np.asarray(u, dtype=np.complex128)
    out["wf_S"] = np.asarray(S)
    out["wf_R"] = np.asarray(R)
    return out


CASES = {
    "elastic_pca": compute_elastic_case,
    "qepn_ca48": compute_qepn_case,
    "folding_ilda": compute_folding_case,
    "transforms_fb": compute_transforms_case,
    "dispersion_dom": compute_dispersion_case,
    "grids": compute_grid_case,
}


def load_golden(name: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"golden {path} missing — run tests/characterization/generate_goldens.py"
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def assert_matches_golden(
    computed: dict[str, np.ndarray],
    golden: dict[str, np.ndarray],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    assert set(computed) == set(
        golden
    ), f"key mismatch: {sorted(set(computed) ^ set(golden))}"
    for key, value in golden.items():
        np.testing.assert_allclose(
            computed[key], value, rtol=rtol, atol=atol, err_msg=f"field {key!r}"
        )
