"""Phase 0 spike: lax blocked solver ≡ current rmatrix engine, per (l, j, E).

Transition-window test (design doc §6): deleted with the old engine in
Phase 7.

Kinematics mapping (spike finding, supersedes design doc §3.2 items 2-3):
the old engine solves, in s = k·r units, ``[T_s + V/Ecm]ψ = ψ`` with
boundary H±(k·R, η) — for a general ``ChannelKinematics`` the tuple
(Ecm, μ, k, η) is over-determined and only classical kinematics satisfy
ħ²k²/(2μ) = Ecm. lax derives k = √(E/mf) and η = z₁z₂e²/(2·mf·k). The
faithful mapping for arbitrary kinematics is therefore

    mf_e       = ħ²c²/(2 μ_e)          (matches η given k)
    E_lax,e    = mf_e · k_e²            (matches k; ≠ Ecm when semi-rel)
    V_lax,e    = V · E_lax,e / Ecm_e    (matches the interior V/Ecm scale)

For classical kinematics E_lax = Ecm and the scale is exactly 1, recovering
the doc's recipe. The alternative — dropping the V rescale — would be the
pure Ingemarsson convention (what Frescox/TALYS do in relativistic mode) but
would *change* jitr's semi-relativistic results by ~(E_lax/Ecm − 1) ≈ 2-3%
at tens of MeV; rejected here to preserve characterization behavior.

Spin-orbit convention: the engine multiplies user arrays by
⟨l·σ⟩ = {l, −(l+1)} (``spin_half_orbit_coupling``), not the doc §3.1's
⟨l·s⟩; pinned by these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from jitr.optical_potentials import kduq
from jitr.reactions import ElasticReaction
from jitr.utils.constants import HBARC
from jitr.utils.kinematics import classical_kinematics

from ..characterization._cases import (
    ELASTIC_CHANNEL_RADIUS_FM,
    ELASTIC_ELAB,
    ELASTIC_LMAX,
    ELASTIC_NBASIS,
    ELASTIC_PROJECTILE,
    ELASTIC_TARGET,
    load_golden,
)
from ..conftest import requires_lax

pytestmark = requires_lax

LMAX = ELASTIC_LMAX
NBASIS = ELASTIC_NBASIS
RADIUS = ELASTIC_CHANNEL_RADIUS_FM


def _ldots(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """⟨l·σ⟩ per l for j = l ± ½ (jitr's spin_half_orbit_coupling values)."""
    ls = np.arange(lmax + 1, dtype=np.float64)
    return ls, -(ls + 1.0)


def _compile_blocked(
    energies: np.ndarray,
    mass_factors: np.ndarray,
    z1z2: tuple[int, int] | None,
    *,
    energy_dependent: bool = False,
):
    import lax

    uniform = bool(np.all(mass_factors == mass_factors[0]))
    return lax.compile(
        mesh=lax.MeshSpec("legendre", "x", n=NBASIS, scale=RADIUS),
        blocks=[
            [lax.ChannelSpec(l=ell, threshold=0.0, mass_factor=float(mass_factors[0]))]
            for ell in range(LMAX + 1)
        ],
        solvers=("spectrum", "smatrix"),
        energies=np.asarray(energies, dtype=np.float64),
        energy_dependent=energy_dependent,
        mass_factor_grid=None if uniform else np.asarray(mass_factors),
        z1z2=z1z2,
        V_is_complex=True,
        method="eig",
    )


def _lax_interactions(solver, central, spin_orbit, *, energy_dependent: bool):
    """Build (V⁺, V⁻) Interactions; ``central`` includes Coulomb."""
    import jax.numpy as jnp

    ldots_plus, ldots_minus = _ldots(LMAX)
    ia = solver.interaction_from_array
    v_central = ia(local=[jnp.asarray(central)], energy_dependent=energy_dependent)
    # block-dependent SO: (N_b, [N_E,] N) via outer product on the leading axis
    expand = (slice(None),) + (None,) * np.asarray(spin_orbit).ndim
    so_plus = ldots_plus[expand] * np.asarray(spin_orbit)[None]
    so_minus = ldots_minus[expand] * np.asarray(spin_orbit)[None]
    v_plus = v_central + ia(
        local=[jnp.asarray(so_plus)],
        block_dependent=True,
        energy_dependent=energy_dependent,
    )
    v_minus = v_central + ia(
        local=[jnp.asarray(so_minus)],
        block_dependent=True,
        energy_dependent=energy_dependent,
    )
    return v_plus, v_minus


def _lax_smatrix_static(solver, v) -> np.ndarray:
    """(N_b, N_E) from the static-V spectral path."""
    return np.asarray(solver.smatrix(solver.spectrum(v)))[:, :, 0, 0]


def _lax_smatrix_grid(solver, v) -> np.ndarray:
    """(N_b, N_E) from the energy-dependent aligned-grid path."""
    return np.asarray(solver.smatrix_grid(solver.spectrum(v)))[:, :, 0, 0]


def _old_engine_smatrix(
    kinematics_list, central, spin_orbit, coulomb
) -> tuple[np.ndarray, np.ndarray]:
    """Per-energy, per-(l, j) loop over the legacy rmatrix engine.

    Replicates the pre-rewrite ``IntegralWorkspace.smatrix`` (deleted in
    Phase 3) directly on ``jitr.rmatrix`` internals; removed with the old
    engine in Phase 7. Returns (N_E, lmax+1) per j.
    """
    from jitr.reactions import ProjectileTargetSystem, spin_half_orbit_coupling
    from jitr.rmatrix import Solver

    reaction = ElasticReaction(ELASTIC_TARGET, ELASTIC_PROJECTILE)
    solver = Solver(NBASIS)
    splus = np.zeros((len(kinematics_list), LMAX + 1), dtype=np.complex128)
    sminus = np.zeros_like(splus)
    for i, kin in enumerate(kinematics_list):
        a = RADIUS * kin.k
        sys = ProjectileTargetSystem(
            a,
            LMAX,
            mass_target=reaction.target.m0,
            mass_projectile=reaction.projectile.m0,
            Ztarget=reaction.target.Z,
            Zproj=reaction.projectile.Z,
            coupling=spin_half_orbit_coupling,
        )
        free_matrices = solver.free_matrix(a, sys.l, coupled=False)
        basis_boundary = solver.precompute_boundaries(a)
        channels, asymptotics = sys.get_partial_wave_channels(*kin)
        channels = [ch.decouple() for ch in channels]
        asymptotics = [asym.decouple() for asym in asymptotics]
        l_dot_s = np.array([np.diag(c) for c in sys.couplings[1:]])

        def im(values, ch):
            return solver.interaction_matrix(
                ch.k[0], ch.E[0], ch.a, ch.size, local_potential=values
            )

        ch0 = channels[0][0]
        im_central = im(central + coulomb, ch0)
        im_so = im(spin_orbit, ch0)
        _, s0, _ = solver.solve(
            ch0,
            asymptotics[0][0],
            free_matrix=free_matrices[0],
            interaction_matrix=im_central,
            basis_boundary=basis_boundary,
        )
        splus[i, 0] = s0[0, 0]
        for ell in sys.l[1:]:
            lds = l_dot_s[ell - 1]
            for j_index, target in ((0, splus), (1, sminus)):
                _, s, _ = solver.solve(
                    channels[ell][j_index],
                    asymptotics[ell][j_index],
                    free_matrix=free_matrices[ell],
                    interaction_matrix=im_central + lds[j_index] * im_so,
                    basis_boundary=basis_boundary,
                )
                target[i, ell] = s[0, 0]
    return splus, sminus


def _classical_setup():
    """Classical kinematics + a fixed (E-independent) KDUQ-shaped potential."""
    reaction = ElasticReaction(ELASTIC_TARGET, ELASTIC_PROJECTILE)
    kins = [
        classical_kinematics(
            reaction.target.m0,
            reaction.projectile.m0,
            float(elab),
            reaction.projectile.Z * reaction.target.Z,
        )
        for elab in ELASTIC_ELAB
    ]
    omp = kduq.KDUQ(ELASTIC_PROJECTILE)
    params = np.array(list(kduq.Global(ELASTIC_PROJECTILE).params.values()))
    from jitr.rmatrix import Solver

    rgrid = Solver(NBASIS).radial_grid(RADIUS, 1.0)
    central, spin_orbit, coulomb = omp.evaluate(rgrid, reaction, kins[1], *params)
    return kins, central, spin_orbit, np.asarray(coulomb, dtype=np.complex128)


def test_classical_static_v_matches_old_engine() -> None:
    kins, central, spin_orbit, coulomb = _classical_setup()
    s_plus_ref, s_minus_ref = _old_engine_smatrix(kins, central, spin_orbit, coulomb)

    mf = np.array([HBARC**2 / (2.0 * k.mu) for k in kins])
    energies = mf * np.array([k.k for k in kins]) ** 2  # = Ecm classically
    np.testing.assert_allclose(energies, [k.Ecm for k in kins], rtol=1e-12)

    solver = _compile_blocked(energies, mf, (1, ELASTIC_TARGET[1]))
    v_plus, v_minus = _lax_interactions(
        solver, central + coulomb, spin_orbit, energy_dependent=False
    )
    s_plus = _lax_smatrix_static(solver, v_plus).T  # (N_E, lmax+1)
    s_minus = _lax_smatrix_static(solver, v_minus).T

    np.testing.assert_allclose(s_plus, s_plus_ref, atol=1e-10, rtol=0)
    # old engine reports sminus[0] = 0 (placeholder); lax computes it — ignore
    np.testing.assert_allclose(s_minus[:, 1:], s_minus_ref[:, 1:], atol=1e-10, rtol=0)


def test_energy_dependent_regime_matches_static() -> None:
    kins, central, spin_orbit, coulomb = _classical_setup()
    mf = np.array([HBARC**2 / (2.0 * k.mu) for k in kins])
    energies = mf * np.array([k.k for k in kins]) ** 2
    n_e = energies.size

    static_solver = _compile_blocked(energies, mf, (1, ELASTIC_TARGET[1]))
    vp_s, vm_s = _lax_interactions(
        static_solver, central + coulomb, spin_orbit, energy_dependent=False
    )

    grid_solver = _compile_blocked(
        energies, mf, (1, ELASTIC_TARGET[1]), energy_dependent=True
    )
    tiled = np.tile(central + coulomb, (n_e, 1))
    vp_g, vm_g = _lax_interactions(
        grid_solver,
        tiled,
        np.tile(spin_orbit, (n_e, 1)),
        energy_dependent=True,
    )

    for v_static, v_grid in ((vp_s, vp_g), (vm_s, vm_g)):
        np.testing.assert_allclose(
            _lax_smatrix_grid(grid_solver, v_grid),
            _lax_smatrix_static(static_solver, v_static),
            atol=1e-11,
            rtol=0,
        )


def test_semi_relativistic_grid_matches_elastic_golden() -> None:
    """Non-uniform μ(E) (C4 path) + E-dependent KDUQ vs the Phase 1 golden.

    Exercises the faithful mapping incl. the V·(E_lax/Ecm) interior rescale.
    """
    golden = load_golden("elastic_pca")
    mu = golden["mu"]
    k = golden["k"]
    ecm = golden["Ecm"]
    mf = HBARC**2 / (2.0 * mu)  # non-uniform: semi-relativistic μ(E)
    assert not np.allclose(mf, mf[0]), "expected non-uniform mass factors"
    energies = mf * k**2
    scale = energies / ecm  # interior V/Ecm convention of the old engine
    assert not np.allclose(scale, 1.0)

    # lax's eta must reproduce jitr's semi-relativistic eta exactly
    e2 = 1.0 / 137.0359991 * HBARC
    np.testing.assert_allclose(
        ELASTIC_TARGET[1] * e2 / (2.0 * mf * k), golden["eta"], rtol=1e-10
    )

    reaction = ElasticReaction(ELASTIC_TARGET, ELASTIC_PROJECTILE)
    omp = kduq.KDUQ(ELASTIC_PROJECTILE)
    params = np.array(list(kduq.Global(ELASTIC_PROJECTILE).params.values()))
    solver = _compile_blocked(
        energies, mf, (1, ELASTIC_TARGET[1]), energy_dependent=True
    )
    rgrid = np.asarray(solver.mesh.radii)

    central = np.zeros((ecm.size, NBASIS), dtype=np.complex128)
    spin_orbit = np.zeros_like(central)
    for i, elab in enumerate(golden["Elab"]):
        kin = reaction.kinematics(float(elab))
        u_c, u_so, u_coul = omp.evaluate(rgrid, reaction, kin, *params)
        central[i] = (u_c + u_coul) * scale[i]
        spin_orbit[i] = u_so * scale[i]

    v_plus, v_minus = _lax_interactions(
        solver, central, spin_orbit, energy_dependent=True
    )
    s_plus = _lax_smatrix_grid(solver, v_plus).T
    s_minus = _lax_smatrix_grid(solver, v_minus).T

    np.testing.assert_allclose(s_plus, golden["Splus"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        s_minus[:, 1:], golden["Sminus"][:, 1:], atol=1e-10, rtol=0
    )


def test_static_observable_raises_on_nonuniform_mass_factor() -> None:
    """C4 guard: static-regime observables must refuse non-uniform μ(E)."""
    golden = load_golden("elastic_pca")
    mf = HBARC**2 / (2.0 * golden["mu"])
    energies = mf * golden["k"] ** 2
    solver = _compile_blocked(energies, mf, (1, ELASTIC_TARGET[1]))
    ia = solver.interaction_from_array
    v = ia(local=[np.full(NBASIS, -10.0 - 1.0j)])
    with pytest.raises(ValueError):
        _lax_smatrix_static(solver, v)
