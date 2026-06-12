"""DistortedWaves validation against the legacy-engine wavefunction golden.

The golden (``characterization/data/grids.npz``, ``wf_*`` fields) stores the
legacy engine's internal l = 1 wave in s = k·r units for p+48Ca at 35 MeV
(classical kinematics, complex Woods-Saxon + Coulomb, R = 10π/k fm). The two
engines drive the interior solution with different boundary sources, so the
waves agree up to one *constant* per (l, j, E) — the physical normalization
applied by ``DistortedWaves`` matches the legacy exterior convention
``(i/2)[H⁻ − S·H⁺]`` at the boundary, making the constant the s↔r
coefficient scale only.
"""

import numpy as np

from jitr.optical_potentials import potential_forms as potentials
from jitr.reactions import DistortedWaves, ElasticReaction
from jitr.utils.kinematics import classical_kinematics
from jitr.xs.elastic import IntegralWorkspace

from .characterization._cases import load_golden
from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 2
NBASIS = 60


def _golden_workspace_and_waves():
    golden = load_golden("grids")
    reaction = ElasticReaction((48, 20), (1, 1))
    kinematics = classical_kinematics(
        reaction.target.m0, reaction.projectile.m0, float(golden["wf_Elab"]), 20
    )
    k = float(np.asarray(kinematics.k))
    radius_fm = 5 * 2 * np.pi / k  # legacy dimensionless a = 10π
    workspace = IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=radius_fm,
        lmax=LMAX,
        nbasis=NBASIS,
        wavefunctions=True,
    )
    rgrid = workspace.radial_grid()
    local = -potentials.woods_saxon_potential(
        rgrid, 70.0, 40.0, 6.0, 1.2
    ) + potentials.coulomb_charged_sphere(rgrid, 20, 6.0)
    waves = DistortedWaves(workspace, local)
    return golden, workspace, waves, k, radius_fm


def test_interior_matches_legacy_golden_up_to_constant():
    golden, _, waves, k, _ = _golden_workspace_and_waves()
    ell = int(golden["wf_l"])
    s_values = golden["wf_s_values"]
    u_golden = golden["wf_u"]

    u_new = waves.interior(s_values / k)[ell, 0]
    mask = np.abs(u_golden) > 0.2 * np.abs(u_golden).max()
    ratio = u_golden[mask] / u_new[mask]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)


def test_smatrix_matches_legacy_golden():
    golden, _, waves, _, _ = _golden_workspace_and_waves()
    ell = int(golden["wf_l"])
    np.testing.assert_allclose(
        waves.splus[ell, 0], complex(golden["wf_S"][0, 0]), rtol=1e-9
    )


def test_interior_exterior_continuity_at_boundary():
    _, _, waves, _, radius_fm = _golden_workspace_and_waves()
    r_boundary = np.array([radius_fm * (1.0 - 1e-9)])
    interior = waves.interior(r_boundary)
    exterior = waves.exterior(np.array([radius_fm]))
    np.testing.assert_allclose(interior, exterior, rtol=1e-5)


def test_spin_orbit_pair_shapes():
    golden, workspace, _, _, _ = _golden_workspace_and_waves()
    rgrid = workspace.radial_grid()
    local = -potentials.woods_saxon_potential(rgrid, 70.0, 40.0, 6.0, 1.2)
    so = 1.5 * np.exp(-((rgrid / 2.5) ** 2))
    waves = DistortedWaves(workspace, local, so)
    r_eval = np.linspace(0.5, 5.0, 7)
    assert waves.interior(r_eval, "plus").shape == (LMAX + 1, 1, 7)
    assert waves.interior(r_eval, "minus").shape == (LMAX, 1, 7)
    assert not np.allclose(
        waves.interior(r_eval, "plus")[1:], waves.interior(r_eval, "minus")
    )
