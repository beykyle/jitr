"""G1: non-local kernels in the elastic workspaces (design doc §1, §3.3).

Anchored against independently compiled single-channel ``channels=`` lax
solvers — a different code path through lax than the blocked workspace.
"""

import numpy as np
import pytest

from jitr.reactions import ElasticReaction
from jitr.utils.kinematics import classical_kinematics

from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 4
NBASIS = 24
RADIUS = 10.0
BETA = 0.85  # Perey-Buck non-locality range, fm


def _perey_buck(ri, rj):
    """Simplified Perey-Buck kernel: Gaussian non-locality on a WS-ish well."""
    center = (ri + rj) / 2.0
    h = (-45.0 - 6.0j) / (1.0 + np.exp((center - 4.2) / 0.65))
    gauss = np.exp(-(((ri - rj) / BETA) ** 2)) / (np.pi**1.5 * BETA**3)
    return h * gauss


@pytest.fixture(scope="module")
def workspace():
    from jitr.xs.elastic import IntegralWorkspace

    reaction = ElasticReaction((48, 20), (1, 0))  # neutron: no Coulomb
    # classical: uniform mass factor, interior rescale exactly 1, so the
    # independent channels= reference below needs no potential rescaling
    kinematics = classical_kinematics(
        reaction.target.m0, reaction.projectile.m0, np.array([10.0, 30.0]), 0
    )
    return IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=RADIUS,
        lmax=LMAX,
        nbasis=NBASIS,
    )


def test_nonlocal_blocked_matches_independent_channels_solvers(workspace):
    """Blocked non-local S(l, E) ≡ per-l independently compiled solvers."""
    import lax

    splus, sminus = workspace.smatrix(_perey_buck, energy_dependent=False)
    grid = workspace.engine.grid

    for ell in range(LMAX + 1):
        solver = lax.compile(
            mesh=lax.MeshSpec("legendre", "x", n=NBASIS, scale=RADIUS),
            channels=[
                lax.ChannelSpec(
                    l=ell, threshold=0.0, mass_factor=float(grid.mass_factors[0])
                )
            ],
            solvers=("spectrum", "smatrix"),
            energies=np.asarray(grid.energies),
            V_is_complex=True,
            method="eig",
        )
        ri, rj = np.meshgrid(
            np.asarray(solver.mesh.radii), np.asarray(solver.mesh.radii), indexing="ij"
        )
        v = solver.interaction_from_array(nonlocal_=[_perey_buck(ri, rj)])
        s_ref = np.asarray(solver.smatrix(solver.spectrum(v)))[:, 0, 0]
        np.testing.assert_allclose(
            np.asarray(splus)[ell], s_ref, rtol=1e-12, atol=1e-13
        )
        if ell >= 1:
            np.testing.assert_allclose(
                np.asarray(sminus)[ell - 1], s_ref, rtol=1e-12, atol=1e-13
            )


def test_nonlocal_absorptive_is_subunitary(workspace):
    splus, _ = workspace.smatrix(_perey_buck, energy_dependent=False)
    s_abs = np.abs(np.asarray(splus))
    assert np.all(s_abs <= 1.0 + 1e-12)
    assert np.any(s_abs < 0.999), "absorption expected from the imaginary part"


def test_mixed_local_plus_nonlocal_terms(workspace):
    """§3.3: local + non-local + spin-orbit terms compose via Interaction add."""
    rgrid = workspace.radial_grid()
    local = (-10.0 - 1.0j) * np.exp(-((rgrid / 3.5) ** 2))
    so = 1.2 * np.exp(-((rgrid / 2.5) ** 2))
    potential = (
        workspace.central(local)
        + workspace.nonlocal_(_perey_buck, energy_dependent=False)
        + workspace.spin_orbit(so)
    )
    splus, sminus = workspace.smatrix(potential)
    assert np.asarray(splus).shape == (LMAX + 1, 2)
    assert np.asarray(sminus).shape == (LMAX, 2)
    assert not np.allclose(np.asarray(splus)[1:], np.asarray(sminus))


def test_l_dependent_nonlocal_kernel(workspace):
    """ℓ-dependent kernels ride the block axis (block_dependent=True)."""
    r = workspace.radial_grid()
    ri, rj = np.meshgrid(r, r, indexing="ij")
    base = _perey_buck(ri, rj)
    stack = np.stack([(1.0 + 0.05 * ell) * base for ell in range(LMAX + 1)])
    splus_stack, _ = workspace.smatrix(stack, energy_dependent=False, l_dependent=True)
    splus_base, _ = workspace.smatrix(base, energy_dependent=False)
    # l = 0 block uses the unscaled kernel in both
    np.testing.assert_allclose(
        np.asarray(splus_stack)[0], np.asarray(splus_base)[0], rtol=1e-12
    )
    assert not np.allclose(np.asarray(splus_stack)[1:], np.asarray(splus_base)[1:])
