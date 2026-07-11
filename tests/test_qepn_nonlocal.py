"""G1 in the (p,n) path: non-local distorting potentials and U₁ kernels."""

import numpy as np

from jitr.reactions import Reaction

from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 3
NBASIS = 16
RADIUS = 9.0
BETA = 0.9


def _kernel(depth):
    def k(ri, rj):
        center = (ri + rj) / 2.0
        h = depth / (1.0 + np.exp((center - 4.0) / 0.6))
        gauss = np.exp(-(((ri - rj) / BETA) ** 2)) / (np.pi**1.5 * BETA**3)
        return h * gauss

    return k


def _workspace():
    from jitr.xs.quasielastic_pn import Workspace

    reaction = Reaction((48, 20), (1, 1), (1, 0), (48, 21))
    kinematics_entrance = reaction.kinematics(25.0)
    kinematics_exit = reaction.kinematics_exit(kinematics_entrance, 2.0)
    return Workspace(
        reaction=reaction,
        kinematics_entrance=kinematics_entrance,
        kinematics_exit=kinematics_exit,
        angles=np.linspace(0.1, np.pi - 0.1, 7),
        lmax=LMAX,
        channel_radius_fm=RADIUS,
        nbasis=NBASIS,
    )


def test_nonlocal_distorting_potentials_run_and_differ_from_local():
    workspace = _workspace()
    rgrid = workspace.radial_grid()
    coulomb = (20.0 * 1.44 / np.maximum(rgrid, 1.2)).astype(np.complex128)
    local_p = (-38.0 - 4.0j) * np.exp(-((rgrid / 3.6) ** 2))
    local_n = (-42.0 - 5.0j) * np.exp(-((rgrid / 3.6) ** 2))

    xs_local = workspace.xs(coulomb, local_p, U_n_central=local_n)

    xs_nonlocal = workspace.xs(
        coulomb,
        _kernel(-38.0 - 4.0j),
        U_n_central=_kernel(-42.0 - 5.0j),
        energy_dependent=False,
    )
    assert np.asarray(xs_nonlocal).shape == np.asarray(xs_local).shape
    assert np.all(np.isfinite(np.asarray(xs_nonlocal)))
    assert np.all(np.asarray(xs_nonlocal) >= 0.0)
    assert not np.allclose(xs_nonlocal, xs_local)


def test_l_dependent_kernel_with_identical_blocks_matches_l_independent():
    """An (lmax+1, N, N) stack with equal blocks must reproduce the
    l-independent nonlocal result exactly (distortion AND U1 paths)."""
    workspace = _workspace()
    rgrid = workspace.radial_grid()
    coulomb = (20.0 * 1.44 / np.maximum(rgrid, 1.2)).astype(np.complex128)
    ri = rgrid[:, None]
    rj = rgrid[None, :]
    K_p = _kernel(-38.0 - 4.0j)(ri, rj).astype(np.complex128)
    K_n = _kernel(-42.0 - 5.0j)(ri, rj).astype(np.complex128)

    xs_flat = workspace.xs(coulomb, K_p, U_n_central=K_n, energy_dependent=False)

    g_p = np.repeat(K_p[None], LMAX + 1, axis=0)
    g_n = np.repeat(K_n[None], LMAX + 1, axis=0)
    xs_stack = workspace.xs(
        coulomb, g_p, U_n_central=g_n, energy_dependent=False, l_dependent=True
    )
    np.testing.assert_allclose(np.asarray(xs_stack), np.asarray(xs_flat), rtol=1e-10)


def test_l_dependent_kernel_distinct_blocks_runs_and_differs():
    """Genuinely l-dependent kernels are accepted (inferred and explicit) and
    change the observable relative to the l=0-broadcast (legacy monopole)."""
    workspace = _workspace()
    rgrid = workspace.radial_grid()
    coulomb = (20.0 * 1.44 / np.maximum(rgrid, 1.2)).astype(np.complex128)
    ri = rgrid[:, None]
    rj = rgrid[None, :]
    K_p = _kernel(-38.0 - 4.0j)(ri, rj).astype(np.complex128)
    K_n = _kernel(-42.0 - 5.0j)(ri, rj).astype(np.complex128)
    ls = np.arange(LMAX + 1)
    g_p = K_p[None] * (1.0 + 0.1 * ls)[:, None, None]
    g_n = K_n[None] * (1.0 + 0.1 * ls)[:, None, None]

    xs_full = workspace.xs(
        coulomb, g_p, U_n_central=g_n, energy_dependent=False, l_dependent=True
    )
    # single-energy (lmax+1, N, N) is unambiguous: inference must agree
    xs_inferred = workspace.xs(coulomb, g_p, U_n_central=g_n, energy_dependent=False)
    np.testing.assert_allclose(np.asarray(xs_inferred), np.asarray(xs_full))

    xs_monopole = workspace.xs(
        coulomb, g_p[0], U_n_central=g_n[0], energy_dependent=False
    )
    assert np.all(np.isfinite(np.asarray(xs_full)))
    assert np.all(np.asarray(xs_full) >= 0.0)
    assert not np.allclose(xs_full, xs_monopole)


def test_tmatrix_shapes_and_l0_minus_channel_zeroed():
    workspace = _workspace()
    rgrid = workspace.radial_grid()
    coulomb = (20.0 * 1.44 / np.maximum(rgrid, 1.2)).astype(np.complex128)
    local_p = (-38.0 - 4.0j) * np.exp(-((rgrid / 3.6) ** 2))
    local_n = (-42.0 - 5.0j) * np.exp(-((rgrid / 3.6) ** 2))
    so = 1.4 * np.exp(-((rgrid / 2.4) ** 2))

    Tpn, Sn, Sp = workspace.tmatrix(coulomb, local_p, so, local_n, so)
    assert Tpn.shape == (LMAX + 1, 2, 1)
    np.testing.assert_array_equal(Tpn[0, 1], 0.0)
    np.testing.assert_array_equal(Sn[0, 1], 0.0)
    assert not np.allclose(Tpn[:, 0], Tpn[:, 1])
