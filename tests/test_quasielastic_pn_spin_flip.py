import numpy as np
import pytest

from jitr.reactions import Reaction
from jitr.rmatrix import Solver
from jitr.xs.quasielastic_pn import Workspace


def _woods_saxon(rgrid: np.ndarray, depth: complex, R: float, a: float) -> np.ndarray:
    return np.asarray(depth / (1 + np.exp((rgrid - R) / a)), dtype=np.complex128)


def _thomas(rgrid: np.ndarray, depth: complex, R: float, a: float) -> np.ndarray:
    # derivative Woods-Saxon form factor used for spin-orbit coupling
    x = np.exp((rgrid - R) / a)
    return np.asarray(-depth * x / (1 + x) ** 2 / (a * rgrid), dtype=np.complex128)


@pytest.fixture(scope="module")
def workspace() -> Workspace:
    reaction = Reaction((48, 20), (1, 1), (1, 0), (48, 21))
    kinematics_entrance = reaction.kinematics(35.0, relativistic=False)
    kinematics_exit = reaction.kinematics_exit(
        kinematics_entrance, 6.67, relativistic=False
    )
    return Workspace(
        reaction=reaction,
        kinematics_entrance=kinematics_entrance,
        kinematics_exit=kinematics_exit,
        solver=Solver(35),
        angles=np.linspace(0.05, np.pi - 0.05, 60),
        lmax=15,
        channel_radius_fm=14.0,
        tmatrix_abs_tol=1e-12,
    )


def _potentials(workspace: Workspace) -> dict[str, np.ndarray]:
    rgrid = workspace.radial_grid()
    return {
        "U_p_coulomb": _woods_saxon(rgrid, 8.0, 4.7, 0.3),
        "U_p_central": _woods_saxon(rgrid, -50.0 - 8.0j, 4.4, 0.65),
        "U_p_spin_orbit": _thomas(rgrid, 6.0 - 0.3j, 4.0, 0.6),
        "U_n_central": _woods_saxon(rgrid, -46.0 - 8.0j, 4.4, 0.65),
        "U_n_spin_orbit": _thomas(rgrid, 5.5 - 0.3j, 4.0, 0.6),
    }


def _spin_amplitudes(workspace: Workspace, **potentials: np.ndarray) -> np.ndarray:
    Tlj, _, _ = workspace.tmatrix(**potentials)
    return np.einsum("abljt,lj->abt", workspace.geometric_factor, Tlj)


def test_spin_flip_geometric_factors_nonzero(workspace: Workspace) -> None:
    gf = workspace.geometric_factor
    for l in range(1, workspace.lmax + 1):
        for ijp in range(2):
            assert np.max(np.abs(gf[0, 1, l, ijp])) > 0
            assert np.max(np.abs(gf[1, 0, l, ijp])) > 0
    # s-wave cannot flip spin
    np.testing.assert_array_equal(gf[0, 1, 0], 0)
    np.testing.assert_array_equal(gf[1, 0, 0], 0)


def test_spin_flip_cancels_over_j(workspace: Workspace) -> None:
    # CG orthogonality: for j-independent T_lj the spin-flip amplitude vanishes
    gf = workspace.geometric_factor
    scale = np.max(np.abs(gf))
    np.testing.assert_allclose(gf[0, 1].sum(axis=1), 0, atol=1e-12 * scale)
    np.testing.assert_allclose(gf[1, 0].sum(axis=1), 0, atol=1e-12 * scale)


def test_no_spin_orbit_has_no_spin_flip(workspace: Workspace) -> None:
    potentials = _potentials(workspace)
    potentials.pop("U_p_spin_orbit")
    potentials.pop("U_n_spin_orbit")
    T = _spin_amplitudes(workspace, **potentials)
    np.testing.assert_allclose(T[0, 1], 0, atol=1e-10 * np.max(np.abs(T)))
    np.testing.assert_allclose(T[1, 0], 0, atol=1e-10 * np.max(np.abs(T)))


def test_spin_orbit_produces_spin_flip(workspace: Workspace) -> None:
    potentials = _potentials(workspace)
    T = _spin_amplitudes(workspace, **potentials)
    xs = workspace.xs(**potentials)

    non_flip = workspace.xs_factor * 10 * (np.abs(T[0, 0]) ** 2 + np.abs(T[1, 1]) ** 2)
    flip = workspace.xs_factor * 10 * (np.abs(T[0, 1]) ** 2 + np.abs(T[1, 0]) ** 2)
    np.testing.assert_allclose(xs, non_flip + flip, rtol=1e-12)

    # parity: |T_{++}| = |T_{--}| and |T_{+-}| = |T_{-+}| in the scattering plane
    np.testing.assert_allclose(np.abs(T[0, 0]), np.abs(T[1, 1]), rtol=1e-10)
    np.testing.assert_allclose(np.abs(T[0, 1]), np.abs(T[1, 0]), rtol=1e-10)

    # spin-flip vanishes at 0 and 180 degrees but is sizable at intermediate angles
    mid = (workspace.angles > np.pi / 4) & (workspace.angles < 3 * np.pi / 4)
    assert np.mean(flip[mid] / xs[mid]) > 0.05


def _default_U1(workspace: Workspace, potentials: dict[str, np.ndarray]):
    f = workspace.isovector_factor
    U1_central = -(potentials["U_n_central"] - potentials["U_p_central"]) * f
    U1_spin_orbit = -(potentials["U_n_spin_orbit"] - potentials["U_p_spin_orbit"]) * f
    return U1_central, U1_spin_orbit


def test_explicit_U1_matches_default(workspace: Workspace) -> None:
    potentials = _potentials(workspace)
    U1_central, U1_spin_orbit = _default_U1(workspace, potentials)

    default = workspace.tmatrix(**potentials)
    explicit = workspace.tmatrix(
        **potentials, U1_central=U1_central, U1_spin_orbit=U1_spin_orbit
    )
    for d, e in zip(default, explicit, strict=True):
        np.testing.assert_allclose(e, d, rtol=1e-12)

    np.testing.assert_allclose(
        workspace.xs(**potentials, U1_central=U1_central, U1_spin_orbit=U1_spin_orbit),
        workspace.xs(**potentials),
        rtol=1e-12,
    )


def test_U1_defaults_are_independent(workspace: Workspace) -> None:
    potentials = _potentials(workspace)
    U1_central, _ = _default_U1(workspace, potentials)

    default = workspace.xs(**potentials)
    only_central = workspace.xs(**potentials, U1_central=U1_central)
    np.testing.assert_allclose(only_central, default, rtol=1e-12)

    no_so_transition = workspace.xs(
        **potentials,
        U1_central=U1_central,
        U1_spin_orbit=np.zeros_like(U1_central),
    )
    assert not np.allclose(no_so_transition, default, rtol=1e-3)


def test_U1_sets_transition_strength(workspace: Workspace) -> None:
    potentials = _potentials(workspace)
    U1_central, _ = _default_U1(workspace, potentials)
    zero = np.zeros_like(U1_central)

    Tlj, Sn, Sp = workspace.tmatrix(
        **potentials, U1_central=U1_central, U1_spin_orbit=zero
    )
    Tlj2, Sn2, Sp2 = workspace.tmatrix(
        **potentials, U1_central=2 * U1_central, U1_spin_orbit=zero
    )
    np.testing.assert_allclose(Tlj2, 2 * Tlj, rtol=1e-12)
    # distorted waves do not depend on the transition potential
    np.testing.assert_allclose(Sn2, Sn)
    np.testing.assert_allclose(Sp2, Sp)

    xs = workspace.xs(**potentials, U1_central=U1_central, U1_spin_orbit=zero)
    xs2 = workspace.xs(**potentials, U1_central=2 * U1_central, U1_spin_orbit=zero)
    np.testing.assert_allclose(xs2, 4 * xs, rtol=1e-12)


def test_U1_shape_is_validated(workspace: Workspace) -> None:
    potentials = _potentials(workspace)
    with pytest.raises(ValueError, match="U1_central"):
        workspace.tmatrix(**potentials, U1_central=np.zeros(3))
