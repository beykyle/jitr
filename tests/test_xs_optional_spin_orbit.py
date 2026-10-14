import numpy as np

from jitr.reactions import ElasticReaction, Reaction
from jitr.rmatrix import Solver
from jitr.xs.coupled import Potential
from jitr.xs.elastic import Workspace
from jitr.xs.quasielastic_pn import Workspace as QuasielasticWorkspace


def _gaussian_potential(
    rgrid: np.ndarray, strength: complex, radius: float
) -> np.ndarray:
    return np.asarray(strength * np.exp(-((rgrid / radius) ** 2)), dtype=np.complex128)


def test_elastic_workspace_treats_missing_spin_orbit_as_zero() -> None:
    reaction = ElasticReaction((48, 20), (1, 0))
    workspace = Workspace(
        reaction,
        reaction.kinematics(12.0),
        channel_radius_fm=8.0,
        lmax=4,
        nbasis=16,
        angles=np.linspace(0.1, np.pi - 0.1, 7),
    )
    central = _gaussian_potential(workspace.r, -25.0 - 2.5j, 3.0)
    omitted = workspace.xs(Potential(central=central))
    explicit = workspace.xs(Potential(central=central, spin_orbit=0 * central))
    for a, b in zip(omitted, explicit, strict=True):
        np.testing.assert_allclose(a, b)


def test_quasielastic_workspace_treats_missing_spin_orbit_as_zero() -> None:
    reaction = Reaction((48, 20), (1, 1), (1, 0), (48, 21))
    kinematics_entrance = reaction.kinematics(20.0)
    kinematics_exit = reaction.kinematics_exit(kinematics_entrance, 2.0)
    solver = Solver(14)
    workspace = QuasielasticWorkspace(
        reaction=reaction,
        kinematics_entrance=kinematics_entrance,
        kinematics_exit=kinematics_exit,
        solver=solver,
        angles=np.linspace(0.15, np.pi - 0.15, 5),
        lmax=3,
        channel_radius_fm=8.0,
    )

    rgrid = workspace.radial_grid()
    proton_coulomb = _gaussian_potential(rgrid, 2.0, 4.0)
    proton_central = _gaussian_potential(rgrid, -30.0 - 1.0j, 3.2)
    neutron_central = _gaussian_potential(rgrid, -22.0 - 1.5j, 2.8)
    zero_spin_orbit = np.zeros_like(rgrid, dtype=np.complex128)

    omitted_tmatrix = workspace.tmatrix(
        proton_coulomb,
        proton_central,
        U_n_central=neutron_central,
    )
    explicit_tmatrix = workspace.tmatrix(
        proton_coulomb,
        proton_central,
        zero_spin_orbit,
        neutron_central,
        zero_spin_orbit,
    )
    for omitted, explicit in zip(omitted_tmatrix, explicit_tmatrix, strict=True):
        np.testing.assert_allclose(omitted, explicit)

    omitted_xs = workspace.xs(
        proton_coulomb,
        proton_central,
        U_n_central=neutron_central,
    )
    explicit_xs = workspace.xs(
        proton_coulomb,
        proton_central,
        zero_spin_orbit,
        neutron_central,
        zero_spin_orbit,
    )
    np.testing.assert_allclose(omitted_xs, explicit_xs)
