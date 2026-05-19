import numpy as np

from jitr.reactions import ElasticReaction, Reaction
from jitr.rmatrix import Solver
from jitr.xs.elastic import DifferentialWorkspace, IntegralWorkspace
from jitr.xs.quasielastic_pn import Workspace as QuasielasticWorkspace


def _gaussian_potential(
    rgrid: np.ndarray, strength: complex, radius: float
) -> np.ndarray:
    return np.asarray(strength * np.exp(-((rgrid / radius) ** 2)), dtype=np.complex128)


def test_elastic_workspaces_treat_missing_spin_orbit_as_zero() -> None:
    reaction = ElasticReaction((48, 20), (1, 0))
    kinematics = reaction.kinematics(12.0)
    solver = Solver(16)
    integral = IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=8.0,
        solver=solver,
        lmax=4,
    )
    angles = np.linspace(0.1, np.pi - 0.1, 7)
    differential = DifferentialWorkspace(integral, angles)

    rgrid = integral.radial_grid()
    central = _gaussian_potential(rgrid, -25.0 - 2.5j, 3.0)
    zero_spin_orbit = np.zeros_like(rgrid, dtype=np.complex128)

    omitted_splus, omitted_sminus = integral.smatrix(central)
    explicit_splus, explicit_sminus = integral.smatrix(central, zero_spin_orbit)
    np.testing.assert_allclose(omitted_splus, explicit_splus)
    np.testing.assert_allclose(omitted_sminus, explicit_sminus)

    omitted_total = integral.xs(central)
    explicit_total = integral.xs(central, zero_spin_orbit)
    np.testing.assert_allclose(omitted_total, explicit_total)

    omitted_transmission = integral.transmission_coefficients(central)
    explicit_transmission = integral.transmission_coefficients(central, zero_spin_orbit)
    np.testing.assert_allclose(omitted_transmission[0], explicit_transmission[0])
    np.testing.assert_allclose(omitted_transmission[1], explicit_transmission[1])

    omitted_differential = differential.xs(central)
    explicit_differential = differential.xs(central, zero_spin_orbit)
    np.testing.assert_allclose(omitted_differential.dsdo, explicit_differential.dsdo)
    np.testing.assert_allclose(omitted_differential.Ay, explicit_differential.Ay)
    np.testing.assert_allclose(omitted_differential.Q, explicit_differential.Q)
    np.testing.assert_allclose(omitted_differential.t, explicit_differential.t)
    np.testing.assert_allclose(omitted_differential.rxn, explicit_differential.rxn)


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
