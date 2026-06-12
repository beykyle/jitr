"""G2: energy-vectorized workspaces ≡ per-energy scalar workspaces."""

import numpy as np
import pytest

from jitr.reactions import ElasticReaction

from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 4
NBASIS = 20
RADIUS = 9.0
ELAB = np.array([6.0, 14.0, 28.0, 45.0])


@pytest.fixture(scope="module")
def reaction():
    return ElasticReaction((48, 20), (1, 1))  # proton: charged channel


def _build(reaction, kinematics):
    from jitr.xs.elastic import IntegralWorkspace

    return IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=RADIUS,
        lmax=LMAX,
        nbasis=NBASIS,
    )


def _static_potentials(rgrid):
    central = (-42.0 - 5.0j) * np.exp(-((rgrid / 3.8) ** 2))
    spin_orbit = 1.6 * np.exp(-((rgrid / 2.4) ** 2))
    coulomb = 20.0 * 1.44 / np.maximum(rgrid, 1.0)
    return central, spin_orbit, coulomb.astype(np.complex128)


def test_static_v_vectorized_matches_scalar_loop(reaction):
    vectorized = _build(reaction, reaction.kinematics(ELAB))
    central, spin_orbit, coulomb = _static_potentials(vectorized.radial_grid())
    splus, sminus = vectorized.smatrix(central, spin_orbit, coulomb)
    t, rxn = vectorized.xs(central, spin_orbit, coulomb)

    for i, elab in enumerate(ELAB):
        scalar_ws = _build(reaction, reaction.kinematics(float(elab)))
        sp_i, sm_i = scalar_ws.smatrix(central, spin_orbit, coulomb)
        np.testing.assert_allclose(
            np.asarray(splus)[:, i], np.asarray(sp_i)[:, 0], rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(sminus)[:, i], np.asarray(sm_i)[:, 0], rtol=1e-10
        )
        t_i, rxn_i = scalar_ws.xs(central, spin_orbit, coulomb)
        np.testing.assert_allclose(np.asarray(t)[i], np.asarray(t_i)[0], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(rxn)[i], np.asarray(rxn_i)[0], rtol=1e-10)


def test_energy_dependent_v_matches_per_energy_solves(reaction):
    vectorized = _build(reaction, reaction.kinematics(ELAB))
    rgrid = vectorized.radial_grid()
    _, spin_orbit, coulomb = _static_potentials(rgrid)
    # energy-dependent depth (DOM-style): one slice per grid energy
    central = np.stack(
        [(-42.0 - 0.2 * e - 4.0j) * np.exp(-((rgrid / 3.8) ** 2)) for e in ELAB]
    )

    splus, sminus = vectorized.smatrix(central, spin_orbit, coulomb)
    assert np.asarray(splus).shape == (LMAX + 1, ELAB.size)

    for i, elab in enumerate(ELAB):
        scalar_ws = _build(reaction, reaction.kinematics(float(elab)))
        sp_i, sm_i = scalar_ws.smatrix(central[i], spin_orbit, coulomb)
        np.testing.assert_allclose(
            np.asarray(splus)[:, i], np.asarray(sp_i)[:, 0], rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(sminus)[:, i], np.asarray(sm_i)[:, 0], rtol=1e-10
        )


def test_differential_vectorized_matches_scalar_loop(reaction):
    from jitr.xs.elastic import DifferentialWorkspace

    angles = np.linspace(0.2, np.pi - 0.2, 9)
    vectorized = DifferentialWorkspace(
        _build(reaction, reaction.kinematics(ELAB)), angles
    )
    central, spin_orbit, coulomb = _static_potentials(vectorized.radial_grid())
    xs = vectorized.xs(central, spin_orbit, coulomb)
    assert np.asarray(xs.dsdo).shape == (ELAB.size, angles.size)

    for i, elab in enumerate(ELAB):
        scalar = DifferentialWorkspace(
            _build(reaction, reaction.kinematics(float(elab))), angles
        )
        xs_i = scalar.xs(central, spin_orbit, coulomb)
        np.testing.assert_allclose(
            np.asarray(xs.dsdo)[i], np.asarray(xs_i.dsdo)[0], rtol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(xs.Ay)[i], np.asarray(xs_i.Ay)[0], rtol=1e-8, atol=1e-12
        )


def test_suggest_lmax_covers_highest_energy(reaction):
    from jitr.xs.elastic import suggest_lmax

    lmax = suggest_lmax(reaction, float(ELAB[-1]), RADIUS)
    kinematics = reaction.kinematics(float(ELAB[-1]))
    grazing = float(np.asarray(kinematics.k)) * RADIUS
    assert lmax >= int(np.ceil(grazing))
    assert isinstance(lmax, int)
