from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jitr.optical_potentials import dom
from jitr.utils.kinematics import ChannelKinematics


@dataclass
class _ParticleStub:
    A: int
    Z: int


@dataclass
class _ReactionStub:
    projectile: _ParticleStub
    target: _ParticleStub
    Ef: float = -6.0


def _sample_params() -> dict[str, float]:
    return {
        "v0": 52.0,
        "v1": 0.006,
        "rv": 1.25,
        "av": 0.65,
        "wv0": 12.0,
        "wv1": 24.0,
        "rw": 1.28,
        "aw": 0.62,
        "ws0": 15.0,
        "ws1": 12.5,
        "ws2": 0.021,
        "rd": 1.28,
        "ad": 0.55,
        "vso0": 6.0,
        "wso0": 0.35,
        "wso1": 25.0,
        "rso": 1.05,
        "aso": 0.59,
        "rC": 1.25,
    }


def _ordered_params() -> list[float]:
    params = _sample_params()
    return [params[name] for name in dom.get_param_names()]


def _reaction(projectile: tuple[int, int] = (1, 1)) -> _ReactionStub:
    return _ReactionStub(
        projectile=_ParticleStub(*projectile),
        target=_ParticleStub(24, 12),
    )


def _kinematics(Ecm: float = 18.0) -> ChannelKinematics:
    return ChannelKinematics(Elab=18.75, Ecm=Ecm, mu=1.0, k=1.0, eta=0.0)


def test_get_param_names_matches_model_params() -> None:
    model = dom.DOM()
    assert model.params == dom.get_param_names()
    assert "Ef" not in model.params


def test_calculate_params_returns_expected_shapes() -> None:
    reaction = _reaction()
    central_params, spin_orbit_params, coulomb_params = dom.calculate_params(
        (reaction.projectile.A, reaction.projectile.Z),
        (reaction.target.A, reaction.target.Z),
        18.0,
        reaction.Ef,
        *_ordered_params(),
    )

    assert len(central_params) == 11
    assert len(coulomb_params) == 2
    assert len(spin_orbit_params) == 4
    assert coulomb_params[0] == 12


def test_neutron_calculate_params_disables_coulomb_correction() -> None:
    proton_reaction = _reaction((1, 1))
    neutron_reaction = _reaction((1, 0))
    proton_params = dom.calculate_params(
        (proton_reaction.projectile.A, proton_reaction.projectile.Z),
        (proton_reaction.target.A, proton_reaction.target.Z),
        18.0,
        proton_reaction.Ef,
        *_ordered_params(),
    )
    neutron_params = dom.calculate_params(
        (neutron_reaction.projectile.A, neutron_reaction.projectile.Z),
        (neutron_reaction.target.A, neutron_reaction.target.Z),
        18.0,
        neutron_reaction.Ef,
        *_ordered_params(),
    )

    proton_central = proton_params[0]
    neutron_central = neutron_params[0]
    proton_coulomb = proton_params[2]
    neutron_coulomb = neutron_params[2]

    assert proton_coulomb[0] == 12
    assert neutron_coulomb[0] == 0
    assert proton_central != neutron_central


def test_dom_wrapper_matches_direct_helper_evaluation() -> None:
    r_grid = np.linspace(0.2, 12.0, 80)
    reaction = _reaction()
    kinematics = _kinematics()
    params = _ordered_params()

    model = dom.DOM()
    central_term, spin_orbit_term, coulomb_term = model.evaluate(
        r_grid,
        reaction,
        kinematics,
        *params,
    )

    central_params, spin_orbit_params, coulomb_params = dom.calculate_params(
        (reaction.projectile.A, reaction.projectile.Z),
        (reaction.target.A, reaction.target.Z),
        kinematics.Ecm,
        reaction.Ef,
        *params,
    )

    np.testing.assert_allclose(central_term, dom.central(r_grid, *central_params))
    np.testing.assert_allclose(
        spin_orbit_term,
        dom.spin_orbit(r_grid, *spin_orbit_params),
    )
    np.testing.assert_allclose(
        coulomb_term,
        dom.coulomb_charged_sphere(r_grid, *coulomb_params),
    )
    np.testing.assert_allclose(
        dom.central_plus_coulomb(r_grid, central_params, coulomb_params),
        central_term + coulomb_term,
    )


def test_dom_wrapper_validates_parameter_count() -> None:
    model = dom.DOM()

    with pytest.raises(ValueError, match="DOM expects"):
        model.evaluate(np.linspace(0.2, 12.0, 10), _reaction(), _kinematics(), 1.0)


def test_extract_params_uses_explicit_reaction_and_kinematics() -> None:
    reaction = _reaction()
    kinematics = _kinematics()
    expected = dom.calculate_params(
        (reaction.projectile.A, reaction.projectile.Z),
        (reaction.target.A, reaction.target.Z),
        kinematics.Ecm,
        reaction.Ef,
        *_ordered_params(),
    )

    assert dom.extract_params(reaction, kinematics, *_ordered_params()) == expected


def test_dom_wrapper_uses_reaction_fermi_energy() -> None:
    r_grid = np.linspace(0.2, 12.0, 80)
    low_ef_reaction = _reaction()
    high_ef_reaction = _reaction()
    high_ef_reaction.Ef = low_ef_reaction.Ef + 4.0
    kinematics = _kinematics()

    model = dom.DOM()
    low_ef_terms = model.evaluate(
        r_grid, low_ef_reaction, kinematics, *_ordered_params()
    )
    high_ef_terms = model.evaluate(
        r_grid, high_ef_reaction, kinematics, *_ordered_params()
    )

    assert not np.allclose(low_ef_terms[0], high_ef_terms[0])
    assert not np.allclose(low_ef_terms[1], high_ef_terms[1])
