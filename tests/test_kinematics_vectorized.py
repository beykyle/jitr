"""Kinematics constructors accept array Elab/Ecm (design doc §3.2 item 1).

Array-in must equal an elementwise scalar loop; scalar-in keeps returning
scalars (the pre-rewrite contract).
"""

import numpy as np
import pytest

from jitr.reactions import ElasticReaction, Reaction
from jitr.utils.kinematics import (
    classical_kinematics,
    classical_kinematics_cm,
    semi_relativistic_kinematics,
)

MASS_TARGET = 44657.26581995028  # 48Ca, MeV/c^2
MASS_PROTON = 938.271653086152  # MeV/c^2
ZZ = 20
ELAB = np.array([5.0, 10.0, 25.0, 35.0, 50.0])

CONSTRUCTORS = {
    "classical": classical_kinematics,
    "classical_cm": classical_kinematics_cm,
    "semi_relativistic": semi_relativistic_kinematics,
}


@pytest.mark.parametrize("name", sorted(CONSTRUCTORS))
def test_array_input_matches_scalar_loop(name: str) -> None:
    constructor = CONSTRUCTORS[name]
    vectorized = constructor(MASS_TARGET, MASS_PROTON, ELAB, ZZ)
    for i, energy in enumerate(ELAB):
        scalar = constructor(MASS_TARGET, MASS_PROTON, float(energy), ZZ)
        for field in ("Elab", "Ecm", "mu", "k", "eta"):
            np.testing.assert_allclose(
                np.broadcast_to(getattr(vectorized, field), ELAB.shape)[i],
                getattr(scalar, field),
                rtol=1e-14,
                err_msg=f"{name}.{field} at Elab={energy}",
            )


@pytest.mark.parametrize("name", sorted(CONSTRUCTORS))
def test_scalar_input_returns_scalars(name: str) -> None:
    kinematics = CONSTRUCTORS[name](MASS_TARGET, MASS_PROTON, 25.0, ZZ)
    for field in ("Elab", "Ecm", "mu", "k", "eta"):
        assert np.ndim(getattr(kinematics, field)) == 0, field


def test_reaction_kinematics_accepts_arrays() -> None:
    reaction = ElasticReaction((48, 20), (1, 1))
    vectorized = reaction.kinematics(ELAB)
    assert np.shape(vectorized.Ecm) == ELAB.shape
    scalar = reaction.kinematics(float(ELAB[2]))
    np.testing.assert_allclose(np.asarray(vectorized.k)[2], scalar.k, rtol=1e-14)

    cm = reaction.kinematics_cm(np.asarray(vectorized.Ecm))
    np.testing.assert_allclose(cm.Ecm, vectorized.Ecm, rtol=1e-12)


def test_kinematics_exit_accepts_arrays() -> None:
    reaction = Reaction(
        target=(48, 20), projectile=(1, 1), product=(1, 0), residual=(48, 21)
    )
    entrance = reaction.kinematics(ELAB)
    exit_kinematics = reaction.kinematics_exit(
        entrance, residual_excitation_energy=6.67
    )
    assert np.shape(exit_kinematics.Ecm) == ELAB.shape
    scalar_exit = reaction.kinematics_exit(
        reaction.kinematics(float(ELAB[3])), residual_excitation_energy=6.67
    )
    np.testing.assert_allclose(
        np.asarray(exit_kinematics.k)[3], scalar_exit.k, rtol=1e-14
    )
