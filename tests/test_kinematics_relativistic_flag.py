"""The ``relativistic`` flag on Reaction kinematics selects between the
semi-relativistic (Ingemarsson, default) and non-relativistic prescriptions."""

import numpy as np

from jitr.reactions.reaction import Reaction
from jitr.utils.kinematics import (
    classical_kinematics,
    classical_kinematics_cm,
    semi_relativistic_kinematics,
)

E_LAB = 35.0
E_IAS = 6.67


def _assert_kinematics_equal(a, b):
    np.testing.assert_allclose(list(a), list(b), rtol=0, atol=0)


def test_entrance_lab_flag():
    reaction = Reaction(target=(48, 20), projectile=(1, 1), process="El")
    Zz = reaction.target.Z * reaction.projectile.Z
    m_t, m_p = reaction.target.m0, reaction.projectile.m0
    _assert_kinematics_equal(
        reaction.kinematics(E_LAB),
        semi_relativistic_kinematics(m_t, m_p, E_LAB, Zz=Zz),
    )
    _assert_kinematics_equal(
        reaction.kinematics(E_LAB, relativistic=False),
        classical_kinematics(m_t, m_p, E_LAB, Zz=Zz),
    )


def test_entrance_cm_flag():
    reaction = Reaction(target=(48, 20), projectile=(1, 1), process="El")
    Zz = reaction.target.Z * reaction.projectile.Z
    m_t, m_p = reaction.target.m0, reaction.projectile.m0
    Ecm = 30.0
    classical = reaction.kinematics_cm(Ecm, relativistic=False)
    _assert_kinematics_equal(classical, classical_kinematics_cm(m_t, m_p, Ecm, Zz=Zz))
    assert np.isclose(classical.Ecm, Ecm)
    assert np.isclose(reaction.kinematics_cm(Ecm).Ecm, Ecm)


def test_exit_channel_flag():
    reaction = Reaction(
        target=(48, 20), projectile=(1, 1), product=(1, 0), residual=(48, 21)
    )
    entrance = reaction.kinematics(E_LAB, relativistic=False)
    exit_ch = reaction.kinematics_exit(
        entrance, residual_excitation_energy=E_IAS, relativistic=False
    )
    m_res = reaction.residual.m0 + E_IAS
    m_n = reaction.product.m0
    assert np.isclose(exit_ch.Ecm, entrance.Ecm + reaction.Q - E_IAS)
    assert np.isclose(exit_ch.mu, m_res * m_n / (m_res + m_n))
    assert exit_ch.eta == 0.0

    # default is semi-relativistic and differs from the classical result
    exit_rel = reaction.kinematics_exit(entrance, residual_excitation_energy=E_IAS)
    assert np.isclose(exit_rel.Ecm, exit_ch.Ecm)
    assert not np.isclose(exit_rel.mu, exit_ch.mu)
