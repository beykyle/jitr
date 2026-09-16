"""The Coulomb-Hankel table and the per-``l`` wrappers agree with :mod:`mpmath`,
satisfy the Wronskian, and feed the partial-wave asymptotics."""

import mpmath
import numpy as np
import pytest

from jitr.reactions.system import ProjectileTargetSystem
from jitr.utils.free_solutions import (
    H_minus,
    H_minus_prime,
    H_plus,
    H_plus_prime,
    coulomb_hankel_table,
)


def _mpmath_reference(rho, eta, l):
    """``(H+, H-, H+', H-')`` from mpmath's Coulomb functions and their derivatives."""
    F = mpmath.coulombf(l, eta, rho)
    G = mpmath.coulombg(l, eta, rho)
    Fp = mpmath.diff(lambda r: mpmath.coulombf(l, eta, r), rho)
    Gp = mpmath.diff(lambda r: mpmath.coulombg(l, eta, r), rho)
    return tuple(
        complex(v) for v in (G + 1j * F, G - 1j * F, Gp + 1j * Fp, Gp - 1j * Fp)
    )


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.85, 1.54, 3.0, 6.0])
@pytest.mark.parametrize("rho", [4.0, 16.1, 37.3, 75.0])
def test_table_matches_mpmath(eta, rho):
    lmax = 60
    table = coulomb_hankel_table(rho, eta, lmax)
    for l in range(0, lmax + 1, 7):
        got = (table.Hp[l], table.Hm[l], table.Hpp[l], table.Hmp[l])
        for g, w in zip(got, _mpmath_reference(rho, eta, l), strict=True):
            assert abs(g - w) <= 1e-9 * abs(w) + 1e-12, (eta, rho, l, g, w)


@pytest.mark.parametrize("eta,rho,l", [(0.0, 9.0, 3), (1.2, 30.0, 11), (4.0, 12.0, 0)])
def test_per_l_wrappers_match_mpmath(eta, rho, l):
    got = (
        H_plus(rho, l, eta),
        H_minus(rho, l, eta),
        H_plus_prime(rho, l, eta),
        H_minus_prime(rho, l, eta),
    )
    for g, w in zip(got, _mpmath_reference(rho, eta, l), strict=True):
        assert isinstance(g, complex)
        assert abs(g - w) <= 1e-9 * abs(w) + 1e-12


@pytest.mark.parametrize("eta,rho", [(0.0, 20.0), (1.2, 30.0), (4.0, 12.0)])
def test_wronskian_is_unity(eta, rho):
    Hp, Hm, Hpp, Hmp = coulomb_hankel_table(rho, eta, 80)
    F, G = (Hp - Hm).imag / 2, (Hp + Hm).real / 2
    Fp, Gp = (Hpp - Hmp).imag / 2, (Hpp + Hmp).real / 2
    np.testing.assert_allclose(Fp * G - F * Gp, 1.0, rtol=1e-8, atol=1e-10)


def test_partial_wave_channels_use_the_table():
    system = ProjectileTargetSystem(channel_radius=25.0, lmax=30, Ztarget=20, Zproj=1)
    _, asym = system.get_partial_wave_channels(
        Elab=20.0, Ecm=19.5, mu=930.0, k=0.95, eta=0.7
    )
    assert len(asym) == 31
    for l in (0, 5, 17, 30):
        Hp, _, _, Hmp = _mpmath_reference(25.0, 0.7, l)
        assert asym[l].Hp.shape == (1,)
        assert np.isclose(asym[l].Hp[0], Hp, rtol=1e-9)
        assert np.isclose(asym[l].Hmp[0], Hmp, rtol=1e-9)
