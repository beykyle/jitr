"""Potential-contract dispatch tests for the lax engine (§3.3 + §8 Q1)."""

import numpy as np
import pytest

from jitr.utils.kinematics import classical_kinematics

from .conftest import requires_lax

pytestmark = requires_lax

LMAX = 3
NBASIS = 12
RADIUS = 8.0
MASS_TARGET = 44657.3
MASS_NEUTRON = 939.565


def _engine(n_energies: int, nbasis: int = NBASIS):
    from jitr.xs._lax_engine import BlockedEngine

    kinematics = classical_kinematics(
        MASS_TARGET, MASS_NEUTRON, np.linspace(5.0, 30.0, n_energies), 0
    )
    return BlockedEngine(kinematics, RADIUS, LMAX, nbasis, (0, 20), V_is_complex=True)


@pytest.fixture(scope="module")
def engine():
    return _engine(2)


@pytest.fixture(scope="module")
def square_engine():
    """Engine with N_E == nbasis to provoke shape ambiguity."""
    return _engine(NBASIS)


def _gaussian(r, depth=-40.0 - 4.0j, width=3.0):
    return depth * np.exp(-((r / width) ** 2))


def _kernel(ri, rj, depth=-30.0 - 2.0j, width=2.5, beta=1.0):
    return (
        depth
        * np.exp(-(((ri - rj) / beta) ** 2))
        * np.exp(-(((ri + rj) / (2 * width)) ** 2))
    )


def _flags(interaction):
    return interaction.block_dependent, interaction.energy_dependent


def _block_shape(interaction):
    return tuple(interaction.block.shape)


def test_local_static_array(engine):
    v = engine.interaction(_gaussian(engine.radial_grid()))
    assert _flags(v) == (False, False)
    assert _block_shape(v) == (NBASIS, NBASIS)


def test_local_energy_dependent_array(engine):
    values = np.stack([_gaussian(engine.radial_grid(), d) for d in (-40, -45)])
    v = engine.interaction(values)
    assert _flags(v) == (False, True)
    assert _block_shape(v) == (2, NBASIS, NBASIS)


def test_nonlocal_static_array(engine):
    ri, rj = np.meshgrid(engine.radial_grid(), engine.radial_grid(), indexing="ij")
    v = engine.interaction(_kernel(ri, rj), energy_dependent=False)
    assert _flags(v) == (False, False)
    assert _block_shape(v) == (NBASIS, NBASIS)


def test_nonlocal_energy_dependent_array(engine):
    ri, rj = np.meshgrid(engine.radial_grid(), engine.radial_grid(), indexing="ij")
    values = np.stack([_kernel(ri, rj, d) for d in (-30, -32)])
    v = engine.interaction(values)
    assert _flags(v) == (False, True)
    assert _block_shape(v) == (2, NBASIS, NBASIS)


def test_l_dependent_local_array(engine):
    values = np.stack(
        [_gaussian(engine.radial_grid(), -40 - l) for l in range(LMAX + 1)]
    )
    v = engine.interaction(values, l_dependent=True)
    assert _flags(v) == (True, False)
    assert _block_shape(v) == (LMAX + 1, NBASIS, NBASIS)


def test_l_and_energy_dependent_local_array(engine):
    values = np.zeros((LMAX + 1, 2, NBASIS), dtype=np.complex128)
    values[:] = _gaussian(engine.radial_grid())
    v = engine.interaction(values)
    assert _flags(v) == (True, True)
    assert _block_shape(v) == (LMAX + 1, 2, NBASIS, NBASIS)


def test_l_dependent_nonlocal_array(engine):
    ri, rj = np.meshgrid(engine.radial_grid(), engine.radial_grid(), indexing="ij")
    values = np.stack([_kernel(ri, rj, -30 - l) for l in range(LMAX + 1)])
    v = engine.interaction(values, energy_dependent=False)
    assert _flags(v) == (True, False)
    assert _block_shape(v) == (LMAX + 1, NBASIS, NBASIS)


def test_unrecognized_shape_raises(engine):
    with pytest.raises(ValueError, match="matches no potential layout"):
        engine.interaction(np.zeros((NBASIS + 1,)))


def test_square_shape_ambiguity_requires_flags(square_engine):
    values = np.zeros((NBASIS, NBASIS), dtype=np.complex128)
    with pytest.raises(ValueError, match="ambiguous"):
        square_engine.interaction(values)
    v_edep = square_engine.interaction(values, energy_dependent=True)
    assert _flags(v_edep) == (False, True)
    v_nonlocal = square_engine.interaction(values, energy_dependent=False)
    assert _flags(v_nonlocal) == (False, False)
    assert _block_shape(v_nonlocal) == (NBASIS, NBASIS)


def test_callable_matches_array(engine):
    v_fn = engine.interaction(_gaussian)
    v_arr = engine.interaction(_gaussian(engine.radial_grid()))
    s_fn = engine.smatrix(v_fn)
    s_arr = engine.smatrix(v_arr)
    np.testing.assert_allclose(s_fn[0], s_arr[0], rtol=1e-12)


def test_nonlocal_callable_matches_array(engine):
    v_fn = engine.interaction(_kernel, energy_dependent=False)
    ri, rj = np.meshgrid(engine.radial_grid(), engine.radial_grid(), indexing="ij")
    v_arr = engine.interaction(_kernel(ri, rj), energy_dependent=False)
    np.testing.assert_allclose(
        engine.smatrix(v_fn)[0], engine.smatrix(v_arr)[0], rtol=1e-12
    )


def test_two_argument_callable_requires_flag(engine):
    def f(r, second):
        return _gaussian(r)

    with pytest.raises(ValueError, match="two-argument callable is ambiguous"):
        engine.interaction(f)
    v = engine.interaction(f, energy_dependent=True)
    assert _flags(v) == (False, True)


def test_energy_dependent_callable_evaluated_on_ecm(engine):
    seen = []

    def f(r, e):
        seen.append(float(e))
        return _gaussian(r)

    engine.interaction(f, energy_dependent=True)
    np.testing.assert_allclose(seen, engine.grid.Ecm)


def test_l_dependent_callable_sequence(engine):
    fns = [lambda r, d=-40.0 - l: _gaussian(r, d) for l in range(LMAX + 1)]
    v = engine.interaction(fns)
    assert _flags(v) == (True, False)
    wrong_length = fns[:-1]
    with pytest.raises(ValueError, match="one callable per partial wave"):
        engine.interaction(wrong_length)


def test_interaction_passthrough(engine):
    v = engine.interaction(_gaussian(engine.radial_grid()))
    assert engine.interaction(v) is v


def test_spin_orbit_pair_scaling(engine):
    """The pair must equal interactions built with explicit {l, -(l+1)}."""
    form = 1.5 * np.exp(-((engine.radial_grid() / 2.0) ** 2))
    pair = engine.spin_orbit_pair(form)
    ls = np.arange(LMAX + 1, dtype=np.float64)
    expected_plus = engine.interaction(ls[:, None] * form, l_dependent=True)
    expected_minus = engine.interaction(-(ls + 1.0)[:, None] * form, l_dependent=True)
    np.testing.assert_allclose(
        np.asarray(pair.plus.block), np.asarray(expected_plus.block)
    )
    np.testing.assert_allclose(
        np.asarray(pair.minus.block), np.asarray(expected_minus.block)
    )


def test_interaction_pair_addition_and_smatrix(engine):
    central = engine.interaction(_gaussian(engine.radial_grid()))
    pair = engine.spin_orbit_pair(1.5 * np.exp(-((engine.radial_grid() / 2.0) ** 2)))
    total = central + pair
    splus, sminus = engine.smatrix(total)
    assert splus.shape == (LMAX + 1, 2)
    assert sminus.shape == (LMAX + 1, 2)
    assert not np.allclose(splus[1:], sminus[1:])

    splus_plain, sminus_plain = engine.smatrix(central)
    np.testing.assert_array_equal(splus_plain, sminus_plain)


def test_spin_orbit_pair_l_dependent_kernel(engine):
    """An intrinsically l-dependent SO form factor (e.g. a Perey-Buck
    nonlocal kernel) rides the block axis and still gets the per-l
    ⟨l·σ⟩ scaling — a tiled l-independent stack must reproduce the
    plain (N, N) path exactly."""
    ri, rj = np.meshgrid(engine.radial_grid(), engine.radial_grid(), indexing="ij")
    kernel = _kernel(ri, rj)
    tiled = np.broadcast_to(kernel, (LMAX + 1,) + kernel.shape)

    pair_plain = engine.spin_orbit_pair(kernel, energy_dependent=False)
    pair_tiled = engine.spin_orbit_pair(
        np.ascontiguousarray(tiled), energy_dependent=False, l_dependent=True
    )
    for member in ("plus", "minus"):
        np.testing.assert_allclose(
            np.asarray(getattr(pair_tiled, member).block),
            np.asarray(getattr(pair_plain, member).block),
            rtol=1e-13,
        )

    # genuinely l-dependent input scales per block
    stack = np.stack([_kernel(ri, rj, -30.0 - 5.0 * l) for l in range(LMAX + 1)])
    pair = engine.spin_orbit_pair(stack, energy_dependent=False, l_dependent=True)
    ls = np.arange(LMAX + 1, dtype=np.float64)
    expected_plus = engine.interaction(
        ls[:, None, None] * stack, energy_dependent=False, l_dependent=True
    )
    np.testing.assert_allclose(
        np.asarray(pair.plus.block), np.asarray(expected_plus.block), rtol=1e-13
    )
