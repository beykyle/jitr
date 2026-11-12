from __future__ import annotations

import numpy as np
import pytest

from jitr.optical_potentials import (
    chuq,
    coulomb_charged_sphere,
    kduq,
    perey_buck_nonlocal,
    wlh,
)


@pytest.mark.parametrize(
    ("potential", "args"),
    [
        pytest.param(
            kduq.central,
            (50.0, 4.5, 0.65, 12.0, 4.4, 0.62, 8.0, 4.3, 0.60),
            id="kduq-central",
        ),
        pytest.param(
            kduq.spin_orbit, (6.0, 4.1, 0.58, 1.5, 4.2, 0.57), id="kduq-spin-orbit"
        ),
        pytest.param(
            chuq.central, (52.0, 9.0, 7.5, 4.6, 0.67, 4.5, 0.61), id="chuq-central"
        ),
        pytest.param(chuq.spin_orbit, (5.5, 4.1, 0.59), id="chuq-spin-orbit"),
        pytest.param(
            wlh.central,
            (48.0, 4.4, 0.64, 10.0, 4.5, 0.63, 7.0, 4.3, 0.60),
            id="wlh-central",
        ),
        pytest.param(wlh.spin_orbit, (5.0, 4.0, 0.58), id="wlh-spin-orbit"),
    ],
)
def test_global_potential_helpers_return_complex_scalar(
    potential: object,
    args: tuple[float, ...],
) -> None:
    result = potential(3.5, *args)

    assert isinstance(result, complex)


@pytest.mark.parametrize(
    ("potential", "args"),
    [
        pytest.param(
            kduq.central,
            (50.0, 4.5, 0.65, 12.0, 4.4, 0.62, 8.0, 4.3, 0.60),
            id="kduq-central",
        ),
        pytest.param(
            kduq.spin_orbit, (6.0, 4.1, 0.58, 1.5, 4.2, 0.57), id="kduq-spin-orbit"
        ),
        pytest.param(
            chuq.central, (52.0, 9.0, 7.5, 4.6, 0.67, 4.5, 0.61), id="chuq-central"
        ),
        pytest.param(chuq.spin_orbit, (5.5, 4.1, 0.59), id="chuq-spin-orbit"),
        pytest.param(
            wlh.central,
            (48.0, 4.4, 0.64, 10.0, 4.5, 0.63, 7.0, 4.3, 0.60),
            id="wlh-central",
        ),
        pytest.param(wlh.spin_orbit, (5.0, 4.0, 0.58), id="wlh-spin-orbit"),
    ],
)
def test_global_potential_helpers_return_complex_arrays(
    potential: object,
    args: tuple[float, ...],
) -> None:
    result = potential(np.linspace(0.2, 8.0, 16), *args)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.complex128
    assert result.shape == (16,)


def test_coulomb_charged_sphere_returns_zero_for_neutral_projectiles() -> None:
    scalar = coulomb_charged_sphere(3.0, 0.0, 0.0)
    array = coulomb_charged_sphere(np.linspace(0.2, 8.0, 16), 0.0, 0.0)

    assert scalar == 0.0
    np.testing.assert_array_equal(array, np.zeros(16))


def test_get_kd03_matches_default_global_parameter_ordering() -> None:
    kd03_neutron = kduq.get_kd03((1, 0))
    kd03_proton = kduq.get_kd03((1, 1))

    assert kd03_neutron.dtype == np.float64
    assert kd03_proton.dtype == np.float64
    np.testing.assert_allclose(
        kd03_neutron, np.asarray(list(kduq.Global((1, 0)).params.values()))
    )
    np.testing.assert_allclose(
        kd03_proton, np.asarray(list(kduq.Global((1, 1)).params.values()))
    )


def test_get_ch89_matches_default_global_parameter_ordering() -> None:
    ch89 = chuq.get_ch89()

    assert ch89.dtype == np.float64
    np.testing.assert_allclose(ch89, np.asarray(list(chuq.Global().params.values())))


def _perey_buck_reference(r, rp, beta, ell):
    """Direct angular projection: 2 pi r r' int dmu H(|r - r'|) P_l(mu)."""
    from scipy import integrate, special

    def f(mu):
        s2 = r * r + rp * rp - 2 * r * rp * mu
        return np.exp(-s2 / beta**2) * special.eval_legendre(ell, mu)

    val, _ = integrate.quad(f, -1, 1, epsabs=0, epsrel=1e-12)
    return 2 * np.pi * r * rp * val / (np.pi**1.5 * beta**3)


@pytest.mark.parametrize(
    ("r", "rp", "ell"),
    [(2.0, 2.3, 0), (4.0, 3.6, 2), (6.0, 6.2, 5), (1.0, 3.0, 1), (3.0, 3.0, 8)],
)
def test_perey_buck_kernel_matches_angular_projection(r, rp, ell) -> None:
    beta = 0.85
    np.testing.assert_allclose(
        perey_buck_nonlocal(r, rp, beta, ell),
        _perey_buck_reference(r, rp, beta, ell),
        rtol=1e-10,
    )


def test_perey_buck_kernel_is_normalized_and_finite_at_large_radius() -> None:
    beta = 0.85
    rp = np.linspace(0.0, 30.0, 6001)
    for r in (8.0, 20.0):
        h = perey_buck_nonlocal(r, rp, beta, 0)
        assert np.all(np.isfinite(h))
        # h_l -> delta(r - r') for r >> beta
        np.testing.assert_allclose(np.trapezoid(h, rp), 1.0, rtol=1e-6)
    assert perey_buck_nonlocal(0.0, 2.0, beta, 0) == 0.0
    r = rp[:, None]
    np.testing.assert_allclose(
        perey_buck_nonlocal(r, rp, beta, 3), perey_buck_nonlocal(rp, r, beta, 3)
    )
