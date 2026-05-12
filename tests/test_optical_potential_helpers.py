from __future__ import annotations

import numpy as np
import pytest

from jitr.optical_potentials import chuq, kduq, wlh


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
