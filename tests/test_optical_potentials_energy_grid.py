"""Built-in optical potentials evaluated over an energy grid equal the stacked
single-energy evaluations, with every term shaped (NE, N). (DOM: in
tests/test_dom_model.py.)"""

import numpy as np
import pytest

from jitr.optical_potentials import chuq, kduq, wlh
from jitr.optical_potentials.omp import LocalOpticalPotential
from jitr.reactions import Reaction

R = np.linspace(0.1, 12.0, 30)
NEUTRON, PROTON = (1, 0), (1, 1)


def check(model, reaction, kinematics, single, params):
    terms = model(R, reaction, kinematics, *params)
    stacked = [model(R, reaction, k, *params) for k in single]
    for t, s in zip(terms, zip(*stacked, strict=True), strict=True):
        assert np.shape(t) == (len(single), R.size)
        np.testing.assert_allclose(t, np.broadcast_to(np.array(s), np.shape(t)))


def grid(reaction, E):
    return reaction.kinematics(np.array(E)), [reaction.kinematics(e) for e in E]


@pytest.mark.parametrize(
    "model, params, projectile, target, E",
    [
        (kduq.KDUQ(NEUTRON), kduq.get_kd03(NEUTRON), NEUTRON, (56, 26), [1, 20, 150]),
        (kduq.KDUQ(PROTON), kduq.get_kd03(PROTON), PROTON, (208, 82), [10, 60]),
        (chuq.CHUQ(), chuq.get_ch89(), PROTON, (90, 40), [12, 40]),
        # the surface term switches off at 40 MeV (n) and 20 MeV (p, A > 100)
        (wlh.WLH(NEUTRON), None, NEUTRON, (90, 40), [20, 39, 41, 80]),
        (wlh.WLH(PROTON), None, PROTON, (208, 82), [15, 19, 21, 60]),
    ],
)
def test_global_potentials_on_an_energy_grid(model, params, projectile, target, E):
    if params is None:
        params = wlh.get_samples(projectile)[0]
    reaction = Reaction(target=target, projectile=projectile, process="El")
    check(model, reaction, *grid(reaction, E), params)


def test_local_potential_and_scalar_energy():
    reaction = Reaction(target=(40, 20), projectile=PROTON, process="El")
    params = [50, 1.2, 0.7, 5, 1.2, 0.7, 4, 0, 1.25, 0.6, 6, 0, 1.0, 0.6, 1.3]
    check(LocalOpticalPotential(), reaction, *grid(reaction, [10, 30]), params)
    terms = kduq.KDUQ(PROTON)(
        R, reaction, reaction.kinematics(20.0), *kduq.get_kd03(PROTON)
    )
    assert all(np.shape(t) == R.shape for t in terms)
