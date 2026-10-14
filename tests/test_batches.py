"""A batch of potentials (one per parameter sample) through jax.vmap equals
solving them one at a time."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jitr.optical_potentials import kduq
from jitr.reactions import Reaction
from jitr.xs import elastic, quasielastic_pn
from jitr.xs.coupled import Potential

P48CA = Reaction(target=(48, 20), projectile=(1, 1), process="El")
N48SC = Reaction(target=(48, 21), projectile=(1, 0), process="El")
PN = Reaction(target=(48, 20), projectile=(1, 1), product=(1, 0), residual=(48, 21))


def stack(potentials):
    return jax.tree.map(lambda *x: jnp.stack(x), *potentials)


def kduq_samples(reaction, ws, kinematics, projectile, n=6):
    model = kduq.KDUQ(projectile)
    samples = kduq.get_samples(projectile)[:n]
    neutron = projectile == (1, 0)
    return [
        Potential(*model(ws.r, reaction, kinematics, *s)[: 2 if neutron else 3])
        for s in samples
    ]


def test_elastic_batch_equals_loop():
    kin = P48CA.kinematics(np.array([15.0, 40.0]))
    ws = elastic.Workspace(P48CA, kin, 12.0, 20, 40, angles=np.linspace(0.2, 3, 25))
    potentials = kduq_samples(P48CA, ws, kin, (1, 1))
    batch = jax.vmap(ws.xs)(stack(potentials))
    for i, p in enumerate(potentials):
        for a, b in zip(batch, ws.xs(p), strict=True):
            np.testing.assert_allclose(a[i], b, rtol=1e-10, atol=1e-13)
    S = jax.vmap(ws.smatrix)(stack(potentials))
    assert S.shape == (len(potentials), *ws.smatrix(potentials[0]).shape)


@pytest.fixture(scope="module")
def pn_workspace():
    ke = PN.kinematics(np.array([35.0]))
    kx = PN.kinematics_exit(ke, residual_excitation_energy=6.7)
    ws = quasielastic_pn.Workspace(
        PN, ke, kx, 12.0, 20, 40, angles=np.linspace(0.1, 1.5, 15)
    )
    return ws, kduq_samples(P48CA, ws, ke, (1, 1)), kduq_samples(N48SC, ws, kx, (1, 0))


@pytest.mark.parametrize("method", ["dwba", "cc"])
def test_pn_batch_equals_loop(pn_workspace, method):
    ws, Up, Un = pn_workspace
    batch = jax.vmap(getattr(ws, method))(stack(Up), stack(Un))
    for i in range(len(Up)):
        single = getattr(ws, method)(Up[i], Un[i])
        np.testing.assert_allclose(batch.dsdo[i], single.dsdo, rtol=1e-10)
        np.testing.assert_allclose(batch.Ay[i], single.Ay, atol=1e-12)


def test_shared_potential_with_in_axes(pn_workspace):
    """A potential that does not vary across the batch is passed once."""
    ws, Up, Un = pn_workspace
    batch = jax.vmap(ws.dwba, in_axes=(0, None))(stack(Up), Un[0])
    for i in range(len(Up)):
        np.testing.assert_allclose(
            batch.dsdo[i], ws.dwba(Up[i], Un[0]).dsdo, rtol=1e-10
        )
