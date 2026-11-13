import numpy as np

from jitr.reactions import ElasticReaction
from jitr.xs.coupled import Potential
from jitr.xs.elastic import Workspace


def _gaussian_potential(
    rgrid: np.ndarray, strength: complex, radius: float
) -> np.ndarray:
    return np.asarray(strength * np.exp(-((rgrid / radius) ** 2)), dtype=np.complex128)


def test_elastic_workspace_treats_missing_spin_orbit_as_zero() -> None:
    reaction = ElasticReaction((48, 20), (1, 0))
    workspace = Workspace(
        reaction,
        reaction.kinematics(12.0),
        channel_radius_fm=8.0,
        lmax=4,
        nbasis=16,
        angles=np.linspace(0.1, np.pi - 0.1, 7),
    )
    central = _gaussian_potential(workspace.r, -25.0 - 2.5j, 3.0)
    omitted = workspace.xs(Potential(central=central))
    explicit = workspace.xs(Potential(central=central, spin_orbit=0 * central))
    for a, b in zip(omitted, explicit, strict=True):
        np.testing.assert_allclose(a, b)
