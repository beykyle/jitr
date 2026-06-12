"""Non-local (Yamaguchi) potential example on the lax-backed workspace.

Non-local kernels are supplied as raw ``K(r, r')`` values (or an
``f(r, r')`` callable) — the solver applies the Gauss quadrature scaling
internally. The s-wave phase shifts are anchored to the published
Lagrange-mesh reference values (Descouvemont 2016, Example 5: N=10,
a=8 fm). Note the legacy ``jitr.rmatrix`` engine's non-local path did
*not* reproduce these values; the lax-backed workspaces do.
"""

import numpy as np

from jitr.optical_potentials.potential_forms import yamaguchi_potential
from jitr.reactions import ElasticReaction
from jitr.utils.constants import HBARC
from jitr.utils.kinematics import ChannelKinematics
from jitr.xs.elastic import IntegralWorkspace

ALPHA = 0.2316053  # fm**-1
BETA = 1.3918324  # fm**-1
W0 = 41.472  # MeV fm**2 (= hbar^2/2mu for the deuteron channel)

# Descouvemont (2016), Example 5 / Appendix E
REFERENCE = {0.1: -15.0770, 10.0: 85.6370}


def nonlocal_interaction_example():
    params = (W0, BETA, ALPHA)
    mu = HBARC**2 / (2 * W0)
    reaction = ElasticReaction((48, 20), (1, 0))  # supplies the (zero) charge

    print("\nYamaguchi potential, s-wave phase shifts:")
    for ecom, reference in REFERENCE.items():
        k = np.sqrt(2 * mu * ecom) / HBARC
        kin = ChannelKinematics(Elab=ecom, Ecm=ecom, mu=mu, k=k, eta=0.0)
        workspace = IntegralWorkspace(
            reaction=reaction,
            kinematics=kin,
            channel_radius_fm=8.0,
            lmax=0,
            nbasis=10,
        )
        splus, _ = workspace.smatrix(
            lambda r, rp: yamaguchi_potential(r, rp, *params),
            energy_dependent=False,
        )
        delta = np.rad2deg(np.real(np.log(complex(np.asarray(splus)[0, 0])) / 2j))
        print(
            f"  E = {ecom:5.1f} MeV: {delta:10.4f} deg "
            f"(published reference {reference:10.4f})"
        )
        assert abs(delta - reference) < 5e-4


if __name__ == "__main__":
    nonlocal_interaction_example()
