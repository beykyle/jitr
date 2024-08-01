[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
# just-in-time R-Matrix (JITR)
## quick start
```
 pip install jitr
```

Package hosted at [pypi.org/project/jitr/](https://pypi.org/project/jitr/).

## testing
From the main repository directory, run:

```
pytest
```

## examples and tutorials

A variety of examples live in [examples/](https://github.com/beykyle/jitr/tree/main/examples)

## description
Solves the radial Bloch-Shrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh, using just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/). The theory generally follows:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = kr$, where $r$ is the radial coordinate and $k$ is the channel wavenumber.

Capable of:
- non-local interactions
- coupled-channels


## simple example: 2-body single-channel elastic scattering

```python
import numpy as np
import jitr
from numba import njit


@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    nuclear = jitr.woods_saxon_potential(r, V0, W0, R0, a0)
    coulomb = jitr.coulomb_charged_sphere(r, zz, r_c)
    return nuclear + coulomb


nodes_within_radius = 5
a = 2 * np.pi * nodes_within_radius

E_lab = 35  # MeV

# target (A,Z)
Ca48 = (28, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2

sys = jitr.ProjectileTargetSystem(
    channel_radii=np.array([a]),
    l=np.array([0]),
    mass_target=mass_Ca48,
    mass_projectile=mass_proton,
    Ztarget=Ca48[1],
    Zproj=proton[1],
)

# initialize solver
solver = jitr.RMatrixSolver(nbasis=40)

channels = sys.build_channels_kinematics(E_lab)

# use same interaction for all channels
interaction_matrix = jitr.InteractionMatrix(1)
interaction_matrix.set_local_interaction(interaction)

# Woods-Saxon and Coulomb potential parameters
V0 = 60  # real potential strength
W0 = 20  # imag potential strength
R0 = 4  # Woods-Saxon potential radius
a0 = 0.5  # Woods-Saxon potential diffuseness
RC = R0  # Coulomb cutoff
```

This should print:

```
phase shift: 86.981 + i 57.979 [degrees]
```

## simple coupled-channel system
Here we present the wavefunctions for a S-wave scattering on 3 coupled $0^+$ levels. For details, see [`examples/coupled`](https://github.com/beykyle/jitr/blob/main/examples/coupled.py).

![](https://github.com/beykyle/jitr/blob/main/assets/cc.png)


## citation
```latex
@software{Beyer_JITR_2024,
author = {Beyer, Kyle},
license = {BSD-3-Clause},
month = oct,
title = {{JITR}},
url = {https://github.com/beykyle/jitr},
version = {1.3.0},
year = {2024}
}
```
