[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
# Just-In-Time R-matrix (JITR)
## Quick start
```
 pip install jitr
```

Package hosted at [pypi.org/project/jitr/](https://pypi.org/project/jitr/).

## Description
Solves the radial Bloch-Shrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh, using just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/). The theory generally follows:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = kr$, where $r$ is the radial coordinate and $k$ is the channel wavenumber.

Capable of:
- non-local interactions
- coupled-channels


## Simple example: 2-body single-channel elastic scattering

```python
import numpy as np
import jitr
from numba import njit

@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return jitr.woods_saxon_potential(r, V0, W0, R0, a0) + jitr.coulomb_charged_sphere(
        r, zz, r_c
    )

energy_com = 26 # MeV
nodes_within_radius = 5

# initialize system and description of the channel (elastic) under consideration
sys = jitr.ProjectileTargetSystem(
    np.array([939.0]),
    np.array([nodes_within_radius * (2 * np.pi)]),
    l=np.array([0]),
    Ztarget=40,
    Zproj=1,
    nchannels=1,
)
ch = sys.build_channels(energy_com)

# initialize solver for single channel problem with 40 basis functions
solver = jitr.LagrangeRMatrixSolver(40, 1, sys)

# use same interaction for all channels
interaction_matrix = jitr.InteractionMatrix(1)
interaction_matrix.set_local_interaction(interaction)

# Woods-Saxon and Coulomb potential parameters
V0 = 60  # real potential strength
W0 = 20  # imag potential strength
R0 = 4  # Woods-Saxon potential radius
a0 = 0.5  # Woods-Saxon potential diffuseness
RC = R0  # Coulomb cutoff
params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, RC)

# set params
interaction_matrix.local_args[0,0] = params

# run solver
R, S, uext_boundary = solver.solve(interaction_matrix, ch, energy_com)

# get phase shift in degrees
delta, atten = jitr.delta(S[0,0])
print(f"phase shift: {delta:1.3f} + i {atten:1.3f} [degrees]")
```

This should print:

```
phase shift: -62.801 + i 90.910 [degrees]
```

## Simple 2-body coupled-channel system
Here we present the wavefunctions for a S-wave scattering on 3 coupled $0^+$ levels. For details, see [`examples/coupled`](https://github.com/beykyle/jitr/blob/main/examples/coupled.py).

![](https://github.com/beykyle/jitr/blob/main/assets/cc.png)


## citation
```latex
@software{Beyer_JITR_2023,
author = {Beyer, Kyle},
license = {BSD-3-Clause},
month = oct,
title = {{JITR}},
url = {https://github.com/beykyle/jitr},
version = {1.0},
year = {2023}
}
```
