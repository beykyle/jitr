# Getting started

`jitr` is a nuclear reaction toolkit, production ready for calibration and uncertainty-quantification, featuring:

- fast calculable $\mathcal{R}$-matrix solver for parametric reaction models
- built in uncertainty-quantified optical potentials
- built in nuclear data
- plenty of examples demonstrating the propagation of uncertainties into reaction observables and model calibration

Give your nuclear reaction UQ workflow a caffeine-kick with jitr!

## Description

Under the hood, jitr solves the Shrödinger equation using the calculable $\mathcal{R}$-Matrix method on a Lagrange-Legendre mesh. It is fast because it gives users the tools to precompute everything that they can for a system and reaction of interest, so given a single parameter sample, the minimal amount of compute is required to spit a cross section back out. For this reason, jitr is really suited to calculating an ensemble of observables corresponding to an ensemble of reactions. Additionally, jitr relies on vectorized operations from [numpy](https://numpy.org/), as well as just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/) for the small subset of performance-critical code. 

The theory generally follows:
- [Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107](https://www.sciencedirect.com/science/article/pii/S0370157314004086)
- [Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219](https://www.sciencedirect.com/science/article/pii/S0010465515003951)
- [Descouvemont P. and Baye D. (2010). The R-matrix theory. Rep. Prog. Phys. 73 036301](https://iopscience.iop.org/article/10.1088/0034-4885/73/3/036301/meta)


## Installation

Install the latest published package with:

```bash
pip install jitr
```

If you use `uv`, add it to an existing project with:

```bash
uv add jitr
```

## Examples and tutorials

Browse the curated [example notebooks](examples/index.md).

### Quick start

Here is full end-to-end example for 
- compiling a solver for a given reaction system
- defining a parametric interaction potential
- calculating an elastic scattering cross section for an ensemble of potential parameters

This is adapted from the [quickstart example](/examples/notebooks/quickstart). 
```python
from jitr.reactions.reaction import Reaction
from jitr.xs import elastic
from jitr.rmatrix import Solver as SolverKernel
from jitr.optical_potentials.potential_forms import (
    woods_saxon_safe,
    woods_saxon_prime_safe,
    coulomb_charged_sphere,
)

from jitr.utils import utils

import numpy as np
from scipy import stats
from tqdm import tqdm
from matplotlib import pyplot as plt

# define reaction system
alpha = (4, 2)
Ca48 = (48, 20)
reaction = Reaction(target=Ca48, projectile=alpha, process="El")

# calculate kinematics for a given lab energy
energy_lab = 28.2
kinematics = reaction.kinematics(energy_lab)

# set the channel radius, number of nodes, and number of partial waves
interaction_range_fm = 1.5 * (48 ** (1 / 3) + 4 ** (1 / 3)) + 3
channel_radius_dimensionless = utils.suggested_dimensionless_channel_radius(
    interaction_range_fm, kinematics.k
)
channel_radius = channel_radius_dimensionless / kinematics.k
N = utils.suggested_basis_size(channel_radius_dimensionless)
lmax = 180

# build a solver for the system and reaction of interest
print(f"Compiling solver for {reaction} at {energy_lab} MeV")
print(f" - channel radius {channel_radius:1.2f} fm")
print(f" - {N} nodes")
print(f" - {lmax} partial waves")

solver = elastic.DifferentialWorkspace.build_from_system(
    reaction=reaction,
    kinematics=kinematics,
    channel_radius_fm=channel_radius,
    solver=SolverKernel(N),
    lmax=lmax,
    angles=np.linspace(0.1, np.pi, 180),
)
rgrid = solver.radial_grid()
print("Done!")
```

```
Compiling solver for 48-Ca(alpha,el) at 28.2 MeV
 - channel radius 13.76 fm
 - 50 nodes
 - 180 partial waves
Done!
```

```python
# define interaction
def U_central(r, Vv, Wv, Rv, av, Rd, ad):
    return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)


def V_Coulomb(r, RC):
    Zz = reaction.target.Z * reaction.projectile.Z
    return coulomb_charged_sphere(r, Zz, RC)


# define parameter distribution and draw samples
# just
param_means = np.array([185, 20, 1.0, 0.6, 1.8, 0.5, 1.2])
param_std_devs = np.array([6, 2, 0.05, 0.05, 0.1, 0.05, 0.05])
num_samples = 1000
param_draws = stats.multivariate_normal(
    mean=param_means, cov=np.diag(param_std_devs) ** 2
).rvs(num_samples)

print(f"Running {num_samples} calculations...")
prediction_samples = np.zeros((num_samples, solver.angles.size))
for i in tqdm(range(param_draws.shape[0])):
    Vv, Wv, rv, av, rd, ad, rC = param_draws[i]
    A_factor = reaction.target.A ** (1 / 3) + reaction.projectile.A ** (1 / 3)
    xs = solver.xs(
        central_potential=U_central(
            rgrid, Vv, Wv, rv * A_factor, av, rd * A_factor, ad
        ),
        coulomb_potential=V_Coulomb(rgrid, rC * A_factor),
    )
    prediction_samples[i, :] = xs.dsdo / solver.rutherford

print("Done!")
```

```
Running 1000 calculations...

100%|███████████████████████████████████████████████████████████| 1000/1000 [00:18<00:00, 55.52it/s]

Done!

```


## API reference and development

Use the [API reference](api/index.md) for detailed documentation of the codebase.

For development setup, test commands, and documentation builds, see
[Advanced users and developers](advanced-users.md) and
[Tests](tests.md).

## BAND

`jitr` is one of the siftware packages included in the [BAND Framework](https://bandframework.github.io/).

## Citations

Please consider citing both `jitr` and the BAND Framework if you use the
code in your research. The BibTeX entries are:

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

```latex
@techreport{bandframework,
    title       = {{BANDFramework: An} Open-Source Framework for {Bayesian} Analysis of Nuclear Dynamics},
    author      = {Kyle Beyer and Landon Buskirk and Manuel Catacora Rios and Moses Y-H. Chan and Tyler H. Chang and Troy Dasher 
    and Richard James DeBoer and Christian Drischler and Richard J. Furnstahl and Pablo Giuliani and
    Kyle Godbey and Kevin Ingles and Sunil Jaiswal and An Le and Dananjaya Liyanage and Filomena M. Nunes
    and Daniel Odell and David O'Gara and Jared O'Neal and Daniel R. Phillips and Matthew Plumlee
    and Matthew T. Pratola and Scott Pratt and Oleh Savchuk and Alexandra C. Semposki and \"Ozge S\"urer and
    Stefan M. Wild and John C. Yannotty},
    institution = {},
    number      = {Version 0.5.0},
    year        = {2025},
    url         = {https://github.com/bandframework/bandframework}
}
```
