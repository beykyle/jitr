[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)

# just-in-time R-Matrix (JITR)

A fast solver for parametric reaction models

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

## description
Solves the radial Bloch-Shrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh, using just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/). The theory generally follows:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = k_0 r$, where $r$ is the radial coordinate and $k_0$ is the entrance channel wavenumber.

Capable of:
- non-local interactions
- coupled-channels


## examples and tutorials

- Various notebooks and scripts live in [`examples/`](https://github.com/beykyle/jitr/tree/main/examples)
- Here are wavefunctions for a S-wave scattering on 3 coupled $0^+$ levels. For details, see [`examples/coupled`](https://github.com/beykyle/jitr/blob/main/examples/coupled.py).

![](https://github.com/beykyle/jitr/blob/main/assets/cc.png)


## BAND

This package is part of the [BAND Framework](https://github.com/bandframework/)


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
