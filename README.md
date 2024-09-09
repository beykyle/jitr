[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)

# just-in-time R-Matrix (JITR)

A fast solver for parametric reaction models, production ready for calibration and uncertainty-quantification.

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
A framework for handling parametric reaction models.

Solves the radial BlochShrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh, using just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/). The theory generally follows:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = k_0 r$, where $r$ is the radial coordinate and $k_0$ is the entrance channel wavenumber.


## contributing

To set up the repository for contributing, or for access to the examples and notebooks, clone the repository and install locally:

```
git clone git@github.com:beykyle/jitr.git
pip install -e ./jitr
```

Feel free to fork and make a pull request if you have things to contribute. There are many [open issues](https://github.com/beykyle/jitr/issues), feel free to add more.

## examples and tutorials

various example scripts live in [`examples/`](https://github.com/beykyle/jitr/tree/main/examples). Tutorials live in [`examples/notebooks/`](https://github.com/beykyle/jitr/tree/main/examples/notebooks).

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
