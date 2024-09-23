[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
[![PyPI publish](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml)

# just-in-time R-Matrix (JITR)

A fast solver for parametric reaction models, production ready for calibration and uncertainty-quantification.

## quick start

```
 pip install jitr
```

The release versions of the package are hosted at [pypi.org/project/jitr/](https://pypi.org/project/jitr/).

## description
A framework for handling parametric reaction models.

Solves the radial Bloch-Shrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh. Fairly fast due to using vectorized operations from [numpy](https://numpy.org/) and just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/). 

The theory generally follows:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = k_0 r$, where $r$ is the radial coordinate and $k_0$ is the entrance channel wavenumber.


## contributing, developing, and testing

To set up the repository for contributing, testing, access to non-release branches, access to the examples and notebooks, etc., clone the repository and install locally:

```
git clone git@github.com:beykyle/jitr.git
pip install -r ./jitr/requirements.txt
pip install -e ./jitr
```

then run the tests from the main project directory:

```
pytest jitr
```

Feel free to fork and make a pull request if you have things to contribute. There are many [open issues](https://github.com/beykyle/jitr/issues), feel free to add more.

## examples and tutorials

Various example scripts live in [`examples/`](https://github.com/beykyle/jitr/tree/main/examples). Tutorials live in [`examples/notebooks/`](https://github.com/beykyle/jitr/tree/main/examples/notebooks).

In particular, [`examples/notebooks/kduq_cross_section_uq_tutorial.ipynb`](https://github.com/beykyle/jitr/tree/main/examples/notebooks/kduq_cross_section_uq_tutorial.ipynb) demonstrates how to perform UQ for $(n,n)$ cross sections using [KDUQ](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.107.014602).

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
