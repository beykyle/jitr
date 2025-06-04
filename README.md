[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
[![PyPI publish](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml)

<p align="center">
<img src="./assets/jitr_logo.png" alt="drawing" width="300" /> 
</p>

# just-in-time R-Matrix (jitR)
A fast calculable $\mathcal{R}$-matrix solver for parametric reaction models, production ready for calibration and uncertainty-quantification. Give your UQ workflow a caffeine-kick with jitR!


## quick start

```
 pip install jitr
```

The release versions of the package are hosted at [pypi.org/project/jitr/](https://pypi.org/project/jitr/).

## description
A framework for uncertainty-quantification of nuclear reaction observables using parametric reaction models. Consider a local coordinate-space potential $V(r;\theta)$ that is a function of some parameters $\theta$. Just write it like so:

```python
def V(r,*theta):
  a,b,c,... = theta
  # calculate and return potential at radial coordinate r as a function of parameters a,b,c,...
```

Then, you can pass it along with many samples of $\theta$ into jitR to calculate many samples of the corresponding cross sections for your system and reaction of interest! The reaction observables jitR can calculate are represented as `Workspace` instances, and live in `src/jitr/xs/`.

Under the hood, jitR solves the radial Bloch-Shrödinger equation in the continuum using the calculable $\mathcal{R}$-Matrix method on a Lagrange-Legendre mesh. It is fast because it gives users the tools to precompute everything that they can for a system and reaction of interest, so given a single parameter sample, the minimal amount of compute is required to spit a cross section back out. For this reason, jitR is really suited to calculating an ensemble of observables for many parameter samples. Additionally, jitR relies on vectorized operations from [numpy](https://numpy.org/), as well as just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/) for the small subset of performance-critical code. 

The theory generally follows:
- [Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107](https://www.sciencedirect.com/science/article/pii/S0370157314004086)
- [Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219](https://www.sciencedirect.com/science/article/pii/S0010465515003951)
- [Descouvemont P. and Baye D. (2010). The R-matrix theory. Rep. Prog. Phys. 73 036301](https://iopscience.iop.org/article/10.1088/0034-4885/73/3/036301/meta)

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = k_0 r$, where $r$ is the radial coordinate and $k_0$ is the entrance channel wavenumber.


## contributing, developing, and testing

To set up the repository for contributing, testing, access to non-release branches, access to the examples and notebooks, etc., clone the repository and install locally:

```
git clone git@github.com:beykyle/jitr.git
pip install -r ./jitr/requirements.txt
pip install -e ./jitr
```

then run the tests:

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
