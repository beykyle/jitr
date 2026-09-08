[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
[![PyPI publish](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml)

Documentation site: <https://beykyle.github.io/jitr/>

<p align="center">
<img src="./assets/jitr_logo.png" alt="drawing" width="300" /> 
</p>

# just-in-time R-Matrix (jitr)
A nuclear reaction toolkit, production ready for calibration and uncertainty-quantification, featuring:

- fast calculable $\mathcal{R}$-matrix solver for parametric reaction models
- built in uncertainty-quantified optical potentials
- built in nuclear data
- plenty of examples demonstrating the propagation of uncertainties into reaction observables and model calibration

Give your nuclear reaction UQ workflow a caffeine-kick with jitr!

## Documentation

The documentation site is available at <https://beykyle.github.io/jitr/>.

Check out the [getting started page](https://beykyle.github.io/jitr/getting-started.html) for an overview of the package and quickstart tutorials.

Check out the [tutorials and examples page](https://beykyle.github.io/jitr/examples/index.html) for a full list of example notebooks and demos.

## Installation

```bash
pip install jitr
```

For `uv` users and other install options, see [Installation](https://beykyle.github.io/jitr/getting-started.html#installation).

## Development and testing

- [Development environment](https://beykyle.github.io/jitr/advanced-users.html#development-environment): cloning the repo, syncing the uv-managed environment, and linting.
- [Tests](https://beykyle.github.io/jitr/tests.html): running the unit, notebook, and regression tests.
- [Building the docs](https://beykyle.github.io/jitr/advanced-users.html#building-the-docs): building this documentation site locally.

## Tutorials

Tutorials live in [`examples/notebooks/`](./examples/notebooks). See [Examples and tutorials](https://beykyle.github.io/jitr/getting-started.html#examples-and-tutorials) for how to run them locally, and the [examples page](https://beykyle.github.io/jitr/examples/index.html) for an annotated list.

## BAND

This package is part of the [BAND Framework](https://bandframework.github.io/).

## Citations

Please consider citing both this package and the BAND Framework if you use this code in your research. The BibTeX entries are:

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
