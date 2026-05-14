# Getting started

`jitr` is a nuclear reaction toolkit, production ready for calibration and uncertainty-quantification, featuring:

- fast calculable $\mathcal{R}$-matrix solver for parametric reaction models
- built in uncertainty-quantified optical potentials
- built in nuclear data
- plenty of examples demonstrating the propagation of uncertainties into reaction observables and model calibration

Give your nuclear reaction UQ workflow a caffeine-kick with jitr!

## description
A framework for uncertainty-quantification of nuclear reaction observables using parametric reaction models. 

Under the hood, jitr solves the Shrödinger equation in partial waves using the calculable $\mathcal{R}$-Matrix method on a Lagrange-Legendre mesh. It is fast because it gives users the tools to precompute everything that they can for a system and reaction of interest, so given a single parameter sample, the minimal amount of compute is required to spit a cross section back out. For this reason, jitr is really suited to calculating an ensemble of observables for many parameter samples. Additionally, jitr relies on vectorized operations from [numpy](https://numpy.org/), as well as just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/) for the small subset of performance-critical code. 

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

## Start here

If you are learning the package for the first time, this is the
recommended path through the documentation:

1. Browse the curated [example notebooks](examples/index.md) to see the
   main workflows in context.
2. Start with
   [`local_omp_demo`](../examples/notebooks/local_omp_demo.ipynb) for a
   concrete elastic-scattering workflow, or
   [`how_to_define_your_interaction`](../examples/notebooks/how_to_define_your_interaction.ipynb)
   if you want to understand how to define local, nonlocal, and
   coupled-channel interactions.
3. Use the [API reference](api/index.md) to drill into the modules that
   appear in those examples.

For development setup, test commands, and documentation builds, see
[Advanced users and developers](advanced-users.md) and
[Tests](tests.md).

## BAND

`jitr` is part of the
[BAND Framework](https://github.com/bandframework/).

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
