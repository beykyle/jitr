[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
[![PyPI publish](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/pypi-publish.yml)

Documentation site: <https://beykyle.github.io/jitr/>

<p align="center">
<img src="./assets/jitr_logo.png" alt="drawing" width="300" /> 
</p>

# just-in-time R-Matrix (jitR)
A nuclear reaction toolkit, production ready for calibration and uncertainty-quantification, featuring:

- fast calculable $\mathcal{R}$-matrix solver for parametric reaction models
- built in uncertainty-quantified optical potentials
- built in nuclear data
- plenty of examples demonstrating the propagation of uncertainties into reaction observables and model calibration

Give your nuclear reaction UQ workflow a caffeine-kick with jitR!

## description
A framework for uncertainty-quantification of nuclear reaction observables using parametric reaction models. 

Under the hood, jitR solves the Shrödinger equation in partial waves using the calculable $\mathcal{R}$-Matrix method on a Lagrange-Legendre mesh. It is fast because it gives users the tools to precompute everything that they can for a system and reaction of interest, so given a single parameter sample, the minimal amount of compute is required to spit a cross section back out. For this reason, jitR is really suited to calculating an ensemble of observables for many parameter samples. Additionally, jitR relies on vectorized operations from [numpy](https://numpy.org/), as well as just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/) for the small subset of performance-critical code. 

The theory generally follows:
- [Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107](https://www.sciencedirect.com/science/article/pii/S0370157314004086)
- [Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219](https://www.sciencedirect.com/science/article/pii/S0010465515003951)
- [Descouvemont P. and Baye D. (2010). The R-matrix theory. Rep. Prog. Phys. 73 036301](https://iopscience.iop.org/article/10.1088/0034-4885/73/3/036301/meta)


## Documentation

The documentation site is available at <https://beykyle.github.io/jitr/>. It includes installation instructions, API reference, and examples. 

## Installation and Development

### For users

Install the latest released version of `jitr` from PyPI:

```bash
pip install jitr
```

Then use it in Python:

```python
import jitr
```

To check the installed version:

```bash
python -c "import jitr; print(jitr.__version__)"
```

If you use `uv`, you can add `jitr` to a uv-managed project with:

```bash
uv add jitr
```

Or install it into the current environment with:

```bash
uv pip install jitr
```

### For developers

This repository uses [`uv`](https://docs.astral.sh/uv/) for Python packaging, dependency management, virtual environments, and locking.

#### Clone the repository

```bash
git clone https://github.com/beykyle/jitr.git
cd jitr
```

#### Create and sync the development environment

Install all development dependencies:

```bash
uv sync --all-groups
```

This creates a local `.venv/` environment and installs the package in editable mode along with the development and example dependencies defined in `pyproject.toml`.

You can run commands through `uv`:

```bash
uv run python
uv run pytest
```

Or activate the environment manually:

```bash
source .venv/bin/activate
```

## Testing

### Run the unit tests

```bash
uv run pytest
```

### Run the notebook tests

The example notebooks are tested with `pytest` and `nbval`:

```bash
uv run --group examples pytest --nbval-lax examples/notebooks/
```

### Build the documentation website

The docs site uses Sphinx and MyST-NB, and it includes the repository notebooks directly:

```bash
uv run --group docs sphinx-build -W -b html -c docs . docs/_build/html
```

The published site is available at <https://beykyle.github.io/jitr/>.

Feel free to fork and make a pull request if you have things to contribute. There are many [open issues](https://github.com/beykyle/jitr/issues), feel free to add more.

## examples and tutorials

Various example scripts live in [`examples/`](https://github.com/beykyle/jitr/tree/main/examples). Tutorials live in [`examples/notebooks/`](https://github.com/beykyle/jitr/tree/main/examples/notebooks). Currently, the best way to run the notebooks is by cloning the repo and running them in a JupyterLab server with the uv-managed environment as the kernel. This way, you can be sure that all the dependencies are correct and that the notebooks will run as expected.

There are some additional requirements to run the examples. Once you've cloned the repo, from the main directory, run:

```
uv sync --group examples
```

Then, register the project environment as a Jupyter kernel:

```
uv run python -m ipykernel install --user --name jitr --display-name "Python (jitr)"
```

and start a Jupyter-lab server:

```
uv run --with jupyter jupyter lab
```

In JupyterLab, select the `Python (jitr)` kernel. Notebooks using this kernel will run against the uv-managed `.venv/` environment.

Then, you can run the notebooks. In particular, check out:
- [`examples/notebooks/reactions.ipynb`](https://github.com/beykyle/jitr/tree/main/examples/notebooks/reactions.ipynb) which demonstrates the use of the `reactions` submodule
- [`examples/notebooks/builtin_omps_uq.ipynb`](https://github.com/beykyle/jitr/tree/main/examples/notebooks/builtin_omps_uq.ipynb) to see how to use the built-in uncertainty-quantified optical model potentials to propagate uncertainties into reaction observables

## BAND

This package is part of the [BAND Framework](https://github.com/bandframework/)


## citations

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
