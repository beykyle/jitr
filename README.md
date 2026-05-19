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
