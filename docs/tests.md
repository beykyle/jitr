# Tests

## Set up the development environment

Clone the repository and install the full development environment:

```bash
git clone https://github.com/beykyle/jitr.git
cd jitr
uv sync --all-groups
```

This creates a local `.venv/` environment and installs the package in
editable mode together with the development, lint, docs, and example
dependencies defined in `pyproject.toml`.

You can run commands through `uv`:

```bash
uv run python
uv run pytest
```

Or activate the environment manually:

```bash
source .venv/bin/activate
```

## Run the unit tests

```bash
uv run pytest
```

## Run the notebook tests

The example notebooks are tested with `pytest` and `nbval`:

```bash
uv run --group examples pytest --nbval-lax examples/notebooks/
```

## Browse the published examples

The curated notebook subset that appears on the documentation site is
listed in [Example notebooks](examples/index.md).

## Run the notebooks locally

The notebooks live in
[`examples/notebooks/`](https://github.com/beykyle/jitr/tree/main/examples/notebooks).
To run them locally, install the example dependencies and register the
project environment as a Jupyter kernel:

```bash
uv sync --group examples
uv run python -m ipykernel install --user --name jitr --display-name "Python (jitr)"
uv run --with jupyter jupyter lab
```

In JupyterLab, select the `Python (jitr)` kernel so the notebooks run
against the `uv`-managed environment.
