# Tests

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

regression-tests
```

## Set up the development environment

Clone the repository and install all dependency groups as described in
[Development environment](advanced-users.md#development-environment).

## Run the unit tests

```bash
uv run pytest
```

## Run the notebook tests

The example notebooks are tested with `pytest` and `nbval`. The
`--nbval-current-env` flag makes nbval ignore the kernel recorded in each
notebook and run the cells in the uv-managed environment, the same way CI does:

```bash
uv run --group examples pytest --nbval-lax --nbval-current-env examples/notebooks/
```

## Run the regression tests

End-to-end regression tests compare `jitr` against committed outputs from
Frescox and TALYS.  See [Regression tests](regression-tests.md) for the full
layout, the landed cases, and regeneration instructions.

## Browse the published examples

The curated notebook subset that appears on the documentation site is
listed in [Example notebooks](examples/index.md).

## Run the notebooks locally

See [Examples and tutorials](getting-started.md#examples-and-tutorials) for
how to launch JupyterLab against the uv-managed environment.
