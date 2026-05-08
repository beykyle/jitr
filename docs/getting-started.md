# Getting started

## Installation

Install the latest published package with:

```bash
pip install jitr
```

If you use `uv`, add it to a project with:

```bash
uv add jitr
```

## Development environment

This repository uses `uv` for dependency management and local environments.

```bash
uv sync --all-groups
```

Common validation commands:

```bash
uv run pytest
uv run --group examples pytest --nbval-lax examples/notebooks/
uv run ruff check .
uv run black --check .
uv run flake8 src tests
uv run mypy src
```

## Building the docs

The documentation site is built with Sphinx and MyST-NB. From the repository root, run:

```bash
uv run --group docs sphinx-build -W -b html -c docs . docs/_build/html
```

That build uses the repository root as the Sphinx source directory so the example notebooks under `examples/notebooks/` are included directly in the site.
