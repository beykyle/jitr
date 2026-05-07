# Copilot instructions for this repository

This is a Python library project managed with `uv`.

## Dependency management

* Use `uv` for Python dependency management.
* Do not use `pip install`, `conda install`, `mamba install`, or `python -m venv` unless explicitly asked.
* Do not manually edit `uv.lock`.
* Keep dependency declarations in `pyproject.toml`.
* Commit both `pyproject.toml` and `uv.lock` after dependency changes.

## Running commands

Use `uv run` for Python commands.

## Setup

Create or update the local development environment with:

```bash
uv sync --all-groups
```


## Testing and validation

Run unit tests:

```bash
uv run pytest
```

Run notebook tests:

```bash
uv run --group examples pytest --nbval-lax examples/notebooks/
```

Run linting and type checks:

```bash
uv run ruff check .
uv run black --check .
uv run flake8 src tests
uv run mypy src
```

When fixing formatting or lint issues, prefer:

```bash
uv run ruff check . --fix
uv run black .
```

## Build

Build the package with:

```bash
uv build
```

Built artifacts are written to `dist/`.

## Project layout

* Source code lives in `src/jitr/`.
* Unit tests live in `tests/`.
* Example notebooks live in `examples/notebooks/`.
* Dependencies are declared in `pyproject.toml`.
* Locked dependency resolution is stored in `uv.lock`.
* The project includes `src/jitr/py.typed`; maintain type annotations for public APIs where practical.

## General guidance

* Use `uv run ...` for Python, test, lint, type-check, notebook, and build commands.
* Do not introduce new `requirements.txt` files unless explicitly requested.
* Do not edit generated files or lockfiles by hand.
* Keep changes focused on the requested task.
* Run the relevant validation commands before considering the task complete.

## Public API
- Avoid changing public APIs unless explicitly requested.
- If public APIs change, update tests, examples, and documentation.
- Preserve backward compatibility where practical.

## Typing
- This package is typed.
- Add or preserve type annotations for public functions and classes.
- Avoid broad `Any` unless necessary.
- Keep `mypy` passing after type-related changes.

## Dependencies
- Do not add new dependencies unless needed.
- Prefer standard-library solutions for small utilities.
- Put runtime dependencies in `[project.dependencies]`.
- Put test, lint, notebook, or development-only dependencies in dependency groups.
