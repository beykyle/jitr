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


## Building documentation

Build the docs locally:

```bash
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

Output lands in `docs/_build/html/`. The `docs/examples/notebooks` and `docs/tests` symlinks are gitignored; create them once before building:

```bash
mkdir -p docs/examples && ln -sf "$(pwd)/examples/notebooks" docs/examples/notebooks
ln -sf "$(pwd)/tests" docs/tests
```

## Testing and validation

Run unit tests:

```bash
uv run pytest
```

Run a focused test file or test selection:

```bash
uv run pytest tests/test_local.py
uv run pytest tests/test_local.py -k some_test_name
```

Run notebook tests:

```bash
uv run --group examples pytest --nbval-lax examples/notebooks/
```

Notebook tests are a separate CI gate and take much longer than the unit suite. Do not treat `uv run pytest` as full validation when changes can affect examples, docs-facing APIs, or shared numerical behavior.

Run linting and type checks:

```bash
uv run ruff check .
uv run black --check .
uv run flake8 src tests
uv run mypy src
```

For changes to typing, linting, formatting, workflow files, test layout, or other repo-wide quality/configuration surfaces, run the full validation matrix before concluding work:

```bash
uv run ruff check .
uv run black --check .
uv run flake8 src tests
uv run mypy src
uv run pytest
uv run --group examples pytest --nbval-lax examples/notebooks/
uv build
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
* Keep references to the test suite consistent across the repository: if test paths change, update `pyproject.toml`, workflows, and instructions together.

## Public API
- Avoid changing public APIs unless explicitly requested.
- If public APIs change, update tests, examples, and documentation.
- Preserve backward compatibility where practical.

## Typing
- This package is typed.
- Add or preserve type annotations for public functions and classes.
- Avoid broad `Any` unless necessary.
- Keep `mypy` passing after type-related changes.
- The mypy policy for this repo lives in `pyproject.toml` under `[tool.mypy]` and `[[tool.mypy.overrides]]`; adjust that first when dealing with third-party stubs or intentionally excluded legacy modules instead of changing the workflow.

## Dependencies
- Do not add new dependencies unless needed.
- Prefer standard-library solutions for small utilities.
- Put runtime dependencies in `[project.dependencies]`.
- Put test, lint, notebook, or development-only dependencies in dependency groups.

## Linting conventions
- Ruff, Black, and Flake8 are intentionally aligned around an 88-character line length.
- Flake8 is configured through `pyproject.toml` and depends on `flake8-pyproject` in the `lint` dependency group; keep that plugin in place if you expect Flake8 to honor repo settings.
- Notebook cells use a targeted Ruff ignore for `examples/notebooks/*.ipynb` `E501`; prefer that narrow exception over broad global relaxations.
