# Advanced users and developers

## Development environment

This repository uses [`uv`](https://docs.astral.sh/uv/) for Python packaging,
dependency management, virtual environments, and locking.

Clone the repository and install the full development environment:

```bash
git clone https://github.com/beykyle/jitr.git
cd jitr
uv sync --all-groups
```

This creates a local `.venv/` environment and installs the package in
editable mode together with the test, lint, docs, and example dependencies
defined in `pyproject.toml`.

You can run commands through `uv`:

```bash
uv run python
uv run pytest
```

Or activate the environment manually:

```bash
source .venv/bin/activate
```

### Linting and type checking

```bash
uv run ruff check .
uv run black --check .
uv run flake8 src tests
uv run mypy src
```

Test commands are documented on the [Tests](tests.md) page.

## Building the docs

The documentation site is built with Sphinx and MyST-NB. From the
repository root, run:

```bash
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

The notebooks rendered on the site are copies of the ones in
`examples/notebooks/`, kept under `docs/examples/notebooks/`.

## Contributing

Contributions are welcome. If you have improvements, bug fixes, or new
examples, feel free to open a pull request. The
[issue tracker](https://github.com/beykyle/jitr/issues) is also a good
place to propose documentation and tutorial improvements.
