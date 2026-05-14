# Advanced users and developers

## Development environment

This repository uses `uv` for dependency management and local
environments.

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

The documentation site is built with Sphinx and MyST-NB. From the
repository root, run:

```bash
uv run --group docs sphinx-build -W -b html -c docs . docs/_build/html
```

That build uses the repository root as the Sphinx source directory, so
published notebooks can be linked directly from `examples/notebooks/`
without copying them into `docs/`.

## Contributing

Contributions are welcome. If you have improvements, bug fixes, or new
examples, feel free to open a pull request. The
[issue tracker](https://github.com/beykyle/jitr/issues) is also a good
place to propose documentation and tutorial improvements.
