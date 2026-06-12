"""Shared pytest configuration for the jitr test suite."""

import importlib.util

import pytest

HAVE_LAX = importlib.util.find_spec("lax") is not None

requires_lax = pytest.mark.requires_lax


def pytest_collection_modifyitems(items):
    if HAVE_LAX:
        return
    skip_lax = pytest.mark.skip(
        reason="requires the lax solver package (editable install; not yet on PyPI)"
    )
    for item in items:
        if "requires_lax" in item.keywords:
            item.add_marker(skip_lax)
