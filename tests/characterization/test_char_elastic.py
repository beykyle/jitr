"""Characterization: elastic workspaces reproduce the pinned golden data."""

from ._cases import assert_matches_golden, compute_elastic_case, load_golden


def test_elastic_matches_golden() -> None:
    assert_matches_golden(compute_elastic_case(), load_golden("elastic_pca"))
