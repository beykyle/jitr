"""Characterization: ILDA folding reproduces the golden."""

from ._cases import assert_matches_golden, compute_folding_case, load_golden


def test_folding_matches_golden() -> None:
    assert_matches_golden(compute_folding_case(), load_golden("folding_ilda"))
