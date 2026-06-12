"""Characterization: quasi-elastic (p,n) workspace reproduces the golden."""

from ._cases import assert_matches_golden, compute_qepn_case, load_golden


def test_qepn_matches_golden() -> None:
    assert_matches_golden(compute_qepn_case(), load_golden("qepn_ca48"))
