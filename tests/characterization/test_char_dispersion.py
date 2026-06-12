"""Characterization: dispersive-OMP kernel reproduces the golden."""

from ._cases import assert_matches_golden, compute_dispersion_case, load_golden


def test_dispersion_matches_golden() -> None:
    assert_matches_golden(compute_dispersion_case(), load_golden("dispersion_dom"))
