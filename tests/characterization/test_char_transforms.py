"""Characterization: Fourier-Bessel transforms reproduce the golden."""

from ._cases import assert_matches_golden, compute_transforms_case, load_golden


def test_transforms_match_golden() -> None:
    assert_matches_golden(compute_transforms_case(), load_golden("transforms_fb"))
