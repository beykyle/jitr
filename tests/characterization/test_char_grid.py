"""Characterization: radial grids and one wavefunction reproduce the golden.

Contrary to the design doc's §3.3 premise, the *public* ``radial_grid()`` is
already energy-independent: the internal mesh lives in s = k·r, but the public
grid is ``abscissa · (a/k) = abscissa · channel_radius_fm``, the k-scaling
cancels, and the result is bitwise identical to lax's
``MeshSpec("legendre", "x")`` radii for the same (nbasis, R). The grid
contract therefore survives the rewrite unchanged; these tests pin that.
"""

import numpy as np

from ._cases import assert_matches_golden, compute_grid_case, load_golden


def test_grids_and_wavefunction_match_golden() -> None:
    assert_matches_golden(compute_grid_case(), load_golden("grids"))


def test_grid_is_energy_independent_fm() -> None:
    # equal up to the a = R·k, a/k round-trip (1 ulp), not bitwise
    golden = load_golden("grids")
    for i in range(1, golden["Elab"].size):
        np.testing.assert_allclose(
            golden["rgrid_0"], golden[f"rgrid_{i}"], rtol=1e-14, atol=0.0
        )
