from __future__ import annotations

import numpy as np

from tests.regression._builders import build_case
from tests.regression._readers import ManifestEntry, load_case


def test_regression(case: ManifestEntry) -> None:
    """Compare one committed external reference against the current API."""
    ref = load_case(case)
    built = build_case(ref)
    computed = built.observables()
    np.testing.assert_allclose(
        computed["dsdo_mb_per_sr"],
        ref.dsdo,
        rtol=ref.tolerance["rtol"],
        atol=ref.tolerance["atol"],
        err_msg=f"[{ref.case_id} / {ref.reference_code}] dsdo disagrees",
    )
    for name, expected in ref.observables.items():
        np.testing.assert_allclose(
            computed[name],
            expected,
            **ref.tolerances[name],
            err_msg=f"[{ref.case_id} / {ref.reference_code}] {name} disagrees",
        )
