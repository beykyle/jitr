from __future__ import annotations

import numpy as np

from tests.regression._builders import build_case
from tests.regression._readers import ManifestEntry, load_case


def test_regression(case: ManifestEntry) -> None:
    """Compare one committed external reference against the current API."""
    ref = load_case(case)
    built = build_case(ref)
    result = built.workspace.xs(**built.xs_kwargs)
    np.testing.assert_allclose(
        result.dsdo,
        ref.dsdo,
        rtol=ref.tolerance["rtol"],
        atol=ref.tolerance["atol"],
        err_msg=f"[{ref.case_id} / {ref.reference_code}] dsdo disagrees",
    )
