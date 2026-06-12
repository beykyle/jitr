from __future__ import annotations

import numpy as np

from tests.conftest import requires_lax
from tests.regression._builders import build_case
from tests.regression._readers import ManifestEntry, load_case

pytestmark = requires_lax


def test_regression(case: ManifestEntry) -> None:
    """Compare one committed external reference against the current API."""
    ref = load_case(case)
    built = build_case(ref)
    result = built.workspace.xs(**built.xs_kwargs)
    # single-energy case: drop the (N_E = 1) leading axis of the new engine
    np.testing.assert_allclose(
        np.asarray(result.dsdo)[0],
        ref.dsdo,
        rtol=ref.tolerance["rtol"],
        atol=ref.tolerance["atol"],
        err_msg=f"[{ref.case_id} / {ref.reference_code}] dsdo disagrees",
    )
