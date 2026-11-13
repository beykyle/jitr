"""The regression reader checks its own inputs."""

import json
import shutil

import pytest

from tests.regression._readers import ROOT, load_case


def test_tolerance_without_a_column_is_rejected(tmp_path):
    """A JSON tolerance for an observable missing from the CSV would silently
    drop coverage, so the reader refuses it."""
    name = "frescox/reference/F13_d_ni58_elastic_noso_22MeV"
    csv, meta = tmp_path / "case.csv", tmp_path / "case.json"
    shutil.copy(ROOT / f"{name}.csv", csv)
    metadata = json.loads((ROOT / f"{name}.json").read_text())
    metadata["tolerances"]["T20"] = {"rtol": 0.0, "atol": 0.005}
    meta.write_text(json.dumps(metadata))
    case = {
        "case_id": metadata["case_id"],
        "reference_code": "frescox",
        "csv": str(csv),
        "json": str(meta),
    }
    with pytest.raises(ValueError, match="T20"):
        load_case(case)
