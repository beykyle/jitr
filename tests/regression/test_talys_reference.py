from __future__ import annotations

from tests.regression._readers import load_case
from tests.regression.talys.tools.parse_talys import find_table


def test_parse_talys_reads_yandf_angular_distribution_direct_column() -> None:
    lines = [
        "# header:",
        "##     Angle           xs           Direct        Compound",
        "##     [deg]         [mb/sr]        [mb/sr]        [mb/sr]",
        "   0.000000E+00   6.327864E+03   6.327860E+03   3.893764E-03",
        "   2.000000E+00   6.263124E+03   6.263120E+03   3.870623E-03",
    ]

    columns, rows = find_table(lines, "direct")

    assert columns == ["angle", "xs", "direct", "compound"]
    assert rows == [(0.0, 6327.86), (2.0, 6263.12)]


def test_talys_jlmb_reference_loads_cleanly() -> None:
    ref = load_case(
        {
            "case_id": "T2_jlmb_elastic",
            "reference_code": "talys",
            "csv": "talys/reference/T2_jlmb_elastic.csv",
            "json": "talys/reference/T2_jlmb_elastic.json",
        }
    )

    assert ref.case_id == "T2_jlmb_elastic"
    assert ref.reference_code == "talys"
    assert ref.observable_type == "elastic"
    assert ref.theta_cm_rad.shape == ref.dsdo.shape
    assert ref.theta_cm_rad.size == 91  # full 0-180 deg; spin-orbit included
    assert ref.metadata["optical_potential"]["kind"] == "jlm"
    assert ref.metadata["optical_potential"]["variant"] == "jlmb"
