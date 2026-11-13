from __future__ import annotations

import numpy as np

from jitr.optical_potentials import kduq
from jitr.xs.coupled import Potential
from tests.regression._builders import build_case
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


def test_jlmb_spin_orbit_matches_kduq_sign() -> None:
    """The JLMB spin-orbit form has the sign and l.sigma normalization of a
    Thomas term: A_y follows KD03 for n + 120Sn at 10 MeV, and the form is
    negative at the nuclear surface like KD03's spin-orbit."""
    ref = load_case(
        {
            "case_id": "T2_jlmb_elastic",
            "reference_code": "talys",
            "csv": "talys/reference/T2_jlmb_elastic.csv",
            "json": "talys/reference/T2_jlmb_elastic.json",
        }
    )
    built = build_case(ref)
    ws = built.workspace
    jlmb = built.xs_kwargs["potential"]

    r = np.asarray(ws.r)
    central, spin_orbit, _ = kduq.KDUQ((1, 0)).evaluate(
        r, ws.reaction, ws.kinematics, *kduq.get_kd03((1, 0))
    )
    central = np.asarray(central).reshape(-1, r.size)[0]
    spin_orbit = np.asarray(spin_orbit).reshape(-1, r.size)[0]

    surface = np.argmin(np.abs(r - 1.2 * 120 ** (1 / 3)))
    assert spin_orbit.real[surface] < 0
    assert jlmb.spin_orbit.real[surface] < 0

    Ay_kd = np.asarray(
        ws.xs(potential=Potential(central=central, spin_orbit=spin_orbit)).Ay[0]
    )
    Ay_jlmb = np.asarray(ws.xs(potential=jlmb).Ay[0])
    assert np.corrcoef(Ay_jlmb, Ay_kd)[0, 1] > 0.8
