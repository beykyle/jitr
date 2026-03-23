import numpy as np
import pytest
from jitr.utils.kinematics import cm_to_lab_frame, lab_to_cm_frame
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    "params",
    [
        {
            "ma": 1.0,
            "mb": 1.0,
            "mc": 1.0,
            "md": 1.0,
            "E": 10.0,
            "Q": 1.0,
            "angles_cm_deg": np.linspace(0, 180, 10),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_cm_deg": np.linspace(0, 180, 10),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_cm_deg": np.linspace(0, 180, 9),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_cm_deg": np.linspace(0, 180, 11),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_cm_deg": np.linspace(0, 180, 3),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_cm_deg": np.linspace(0, 180, 36),
        },
    ],
)
def test_cm_to_lab(params):
    """Test cm_to_lab_frame conversion."""
    angles_lab_deg = cm_to_lab_frame(
        params["angles_cm_deg"],
        params["ma"],
        params["mb"],
        params["mc"],
        params["md"],
        params["E"],
        params["Q"],
    )

    # Perform a round-trip conversion
    angles_cm_deg_recovered = lab_to_cm_frame(
        angles_lab_deg,
        params["ma"],
        params["mb"],
        params["mc"],
        params["md"],
        params["E"],
        params["Q"],
    )
    assert_allclose(params["angles_cm_deg"], angles_cm_deg_recovered, atol=1e-6)


@pytest.mark.parametrize(
    "params",
    [
        {
            "ma": 1.0,
            "mb": 1.0,
            "mc": 1.0,
            "md": 1.0,
            "E": 10.0,
            "Q": 1.0,
            "angles_lab_deg": np.linspace(0, 180, 10),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_lab_deg": np.linspace(0, 180, 10),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_lab_deg": np.linspace(0, 180, 9),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_lab_deg": np.linspace(4, 17, 11),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_lab_deg": np.linspace(3, 90, 3),
        },
        {
            "ma": 2.0,
            "mb": 2.0,
            "mc": 2.0,
            "md": 2.0,
            "E": 20.0,
            "Q": 2.0,
            "angles_lab_deg": np.linspace(0, 180, 36),
        },
    ],
)
def test_lab_to_cm(params):
    """Test lab_to_cm_frame conversion."""
    angles_cm_deg = lab_to_cm_frame(
        params["angles_lab_deg"],
        params["ma"],
        params["mb"],
        params["mc"],
        params["md"],
        params["E"],
        params["Q"],
    )

    # Perform a round-trip conversion
    angles_lab_deg_recovered = cm_to_lab_frame(
        angles_cm_deg,
        params["ma"],
        params["mb"],
        params["mc"],
        params["md"],
        params["E"],
        params["Q"],
    )
    assert_allclose(params["angles_lab_deg"], angles_lab_deg_recovered, atol=1e-6)
