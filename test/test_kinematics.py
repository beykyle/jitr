import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from jitr.utils.kinematics import cm_to_lab_frame, lab_to_cm_frame


@pytest.fixture
def example_parameters():
    return {
        "ma": 1.0,
        "mb": 1.0,
        "mc": 1.0,
        "md": 1.0,
        "E": 10.0,
        "Q": 1.0,
        "angles_cm_deg": np.linspace(0, 180, 10),
    }


def test_cm_to_lab(example_parameters):
    """Test cm_to_lab_frame conversion."""
    params = example_parameters
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
    assert_almost_equal(params["angles_cm_deg"], angles_cm_deg_recovered, decimal=6)


def test_lab_to_cm(example_parameters):
    """Test lab_to_cm_frame conversion."""
    params = example_parameters
    angles_lab_deg = np.linspace(0, 90, 5)  # Example angles in lab frame
    angles_cm_deg = lab_to_cm_frame(
        angles_lab_deg,
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
    assert_almost_equal(angles_lab_deg, angles_lab_deg_recovered, decimal=6)
