"""Utility functions, constants, kinematics, and packaged-data helpers."""

from . import constants as constants
from . import density as density
from . import free_solutions as free_solutions
from . import kinematics as kinematics
from . import mass as mass
from . import poly as poly
from .utils import delta as delta
from .utils import interaction_range as interaction_range
from .utils import radial_equation as radial_equation
from .utils import suggest_lmax as suggest_lmax
from .utils import suggested_basis_size as suggested_basis_size
from .utils import (
    suggested_dimensionless_channel_radius as suggested_dimensionless_channel_radius,
)

# read mass table into memory for fast lookup later
density.init_density_db()
mass.init_mass_db()

__all__ = [
    "constants",
    "delta",
    "density",
    "free_solutions",
    "interaction_range",
    "kinematics",
    "mass",
    "poly",
    "radial_equation",
    "suggest_lmax",
    "suggested_basis_size",
    "suggested_dimensionless_channel_radius",
]
