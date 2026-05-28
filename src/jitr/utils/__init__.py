"""Utility functions, constants, kinematics, and packaged-data helpers."""

from . import constants as constants
from . import density as density
from . import free_solutions as free_solutions
from . import kinematics as kinematics
from . import mass as mass
from . import poly as poly
from .utils import block as block
from .utils import complex_det as complex_det
from .utils import delta as delta
from .utils import eval_scaled_interaction as eval_scaled_interaction
from .utils import eval_scaled_nonlocal_interaction as eval_scaled_nonlocal_interaction
from .utils import interaction_range as interaction_range
from .utils import schrodinger_eqn_ivp_order1 as schrodinger_eqn_ivp_order1
from .utils import second_derivative_op as second_derivative_op
from .utils import smatrix as smatrix
from .utils import suggested_basis_size as suggested_basis_size
from .utils import (
    suggested_dimensionless_channel_radius as suggested_dimensionless_channel_radius,
)

# read mass table into memory for fast lookup later
density.init_density_db()
mass.init_mass_db()

__all__ = [
    "block",
    "complex_det",
    "constants",
    "delta",
    "density",
    "eval_scaled_interaction",
    "eval_scaled_nonlocal_interaction",
    "free_solutions",
    "interaction_range",
    "kinematics",
    "mass",
    "poly",
    "schrodinger_eqn_ivp_order1",
    "second_derivative_op",
    "smatrix",
    "suggested_basis_size",
    "suggested_dimensionless_channel_radius",
]
