from .utils import *
from . import mass
from . import kinematics
from . import constants
from . import free_solutions
from . import angular_momentum

# read mass table into memory for fast lookup later
mass.init_mass_db()
