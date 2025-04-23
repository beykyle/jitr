from .utils import *
from . import mass
from . import kinematics
from . import constants
from . import free_solutions
from . import coupling

# read mass table into memory for fast lookup later
mass.init_mass_db()
