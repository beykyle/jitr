from .utils import *
from . import kinematics
from . import constants
from . import free_solutions
from . import angular_momentum

# read AME mass table into memory for fast lookup later
kinematics.init_AME_db()
