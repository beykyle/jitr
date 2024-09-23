from .utils import *
from . import kinematics
from . import constants
from . import free_solutions

# read AME mass table into memory for fast lookup later
kinematics.init_AME_db()
