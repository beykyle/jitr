import numpy as np
import jitr
from numba import njit


@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    return jitr.woods_saxon_potential(r, V0, W0, R0, a0) + jitr.coulomb_charged_sphere(
        r, zz, r_c
    )


energy_com = 26  # MeV
nodes_within_radius = 5

# initialize system and description of the channel (elastic) under consideration
sys = jitr.ProjectileTargetSystem(
    np.array([939.0]),
    np.array([nodes_within_radius * (2 * np.pi)]),
    l=np.array([0]),
    Ztarget=40,
    Zproj=1,
    nchannels=1,
)
ch = sys.build_channels(energy_com)

# initialize solver for single channel problem with 40 basis functions
solver = jitr.LagrangeRMatrixSolver(40, 1, sys)

# use same interaction for all channels
interaction_matrix = jitr.InteractionMatrix(1)
interaction_matrix.set_local_interaction(interaction)

# Woods-Saxon and Coulomb potential parameters
V0 = 60  # real potential strength
W0 = 20  # imag potential strength
R0 = 4  # Woods-Saxon potential radius
a0 = 0.5  # Woods-Saxon potential diffuseness
RC = R0  # Coulomb cutoff
params = (V0, W0, R0, a0, sys.Zproj * sys.Ztarget, RC)

# set params
interaction_matrix.local_args[0, 0] = params

# run solver
R, S, uext_boundary = solver.solve(interaction_matrix, ch, energy_com)

# get phase shift in degrees
delta, atten = jitr.delta(S[0, 0])
print(f"phase shift: {delta:1.3f} + i {atten:1.3f} [degrees]")
