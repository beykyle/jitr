import numpy as np
import jitr
from numba import njit


@njit
def interaction(r, *args):
    (V0, W0, R0, a0, zz, r_c) = args
    nuclear = jitr.woods_saxon_potential(r, V0, W0, R0, a0)
    coulomb = jitr.coulomb_charged_sphere(r, zz, r_c)
    return nuclear + coulomb


nodes_within_radius = 5
a = 2 * np.pi * nodes_within_radius

E_lab = 35  # MeV

# target (A,Z)
Ca48 = (28, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2

sys = jitr.ProjectileTargetSystem(
    channel_radii=np.array([a]),
    l=np.array([0]),
    mass_target=mass_Ca48,
    mass_projectile=mass_proton,
    Ztarget=Ca48[1],
    Zproj=proton[1],
)

# initialize solver
solver = jitr.RMatrixSolver(nbasis=40)

channels = sys.build_channels_kinematics(E_lab)

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
R, S, uext_boundary = solver.solve(interaction_matrix, channels)

# get phase shift in degrees
delta, atten = jitr.delta(S[0, 0])
print(f"phase shift: {delta:1.3f} + i {atten:1.3f} [degrees]")
