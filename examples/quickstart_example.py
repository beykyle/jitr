import numpy as np
from jitr import reactions, rmatrix
from jitr.optical_potentials.potential_forms import (
    woods_saxon_potential,
    coulomb_charged_sphere,
)
from jitr.utils import kinematics, delta, mass, constants


# define interaction
def interaction(r, V0, W0, R0, a0, Zz):
    nuclear = -woods_saxon_potential(r, V0, W0, R0, a0)
    coulomb = coulomb_charged_sphere(r, Zz, R0)
    return nuclear + coulomb


# define system
Elab = 35  # MeV
Ca48 = (48, 20)
proton = (1, 1)

sys = reactions.ProjectileTargetSystem(
    channel_radius=5 * np.pi,
    lmax=10,
    mass_target=mass.mass(*Ca48)[0],
    mass_projectile=constants.MASS_P,
    Ztarget=Ca48[1],
    Zproj=proton[1],
)
Elab, Ecm, mu, k, eta = kinematics.classical_kinematics(
    sys.mass_target,
    sys.mass_projectile,
    Elab,
    sys.Ztarget * sys.Zproj,
)
channels, asymptotics = sys.get_partial_wave_channels(Elab, Ecm, mu, k, eta)

# set up solver
solver = rmatrix.Solver(nbasis=40)

# solve for a set of parameters
l = 0
params = (42.0, 18.1, 4.8, 0.7, sys.Zproj * sys.Ztarget)
R, S, uext_boundary = solver.solve(
    channels[l], asymptotics[l], interaction, local_args=params
)

# get phase shift in degrees
phase_shift, phase_attenuation = delta(S[0, 0])
print(f"phase shift: {phase_shift:1.3f} + i ({phase_attenuation:1.3f}) [degrees]")
