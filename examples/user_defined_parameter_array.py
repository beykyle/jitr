import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from jitr import (
    ProjectileTargetSystem,
    InteractionMatrix,
    RMatrixSolver,
    woods_saxon_potential,
    woods_saxon_prime
)

# target (A,Z)
Ca48 = (28, 20)
mass_Ca48 = 44657.26581995028  # MeV/c^2

# projectile (A,z)
proton = (1, 1)
mass_proton = 938.271653086152  # MeV/c^2

# for speed, we can use numpy structured types to
# allocate sets of parameters contiguously in memory
my_model_param_dtype = [
    ("V", np.float64),
    ("W", np.float64),
    ("Vso", np.float64),
    ("Wso", np.float64),
    ("R", np.float64),
    ("a", np.float64),
    ("l", np.int32)
    ("j", np.int32)
]

# only a subset of the actual model params are of interest
# in a statistical calibration. In this case, we ignore l and j
my_model_statistical_params = my_model_param_dtype[:-2]

my_param_names = [param[0] for param in my_model_statistical_params]

N_params = len(my_model_statistical_params)

@njit
def my_model_interaction(r, params: np.array):
    assert params.dtype == my_model_param_dtype
    assert params.shape == (1,)

    l, j = params[0][["l", "j"]]
    central = woods_saxon_potential(r, params[0][["V", "W", "R", "a"]])
    spin_orb = woods_saxon_prime(r, params[0][["Vso", "Wso", "R", "a"]])

    return central + (j * (j + 1) - l * (l + 1) - 0.5 * (0.5 + 1)) * spin_orb


# let's assume out model is described by a multivariate normal
# distribution of parameters
def sample_params(mu, cov, N):
    return np.random.multivariate_normal(mu, cov, N)


# lets assume some mean and covariance info for our example model
# leave off l and j for the sake of sampling

mu = np.array([40.2, 10.0, 5.1, 0.3, 5.8, 3.1])
cov = np.zeros((N_params, N_params))
cov += np.diag([10.1, 6.8, 3.1, 2.9, 1.1, 0.5])**2

# let's add some correlations between depth terms
cov[1,0] = cov[0,1] = 0.4
cov[1,2] = cov[2,1] = -0.1
cov[1,3] = cov[3,1] = -0.6
cov[0,2] = cov[2,0] = -0.6
cov[0,3] = cov[3,0] = -0.1

# and some depth-radius correlations
cov[0,4] = cov[4,0] = -1.2
cov[1,4] = cov[4,1] = -0.3
cov[2,4] = cov[4,2] = -0.8

samples = sample_params(mu, cov, 1000)

