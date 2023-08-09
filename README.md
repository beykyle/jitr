[![Python package](https://github.com/beykyle/lagrange_rmatrix/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/lagrange_rmatrix/actions/workflows/python-package.yml)
# Lagrange R-Matrix
## Quick start
```
 pip install lagrange-rmatrix
```

## Description
Solves the radial Bloch-Shrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh, following:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses a dimensionless domain for the Shrödinger-Bloch equation, $s = kr$, where $r$ is the radial coordinate and $k$ is the channel wavenumber. 

Capable of:
- non-local interactions
- coupled-channels


## Simple example

The following is an example of solving a simple local Woods-Saxon + Coulomb interaction problem in the elastic channel
```python
  
  from lagrange_rmatrix import (
    ProjectileTargetSystem,       # defines channel-independent data for the system
    RadialSEChannel,              # channel description
    LagrangeRMatrix,              # solver
    woods_saxon_potential,        # short-range nuclear interaction
    coulomb_potential,            # long-range EM interaction
    delta,                        # function to calculate phase shift 
  )

  # Woods-Saxon potential parameters
  V0 = 60  # real potential strength
  W0 = 20  # imag potential strength
  R0 = 4  # Woods-Saxon potential radius
  a0 = 0.5  # Woods-Saxon potential diffuseness
  params = (V0, W0, R0, a0)

  # We want to have 5 nodes of the wavefunction within the channel radius -
  # this should allow for the wavefunction to reach its asymptotic phase
  nodes_within_radius = 5

  # set up the system
  sys = ProjectileTargetSystem(
    incident_energy=20, # [MeV]
    reduced_mass=939,   # [MeV/c^2]
    Ztarget=40,         # charged target and projectile
    Zproj=1
    channel_radius=nodes_within_radius * (2 * np.pi),
  )

  # set up the Bloch-SE equation in the elastic channel
  se = RadialSEChannel(
    l=1,             # p-wave scattering
    system=sys,
    interaction=lambda r: woods_saxon_potential(r, params),
    coulomb_interaction=lambda zz, r: np.vectorize(coulomb_potential)(zz, r, R0)
  )

  # set up and run solver
  nbasis = 30 # number of Lagrange-Legendre functions in the basis
  solver = LagrangeRMatrix(nbasis, sys, se)
  R_l, S_l, _ = solver_lm.solve()   # Get the R and S-Matrices
  delta_l, atten_l = delta(S_lm)    # get the phase shift
  print(f"phase shift: {delta_lm:.4f} + i {atten_lm:.4f} [deg]")
```
which should produce the output:
```
phase shift: 38.8723 + i 53.7059 [deg]
```

## Example result for a coupled-channel toy problem 
Here we show a simple toy coupled-channels problem with 3 0 $^+$ levels, and flux incident on the $n=0$ (elastic) channel. For more information, see [`examples/coupled`](https://github.com/beykyle/lagrange_rmatrix/blob/main/examples/coupled.py):

![3-channel problem](https://github.com/beykyle/lagrange_rmatrix/blob/main/assets/cc.png)
