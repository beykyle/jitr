[![Python package](https://github.com/beykyle/jitr/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/jitr/actions/workflows/python-package.yml)
# Just-In-Time R-matrix (JITR)
## Quick start
```
 pip install jitr
```

## Description
Solves the radial Bloch-Shrödinger equation in the continuum using the calculable R-Matrix method on a Lagrange-Legendre mesh, using just-in-time (JIT) compilation from [`numba`](https://numba.pydata.org/). The theory generally follows:
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219,
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,

with the primary difference being that this code uses the energy-scaled version of the Bloch-Shrödinger equation, with dimensionless domain, $s = kr$, where $r$ is the radial coordinate and $k$ is the channel wavenumber.

Capable of:
- non-local interactions
- coupled-channels


## Simple example


