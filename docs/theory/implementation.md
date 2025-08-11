# Implementation Details

## Core Architecture
jitr implements the R-matrix method using a combination of:
1. **Numpy** for vectorized operations
2. **Numba** for JIT compilation of performance-critical code
3. **Scipy** for special functions and numerical integration

## Key Components
### Mesh Generation
The Lagrange-Legendre mesh is generated using:

```python
from jitr.mesh import LegendreMesh

mesh = LegendreMesh(N, a)
```

Where:
- `N`: Number of mesh points
- `a`: Channel radius

### Wavefunction Solution
The radial Schrödinger equation is solved using:

```python
from jitr.solver import Solver

solver = Solver(V, L, mu, E, mesh)
wavefunction = solver.solve()
```

Parameters:
- `V`: Potential function
- `L`: Orbital angular momentum
- `mu`: Reduced mass
- `E`: Center-of-mass energy

### Cross Section Calculation
Cross sections are computed using:

```python
from jitr.xs import elastic

sigma = elastic.cross_section(wavefunction, k, Z1, Z2)
```

## Performance Optimizations
- Precomputation of basis functions
- Vectorized matrix operations
- JIT compilation of inner loops
- Caching of intermediate results