# R-Matrix Theory

## Overview
The R-matrix method is a powerful technique for solving nuclear reaction problems. It provides a framework for describing nuclear reactions by dividing space into two regions:

1. **Internal region** (r ≤ a): Where nuclear forces are strong and the wave function is expanded in a basis set
2. **External region** (r > a): Where only Coulomb and centrifugal forces act, and the wave function is known analytically

## Key Equations
The Schrödinger equation for a nuclear system can be solved using the R-matrix approach:

$$
R_{cc'} = \frac{\hbar^2}{2\mu a^2} \sum_{\lambda} \frac{\gamma_{\lambda c} \gamma_{\lambda c'}}{E_\lambda - E}
$$

Where:
- $R_{cc'}$ is the R-matrix
- $\gamma_{\lambda c}$ are the reduced width amplitudes
- $E_\lambda$ are the R-matrix pole energies
- $E$ is the center-of-mass energy

## Implementation in jitr
jitr implements the calculable R-matrix method using a Lagrange-Legendre mesh, following the approach described in:

- Descouvemont P. and Baye D. (2010). The R-matrix theory. Rep. Prog. Phys. 73 036301
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107