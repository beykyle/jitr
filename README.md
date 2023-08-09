# Lagrange R-Matrix
Solves the Shrödinger equation in the continuum useing the calculable R-Matrix method on a Lagrange-Legendre mesh, following 
- Descouvemont, P. (2016). An R-matrix package for coupled-channel problems in nuclear physics. Computer physics communications, 200, 199-219.
- Baye, D. (2015). The Lagrange-mesh method. Physics reports, 565, 1-107.

with the primary difference being that this code uses a dimensionless domain for the Shrödinger-Bloch equation, $s = kr$, where $r$ is the radial coordinate and $k$ is the wavenumber. 

Capable of:
- non-local interactions
- coupled-channels calculations
