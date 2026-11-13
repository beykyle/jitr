# Observables and workspaces

A workspace is built once for a reaction, a radial mesh and a grid of
energies, and then evaluates observables for any number of potentials. The
`coupled` workspace solves general coupled-channels systems by $J^\pi$ block;
`elastic` and `quasielastic_pn` build on it, and `amplitudes` turns collision
matrices into cross sections and polarization observables.

```{eval-rst}
.. automodule:: jitr.xs.coupled
   :members:
```

```{eval-rst}
.. automodule:: jitr.xs.elastic
   :members:
```

```{eval-rst}
.. automodule:: jitr.xs.quasielastic_pn
   :members:
```

```{eval-rst}
.. automodule:: jitr.xs.amplitudes
   :members:
```
