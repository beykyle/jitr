# Legacy notebooks (pre-2.0 solver engine)

The notebooks in this directory target the internal `jitr.rmatrix` /
`jitr.quadrature` solver engine, which was removed when `jitr.xs` was
rebuilt on the external [`lax`](https://github.com/beykyle/lax) package.
They **will not run** against current `jitr` and are kept for reference
only (the last working release is the final 1.x tag).

- `comparison_to_Runge_Kutta.ipynb` — benchmarked the removed
  `jitr.rmatrix.Solver` against a Runge–Kutta integrator. A lax-native
  equivalent belongs upstream in lax's own benchmarks.
- `how_to_define_your_interaction.ipynb` — documented the old
  solver-level interaction API. Superseded by the workspace term builders;
  see `docs/potential-contract.md` and `quickstart.ipynb`.
- `integration.ipynb` — demonstrated `jitr.quadrature.Kernel`. The
  surviving pure-NumPy transforms live in `jitr.utils.transforms`; mesh
  construction and integration live in `lax`.
- `test_coupled_single_dwba.ipynb` (+ `solver.pkl`) — coupled-channels
  demo. Coupled channels were an explicit non-goal of the lax rewrite and
  have no replacement yet; migrating this requires upstream lax support.

If you need one of these workflows on the new engine, open an issue.
