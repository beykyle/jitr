# Potential contract and conventions

The `jitr.xs` workspaces are backed by the external
[`lax`](https://github.com/beykyle/lax) solver package. This page defines
the contract between user-supplied potentials and the workspaces, the
shapes of every output, and the kinematics conventions used internally.

## The radial grid

`workspace.radial_grid()` returns the quadrature abscissae in **physical
fm**, on `[0, channel_radius_fm]`. The grid is **energy-independent** and
identical for all partial waves: sample each potential term once and reuse
it for every energy on the grid.

The grid is the Gauss–Legendre mesh of the underlying Lagrange basis
(`nbasis` points). It is numerically identical to the grid returned by the
pre-2.0 `radial_grid()`, so existing potential-sampling code carries over
unchanged.

## Accepted potential inputs

Each term builder (`workspace.central`, `workspace.coulomb`,
`workspace.spin_orbit`, `workspace.nonlocal_`) accepts, with
`N = nbasis`, `N_E` the number of grid energies, and
`N_b = lmax + 1` partial-wave blocks:

| Input | Interpretation |
|---|---|
| `(N,)` array | local, energy-independent |
| `(N_E, N)` array | local, energy-dependent (e.g. DOM, dispersive) |
| `(N, N)` array | non-local kernel `K(r_i, r_j)`, raw values — quadrature scaling is applied internally |
| `(N_E, N, N)` array | non-local, energy-dependent |
| leading `(N_b,)` axis on any of the above | ℓ-/block-dependent (e.g. ℓ-dependent Perey–Buck kernels, parity-dependent terms) |
| callable `f(r)`, `f(r, E)`, `f(r, r')`, `f(r, r', E)` | convenience path; evaluated on the mesh internally |

Dispatch is shape-driven. When shapes are ambiguous (`N_E == N` or
`N_b == N_E`), pass the explicit keyword flags `energy_dependent=` /
`l_dependent=`; the builders raise rather than guess.

Term builders return `Interaction` (or `InteractionPair` for spin-orbit)
objects that support `+`, so terms of mixed locality, energy dependence,
and ℓ dependence compose freely:

```python
V = (ws.central(U_central)        # (N_E, N): energy-dependent local
     + ws.nonlocal_(K_exchange)   # (N, N):   static non-local
     + ws.spin_orbit(U_so)        # (N,):     scaled by ⟨l·σ⟩ internally
     + ws.coulomb(U_coul))        # (N,):     static local
splus, sminus = ws.smatrix(V)
```

## Spin-orbit convention

Spin-orbit inputs are the radial form factor only. The workspace
multiplies by

- `⟨l·σ⟩ = l` for `j = l + 1/2`,
- `⟨l·σ⟩ = −(l + 1)` for `j = l − 1/2`,

i.e. the **2⟨l·s⟩** convention used by `jitr` since 1.x. If your form
factor is defined against ⟨l·s⟩, multiply it by 2 before passing it in.

## Output shapes and axis conventions

There is no scalar-energy mode: a scalar `Elab` is treated as a length-1
energy grid.

- **Partial-wave arrays are trailing-energy:** `Splus` is
  `(lmax + 1, N_E)`, `Sminus` is `(lmax, N_E)` (no `j = l − 1/2` wave for
  `l = 0`), transmission coefficients likewise.
- **Observables are leading-energy:** `dsdo`, `Ay`, `Q` are
  `(N_E, N_θ)`; `t`, `rxn` are `(N_E,)`.

## Energies, kinematics, and cost model

The workspace consumes a `ChannelKinematics` object whose fields may be
scalars or `(N_E,)` arrays — build it with `classical_kinematics`,
`semi_relativistic_kinematics`, or your own calculation. Semi-relativistic
(energy-dependent effective μ) kinematics are fully supported; the
workspace maps `(Ecm, k, μ, η)` faithfully onto the solver so that the
interior equation and the asymptotic matching both reproduce the requested
kinematics exactly.

The energy grid, `lmax`, channel radius, and `nbasis` are **compile-time**:
changing any of them means constructing a new workspace (seconds,
dominated by arbitrary-precision Coulomb boundary values for charged
channels). Changing the *potential* does not recompile — the jitted solve
path simply re-executes. The intended fitting/UQ workflow is: compile one
workspace per (reaction, energy grid), then evaluate per parameter sample.

All partial waves up to `lmax` are always computed for all energies (there
is no per-ℓ early exit, and the former `smatrix_abs_tol` /
`tmatrix_abs_tol` arguments are gone). Choose `lmax` for the **highest**
grid energy — `jitr.xs.elastic.suggest_lmax(reaction, Elab_max,
channel_radius_fm)` provides a grazing-ℓ heuristic.

## Differentiability

The full potential → observable pipeline is JAX-differentiable when the
workspace is constructed with `method="linear_solve"`. The default
spectral path (`eig`, used for complex potentials on CPU) is faster for
large energy grids but is **not** differentiable.
