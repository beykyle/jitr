# Migrating to jitr v4

jitr v4 replaces the numba engine with a JAX R-matrix core organized by
$J^\pi$ blocks. Workspaces take any number of energies at once, results are
differentiable with `jax.grad`, and the same code runs on GPUs. This page maps
the v3 API onto v4.

## Conventions that changed everywhere

- **Energy first, never squeezed.** Every result has a leading energy axis,
  also for a single energy: `xs.dsdo` has shape `(NE, Nθ)`, integrated cross
  sections `(NE,)`. A single-energy workspace has `NE = 1`, so use
  `xs.dsdo[0]`.
- **One workspace for an energy grid.** Pass arrays to
  `reaction.kinematics(np.array([...]))`. The radial mesh `ws.r` is the same
  for every energy, so choose the channel radius and `nbasis` for the largest
  `k`. `jitr.utils.suggest_lmax(kin.k, channel_radius_fm)` replaces the old
  early-exit tolerances (`smatrix_abs_tol`, `tmatrix_abs_tol`), which are gone.
- **Potentials** are arrays on `ws.r`: local `(N,)` or energy-dependent
  `(NE, N)`; nonlocal `(N, N)` or `(NE, N, N)` evaluated on
  `np.meshgrid(ws.r, ws.r, indexing="ij")`. An optional leading `l` axis,
  `(L, NE, N[, N])`, makes a form `l`-dependent.
- **Compilation.** The first call of a workspace compiles (about half a
  second). Build the workspace once, in its own notebook cell, and call it
  repeatedly with new potentials. Workspaces of the same shape reuse the
  compiled code.
- **Precision.** `import jitr` enables 64-bit floats in JAX for the whole
  process.

## Elastic scattering

| v3 | v4 |
|---|---|
| `Solver(N)` | gone; pass `nbasis` to the workspace |
| `elastic.DifferentialWorkspace.build_from_system(reaction, kin, R, solver, lmax, angles)` | `elastic.Workspace(reaction, kin, R, lmax, nbasis, angles=angles)` |
| `elastic.IntegralWorkspace(...)` | the same `Workspace`, without `angles` |
| `ws.radial_grid()` | `ws.r` |
| `ws.xs(central_potential=U, spin_orbit_potential=Uso, coulomb_potential=Vc)` | `ws.xs(Potential(central=U, spin_orbit=Uso, coulomb=Vc))` |
| `xs.dsdo`, `xs.Ay`, `xs.Q` `(Nθ,)`; `xs.t`, `xs.rxn` scalars | the same fields, `(NE, Nθ)` and `(NE,)`; `xs.T` holds the tensor analyzing powers `T_kq` |
| `ws.rutherford` `(Nθ,)` | `(NE, Nθ)`, or `None` for neutral projectiles |
| `splus, sminus = ws.smatrix(...)` | `S = ws.smatrix(Potential(...))`, `(NE, NB, nch, nch)` over J^π blocks; `ws.to_lj(S)` gives `[energy, l, j]` with `j = l + 1/2, l - 1/2` |
| `ws.transmission_coefficients(...)` | `1 - abs(ws.to_lj(S))**2` |

`Potential` is `jitr.xs.coupled.Potential`, with fields `central`,
`spin_orbit`, `coulomb`, `nonlocal_central` and `nonlocal_spin_orbit`, each
optional. The projectile spin follows from the reaction: 1/2 for n, p, t and
³He, 1 for d (with tensor analyzing powers), 0 otherwise (no spin-orbit). The
spin-orbit potential multiplies `2 l·s` (`l·σ` for nucleons). For experimental
data at different angles at each energy,
`ws.xs_from_smatrix(S, [angles_0, angles_1, ...])` returns one result per
energy without recompiling the solve.

## Quasi-elastic (p,n)

`jitr.xs.lane_pn` is merged into `jitr.xs.quasielastic_pn`:

| v3 | v4 |
|---|---|
| `quasielastic_pn.Workspace(reaction, ke, kx, solver, angles, lmax, R, tmatrix_abs_tol)` | `quasielastic_pn.Workspace(reaction, ke, kx, R, lmax, nbasis, angles=angles)` |
| `ws.xs(U_p_coulomb, U_p_central, U_p_spin_orbit, U_n_central, U_n_spin_orbit, U1_central, U1_spin_orbit)` | `ws.dwba(Up, Un, U1).dsdo` with `Up = Potential(central, spin_orbit, coulomb)`, `Un = Potential(central, spin_orbit)`, `U1 = Potential(central, spin_orbit)` or `None` |
| `ws.observables(...)` (DWBA), `lane_pn.Workspace(...).observables(...)` (CC) | `ws.dwba(...)`, `ws.cc(...)`: `QuasielasticPnXS(dsdo, Ay, Q, rxn_pn, f)` |
| `lane_pn` `integrated_xs` | `.rxn_pn` |
| `ws.tmatrix(...)` → `(Tpn, Sn, Sp)` indexed `[l, j]` | `ws.tmatrix(Up, Un, U1)` → `(T, Sp, Sn)` indexed `[energy, block]` |

Each missing term of `U1` defaults to the isovector difference
`-(U_n - U_p)·sqrt|N-Z|/(N-Z-1)`; pass zeros to switch a term off.

## Coupled channels and the solver

| v3 | v4 |
|---|---|
| `ProjectileTargetSystem`, `get_partial_wave_channels`, `Channels`, `Asymptotics`, coupling callables | `jitr.xs.coupled.Workspace(R, nbasis, levels, lmax, k2, mu, eta)` with `levels = [Level(I, parity, s), ...]` and `k2, mu, eta = level_kinematics(kin, Ex)` |
| `solver.solve(channels[l], asymptotics[l], local_potential=V)` | `ws.smatrix(local=[(C, v), ...], nonlocal_=[...])` over all J^π blocks and energies; `C` is a constant coupling matrix (`np.eye`, `spin_orbit(ws.blocks, levels)`, `rotor(ws.blocks, levels, λ)`, ...) |
| `Wavefunctions(...).uint()` | `S, u, u_a = ws.wavefunctions(local=..., weights=...)`, values on `ws.r` |
| `jitr.quadrature.Kernel`, `jitr.rmatrix.Solver` | `jitr.rmatrix`: `mesh(n, a)`, `solve(...)` (one block), `interaction`, `integrate`, `integrate2`, `fourier_bessel`, `fourier_bessel2`, `laguerre` |
| `utils.schrodinger_eqn_ivp_order1`, `utils.smatrix`, `make_channel_data` | `utils.radial_equation(r, y, l, k, mu, potential)` with `scipy.integrate.solve_ivp`, matched with `free_solutions.coulomb_hankel_table` (see `examples/local.py`) |

The interior wavefunctions are the symmetrized, flux-normalized
`ũ_c = u_c/√μ_c`, which outside the potential are
`k_c^{-1/2} (i/2)[w_c H⁻_c − (S w)_c H⁺_c]`.

## Results that change

- **Nonlocal potentials.** v3 divided nonlocal interactions by the channel
  wavenumber once too often. v4 reproduces the exact Yamaguchi phase shifts
  of Descouvemont, Comput. Phys. Commun. 200 (2016) 199, Example 5.
- **`yamaguchi_swave_delta`** had the sign of the scattering length wrong.
- **No early exit.** Every partial wave up to `lmax` is solved, so results can
  differ from v3 at the level of the old tolerances (`1e-6` by default).
