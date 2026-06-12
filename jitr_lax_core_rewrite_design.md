# jitr core rewrite: `lax` as the solver engine

**Status:** draft for implementation handoff
**Date:** 2026-06-11 (lax pin updated 2026-06-12)
**Supersedes:** `jitr_lax_core_rewrite_design.md` (v1). This revision incorporates the lax v1.5 symmetry-block batch axis and the per-energy `mass_factor_grid` (both now shipped in lax), promotes energy vectorization and ℓ-block vectorization to first-class goals, adds non-local kernels to the `jitr.xs` public API, and expands the numba-removal decision from "keep the observable kernels" to full removal.
**Canonical lax reference:** lax `DESIGN.md` **v1.6** (in-tree at the pinned commit). Section map for the F/C labels used below: F1 `matrix_element` → §13.4; F2 `wavefunction_grid`/`wavefunction_direct_grid` → §11.2–11.3; F3 block-batched transforms → §13 (v1.6 note) + §15.5; C4 μ(E) regime rule → Appendix C.10; C8 `eig` non-differentiability → Appendix C.11 (+ §17.2/§17.5); C7 cross-engine normalization → Appendix C.12; batched (p,n) DWBA example → §16.10. (The implementation-handoff spec `lax_design_v0.1.5.1.md` — the source of this F/C numbering — remains the historical decision record.)
**lax pin:** branch `v0.1.5.1`, commit `2c4672038ebd3ab586d636bd10f7ab349d90d81d` ("update design to include ref case and fix toc rendering"; implementation landed in its parent `54df40c`, "add DWBA reference test"). At this commit the entire upstream scope (§2.3) is implemented and green — F1 `matrix_element`, F2 `wavefunction_grid`/`wavefunction_direct_grid` with baked sources, F3 block-batched transforms, the C4 spectrum-kernel μ(E) fix + guards — `DESIGN.md` v1.6 is in-tree with every section anchor cited below (incl. Appendix C.12), and the cross-engine (p,n) DWBA anchor is **active** in lax's suite (`tests/acceptance/`, generated from this repo's `quasielastic_pn` engine on the ⁴⁸Ca(p,n) case with classical kinematics; S-matrices agree to ~8e-13, T-matrices to rtol 1e-8 after the closed-form C7 conversion). Build jitr Phases 2+ against this commit or later.

---

## 1. Goals

- **G1 — Non-local kernels in `jitr.xs`.** Expose non-local (energy- and angular-momentum-dependent) interaction kernels in the elastic workspaces for the first time, backed by `lax`'s verified/tested non-local solver. The current `xs` workspaces accept only local potential arrays.
- **G2 — Energy-vectorized `jitr.xs.*` workspaces.** Every workspace accepts an array of energies at construction and returns observables (S-matrices, cross sections, transmission coefficients, T-matrix elements) as vectors over that grid.
- **G3 — Symmetry-block vectorization.** The ℓ-dependent partial-wave set is compile-time static and carried as a batch axis on the `Interaction` and every observable, via `lax.compile(blocks=…)` (lax DESIGN.md §15.5).
- **G4 — Remove numba from jitr entirely**, replacing it with JAX (or plain NumPy where compilation buys nothing).

**Non-goals (unchanged from v1):** coupled-channel workspaces (`CoupledElasticWorkspace`) and coupled Descouvemont example migration; transfer-reaction DWBA (`DWBAWorkspace` from the separate design doc); off-grid energy interpolation (explicitly out of scope in lax §12 — the energy grid is the contract).

---

## 2. Current state of both packages

### 2.1 jitr today

The `jitr.xs` public surface this rewrite must preserve in spirit (not signature):

| Module | Class / function | Role |
|---|---|---|
| `xs.elastic` | `IntegralWorkspace` | per-(reaction, kinematics, lmax): `smatrix()`, `xs()`, `transmission_coefficients()` from local potential arrays |
| `xs.elastic` | `DifferentialWorkspace` | wraps an `IntegralWorkspace` + angle grid: `xs()` → `ElasticXS` (dσ/dΩ, A_y, Q, σ_t, σ_rxn), Rutherford ratio, Coulomb amplitude |
| `xs.elastic` | `integral_elastic_xs`, `differential_elastic_xs` | `@njit` observable kernels (S-matrix → observables) |
| `xs.quasielastic_pn` | `Workspace` | (p,n) quasi-elastic DWBA: `tmatrix()` → (Tpn, Sn, Sp), `xs()` |

Internals slated for retirement: `jitr.rmatrix` (incl. `rmatrix/core.py`, 4 `@njit` kernels), `jitr.quadrature`, `jitr.reactions.channel_on_grid` (numba `jitclass`), and the wavefunction-reconstruction path in `jitr.reactions.wavefunction` that depends on them.

Key properties of the current engine that the rewrite **changes**:

1. **One energy per workspace.** `ChannelKinematics` is scalar; an energy scan is a Python loop over workspaces.
2. **Per-(l, j) sequential solves** with an early-exit convergence break on `smatrix_abs_tol` / `tmatrix_abs_tol`.
3. **Dimensionless mesh.** The quadrature lives in `s = k·r`, so `radial_grid()` *depends on energy* through `a = k·R`. Potentials are re-sampled per energy.
4. **Local-only potential contract:** pre-sampled `(nbasis,)` complex arrays for central / spin-orbit / Coulomb terms.

### 2.2 lax today (v1.5, shipped)

Everything below is implemented and tested in lax — verified at the pinned commit (header):

- **`lax.compile(mesh=, channels=|blocks=, energies=, energy_dependent=, mass_factor_grid=, z1z2=, V_is_complex=, method=, grid=, dtype=, device=) → Solver`.** Compile-time: mesh, channel structure, energy grid, boundary values (mpmath Coulomb at `dps` digits), transforms.
- **Symmetry-block batch axis (§15.5).** `blocks=[[ChannelSpec(l=0,…)], [ChannelSpec(l=1,…)], …]` declares N_b independent same-shaped blocks; partial waves are the N_c = 1 case. Boundary values and every observable gain a leading `(N_b,)` axis; the whole pipeline is `vmap`-ped over it. **This is exactly G3's mechanism.**
- **`Interaction` pytree** with static `energy_dependent` / `block_dependent` flags and canonical axis order `(N_b, N_E, M, M)`; `__add__` broadcasts missing axes. Built via `solver.local_potential(fn)`, `solver.nonlocal_potential(fn)`, or `solver.interaction_from_{block,array,funcs}`. Non-local terms get the Gauss scaling `√(λ_i λ_j)·a` applied *by lax* — callers supply raw kernel values `K(r_i, r_j)`. **This is G1's mechanism.**
- **Two evaluation regimes for G2:**
  - *Energy-independent V:* one `solver.spectrum(V)` (diagonalize once), then `solver.smatrix(spectrum)` / `phases(spectrum)` evaluate on the compile-time energy grid → shape `(N_b, N_E, N_c, N_c)`.
  - *Energy-dependent V:* build with `energy_dependent=True`; `spectrum` dispatches the energy axis internally (batched `Spectrum`), then `smatrix_grid` / `phases_grid` / `rmatrix_grid` give aligned-grid observables.
- **`mass_factor_grid`** with shapes scalar / `(N_E,)` / `(N_E, N_c)` — per-energy ℏ²/2μ enters both boundary values and aligned-grid Hamiltonian assembly. This is the upstream change planned in v1, now shipped.
- **Physical-fm mesh.** `MeshSpec.scale` is the channel radius in fm; `mesh.radii` is energy-independent.
- **Method dispatch** `eigh` / `eig` / `linear_solve` keyed on `V_is_complex` and backend; complex optical potentials use `eig` (CPU spectral) or `linear_solve` (GPU direct).

### 2.3 lax gaps (upstream work this rewrite required — **SHIPPED at the pinned commit**)

1. **Two-state non-conjugated matrix element — shipped.** `solver.integrate(values, operator)` computes the single-state conjugating ⟨ψ|O|ψ⟩. The (p,n) DWBA needs the *bilinear* `T ∝ Σᵢ xp_i · U₁(r_i) · xn_i` between two different distorted waves, non-conjugated, batched over `(N_b, N_E)`. lax now provides `solver.matrix_element(bra, ket, operator=None, *, conjugate)` — `conjugate` keyword-required with no default — plus a standalone `lax.transforms.matrix_element` (F1; lax `DESIGN.md` §13.4).
2. **Energy-grid wavefunctions — shipped.** `solver.wavefunction(spectrum, energy, source)` is single-energy. The energy-vectorized DWBA (G2 × Phase 4) needs distorted waves at every grid energy and block: lax now provides `wavefunction_grid` (spectral path, both evaluation regimes) and `wavefunction_direct_grid` (linear-solve path) observables, with the per-energy source stack baked at compile time on `Solver.wavefunction_sources` and exposed via `lax.make_wavefunction_source_grid` (F2; lax `DESIGN.md` §11.2–11.3).

Known lax constraints to design around (documented in lax, confirmed in `compile.py`):

- ~~`momenta=` Fourier transforms rejected with `blocks=`~~ — **lifted upstream** (F3; lax `DESIGN.md` §13 v1.6 note + §15.5): block-batched Fourier and grid transforms shipped in the same lax cycle as F1/F2, so the `folding`/transforms migration (Phase 6) may use blocked solvers, `channels=` solvers, or the standalone `lax.transforms` functions interchangeably.
- Symmetry-block batching is **not supported on propagated multi-interval meshes** → blocked solvers use single-interval Legendre meshes. Fine for the optical-model use case; flag if anyone wants `propagate.py` + blocks later.
- On the **spectral path**, all blocks share one mass factor (`mass_factor_grid` may be `(N_E,)` but not per-block). For elastic j = l ± ½ blocks of a single system this is automatically satisfied; the (p,n) workspace uses *two separate solvers* (proton, neutron) anyway, each internally uniform.
- A **non-uniform `mass_factor_grid` forces the energy-dependent regime** on the spectral path, even for an energy-independent V — diagonalize-once is invalid when H/μ varies with E, and lax's static-regime observables (`smatrix`, `wavefunction_grid`, …) *raise* on that configuration (C4; lax `DESIGN.md` Appendix C.10). See §3.4 for the workspace-side regime rule.
- The **`eig` spectral path is not differentiable** (host-callback eigensolver; C8, lax `DESIGN.md` Appendix C.11): gradient/UQ pipelines with complex optical potentials must use `method="linear_solve"` + `wavefunction_direct_grid`, or restrict to real-V `eigh` problems. See §8.3.

---

## 3. Architecture of the new core

### 3.1 One blocked solver per workspace

Each workspace owns exactly one compiled `lax.Solver` (the (p,n) workspace owns two). Compilation happens in the constructor and is the dominant setup cost; everything after is jitted JAX.

**Block layout for spin-½ on spin-0 elastic.** The current engine solves 2·lmax + 1 decoupled (l, j) channels: l = 0 (j = ½ only), then (l, j = l ± ½) for l ≥ 1. Spin-orbit is diagonal in the (l, j) basis, so the j split is *potential* data, not solver structure: compile **one single-channel block per ℓ**,

```
blocks = [[ChannelSpec(l=l, threshold=0.0, mass_factor=mf)] for l in range(lmax + 1)]
# N_b = lmax + 1
```

and carry j = l ± ½ as **two block-dependent `Interaction`s evaluated against the same solver** — `V⁺` with the j = l + ½ spin-orbit scaling and `V⁻` with j = l − ½:

```python
self._ldots_plus  : (lmax+1,) float   # ⟨l·s⟩ = l/2       for j = l+1/2
self._ldots_minus : (lmax+1,) float   # ⟨l·s⟩ = -(l+1)/2  for j = l-1/2 (l=0 entry unused)
```

Each `smatrix()` evaluation makes **two lax calls** (one per j), each returning `(N_b, N_E, 1, 1)`: `Splus[l] = S⁺[l]` for all l, `Sminus[l] = S⁻[l]` for l ≥ 1, and the l = 0 entry of the minus call is computed-but-ignored (it carries zero weight in every observable kernel). Observables and `spectrum` are stateless in the `Interaction`, and the boundary depends only on ℓ, never j — so this layout costs the same number of solves as 2·lmax + 1 blocks but **halves the compile-time mpmath boundary work** and needs no block-index maps.

Rejected alternatives: (a) interleaved 2·lmax + 1 blocks, one per (l, j) — duplicates every per-ℓ boundary computation and needs `_lsplus`/`_lsminus` index maps; (b) two separate solvers (a "plus" and a "minus" solver) — doubles compile artifacts and splits the spin-orbit logic.

The spin-orbit term enters as a **block-dependent** `Interaction` per j: sample `V_so` once on `mesh.radii` → `(N,)`, outer-product with `_ldots_plus` (resp. `_ldots_minus`) → `(N_b, N)` → diagonal-embed via `interaction_from_array` with `block_dependent=True` (or equivalently `local_potential` with a sequence of N_b scaled callables — use the array path; it avoids N_b retraces).

**Centrifugal/boundary correctness per block is free:** `blocks=` bakes per-l centrifugal operators and per-(l, E) Coulomb boundary values at compile time (`compute_boundary_values_blocks`), eliminating jitr's `free_matrices` / `basis_boundary` machinery.

### 3.2 Energy grid and kinematics

The workspace constructor takes a user-built kinematics object once:

```python
IntegralWorkspace(
    reaction:           Reaction,
    kinematics:         ChannelKinematics, # fields scalar (squeezed 0-d) or (N_E,)
    channel_radius_fm:  float,
    lmax:               int,
    nbasis:             int = 40,
    *,                  # solver knobs forwarded to lax.compile
    V_is_complex:       bool = True,
    method:             str | None = None,
    dtype=jnp.float64, device=None,
)
```

Internally:

1. **The user constructs `kinematics` by whatever method they choose** — `classical_kinematics`, `semi_relativistic_kinematics`, or a custom calculation; the workspace consumes the *object* and never recomputes it. `ChannelKinematics` fields (`Ecm`, `k`, `mu`, `eta`) are allowed to be squeezed 0-d/scalar (single energy) or `(N_E,)` arrays; the workspace normalizes with `np.atleast_1d`, so a scalar reproduces today's single-energy behavior. The `jitr.utils.kinematics` constructors are vectorized to accept array `Elab` (they are nearly so already; audit for scalar-only branches).
2. `energies = jnp.asarray(np.atleast_1d(kinematics.Ecm))` (the lax energy grid is CM, MeV — same convention as `ChannelSpec.threshold`).
3. `mass_factor` from `kinematics.mu`: a scalar/uniform μ → constant `mass_factor`; an `(N_E,)` μ(E) → `mass_factor_grid = hbar²c²/(2 μ(E))` — precisely what the relativistic effective-μ correction needs and why `mass_factor_grid` was pushed upstream.
4. `z1z2 = (Z_proj, Z_target)` for compile-time Coulomb boundary values; `dps=40` default retained.

**Cost model to document for users:** the energy grid is compile-time. Changing energies, lmax, channel radius, or nbasis ⇒ new workspace ⇒ recompile (seconds, mpmath-dominated for charged channels). Changing the *potential* ⇒ no recompile; the jitted `spectrum`/observable path re-executes only. This is the intended fitting/UQ workflow: compile once per (reaction, grid), evaluate per parameter sample.

### 3.3 Potential contract (breaking change, the big one)

**Old contract:** local `(nbasis,)` arrays sampled on `radial_grid()` where the grid is `x·a/k` — energy-dependent.

**New contract:** the grid is `workspace.radial_grid() == np.asarray(solver.mesh.radii)` — physical fm, **energy-independent**, identical for all blocks. Accepted potential inputs per term (central, spin-orbit, Coulomb):

| Input | Interpretation | Resulting `Interaction` axes |
|---|---|---|
| `(N,)` array | local, energy-independent | `(M, M)` |
| `(N_E, N)` array | local, energy-dependent (DOM, dispersive) | `(N_E, M, M)` |
| `(N, N)` array | **non-local kernel** K(rᵢ, rⱼ), raw values (lax applies Gauss scaling) | `(M, M)` |
| `(N_E, N, N)` array | non-local, energy-dependent | `(N_E, M, M)` |
| `(N_b, …)` leading axis on any of the above | **ℓ-/block-dependent** (e.g. Perey–Buck-style ℓ-dependent kernels, parity-dependent terms) | `(N_b, …, M, M)` |
| callable `f(r)`, `f(r, E)`, `f(r, r')`, `f(r, r', E)`, or a length-N_b sequence thereof | convenience path via `solver.{local,nonlocal}_potential` | as above |

Dispatch rule (no arity inference on arrays — shape-driven, with N_E ≠ N and N_b ≠ N_E ambiguities resolved by explicit keyword flags `energy_dependent=`/`l_dependent=` that default to shape-based inference and raise on ambiguity). The workspace sums the per-term `Interaction`s — lax's `__add__` handles axis broadcasting, so mixing a local energy-dependent central term with a static non-local exchange kernel and a block-dependent spin-orbit term "just works":

```python
V = (ws.central(U_central)            # (N_E, N) → energy-dependent local
     + ws.nonlocal(K_exchange)        # (N, N)   → static non-local
     + ws.spin_orbit(U_so)            # (N,)     → block-scaled by ⟨l·s⟩ internally
     + ws.coulomb(U_coul))            # (N,)     → static local
splus, sminus = ws.smatrix(V)         # each (lmax+1, N_E) / (lmax, N_E)
```

**Shape note (internal vs public):** under the §3.1 layout both per-j lax calls return full `(N_b, N_E, 1, 1)` arrays with `N_b = lmax + 1`, including the physically meaningless l = 0 entry of the minus call. The *public* surface keeps today's convention — `Splus` is `(lmax+1, N_E)` and `Sminus` is `(lmax, N_E)` — so the workspace slices the ignored l = 0 entry off the minus call (`Sminus = S⁻[1:]`) before returning. Implementers should not let the internal `(lmax+1,)` minus-call shape leak into the API or the observable kernels.

Expose the term builders (`ws.central` etc.) as thin, validated wrappers over the solver's interaction builders so the assembled `Interaction` is a first-class object users can cache, differentiate through, and reuse — this is what G3 means by "carried as a batch axis on the `Interaction`".

**Migration shim:** none for the grid change — it is silently numerically wrong to feed old-grid samples to the new engine only if shapes happen to match, so the constructor signature break (no more `solver=`, `kinematics=` scalar) is deliberate: old call sites fail loudly. Provide a `CHANGELOG` migration recipe and update every example/notebook (Phase 5).

### 3.4 Observable path

For an energy-independent total `Interaction` (common case: fixed optical potential, all energies):

```
spectrum = solver.spectrum(V)            # diagonalize once per block
S        = solver.smatrix(spectrum)      # (N_b, N_E, 1, 1) on the compile-time grid
```

For energy-dependent `V` (DOM, microscopic E-dependent folding, SCGF-derived potentials): `spectrum` internally dispatches per energy and `solver.smatrix_grid(spectra)` returns the aligned `(N_b, N_E, 1, 1)`. The workspace hides the regime split behind one `smatrix()` method keyed on **`V.energy_dependent` *or* a non-uniform `mass_factor_grid`** — never on `V.energy_dependent` alone: a semi-relativistic μ(E) forces the energy-dependent dispatch even for a static V, and lax's static-regime observables raise on the misconfiguration rather than silently mis-matching (C4; lax `DESIGN.md` Appendix C.10). Under the §3.1 layout each elastic evaluation runs this pipeline **twice** — once for `V⁺`, once for `V⁻` — against the same compiled solver.

From `S`, all observables are closed-form reductions — see §3.5.

**Early-exit semantics change (behavioral, must be documented):** the current per-l loop breaks at `smatrix_abs_tol`. Blocks are static, so all lmax+1 waves are always computed for all energies. Consequences:
- `lmax` must be sufficient for the **highest** grid energy. Provide `jitr.xs.elastic.suggest_lmax(reaction, Elab_max, channel_radius_fm)` using the existing grazing-l heuristic.
- The `smatrix_abs_tol` / `tmatrix_abs_tol` ctor args are **removed entirely** (not kept, not repurposed): there is no per-l early exit to control, and a truncation diagnostic can be reintroduced later as an explicit helper if wanted. `lmax` + `suggest_lmax` are the truncation contract.

### 3.5 Observable kernels in JAX (G4 in `xs`)

Replace the two `@njit` kernels with vectorized JAX (jitted, energy axis native):

- `integral_elastic_xs`: per-energy σ_t, σ_rxn are weighted sums over l of `1 − Re S` and `1 − |S|²`. One einsum each over `(l, N_E)`; weights `(l+1)`/`l` precomputed.
- `differential_elastic_xs`: scattering amplitudes `a(θ, E)`, `b(θ, E)` are matrix products of `(N_θ, l)` Legendre tables (precomputed at ctor with SciPy, static) against `(l, N_E)` S-matrix combinations with the Coulomb phase factors — two einsums, then elementwise dσ/dΩ, A_y, Q. Output shapes `(N_E, N_θ)`.
- `transmission_coefficients`: `1 − |S|²`, shapes `(lmax+1, N_E)` / `(lmax, N_E)`.

`ElasticXS` gains an energy axis on every field; add a `t_plus`/`t_minus` (or keep the separate method — decide in review). Rutherford normalization, Coulomb amplitude `f_c(θ, E)`, and σ_l(E) all gain the `(N_E,)` axis (σ_l now varies with E through η(E) — currently a scalar-η computation; this is a *correctness improvement* for the vectorized workspace, not just a reshape).

These reductions are cheap; JAX is chosen over NumPy here so the full potential → cross-section pipeline stays differentiable end-to-end (UQ/fitting), not for speed.

### 3.6 Quasi-elastic (p,n) workspace

Two blocked solvers, identical `blocks`/`mesh`, different physics:

- **proton (entrance):** `z1z2 = (1, Z)`, `energies = Ecm_p(E)`, `mass_factor_grid = mf_p(E)`
- **neutron (exit):** `z1z2 = (0, Z+1 system)` i.e. no Coulomb, `energies = Ecm_n(E) = Ecm_p(E) − Q_pn`, `mass_factor_grid = mf_n(E)`

(The two solvers may not share a compiled cache; that's fine — the boundary tables differ.)

Pipeline per call to `tmatrix(...)` with the new potential contract (now including optional non-local terms for both distorting potentials, G1):

1. Assemble `V_p` (central + Coulomb + l·s-scaled spin-orbit, any locality) and `V_n` (central + spin-orbit) as block-dependent `Interaction`s — per §3.1, one `V⁺`/`V⁻` pair per distorting potential, evaluated as two calls each.
2. Distorted waves: `xp = wavefunction_grid(spectrum_p)`, `xn = wavefunction_grid(spectrum_n)` → `(N_b, N_E, N)` mesh-coefficient vectors per j-call (upstream item §2.3.2), plus `Sp`, `Sn` from `smatrix`/`smatrix_grid`. (Under `method="linear_solve"` the same shapes come from `wavefunction_direct_grid(V)`.)
3. Isovector transition operator `U₁ = −(U_n − U_p)·(2/√(...))` factor as today; if either input is non-local, `U₁` is a `(N_b?, N_E?, N, N)` kernel and the element is the full bilinear `xpᵀ U₁ xn`; in the local case `U₁` is diagonal and the element reduces to the current node sum.
4. `Tpn[b, e] = matrix_element(xp, xn, U₁, conjugate=False) / (R · k_p(e) · k_n(e))` via the new lax evaluator (upstream item §2.3.1), batched over `(N_b, N_E)`.
5. `xs()` reduces (Tpn, Sn, Sp) over blocks exactly as today, vectorized over E — Legendre tables precomputed.

**Validation anchor:** `chex_qepn_xs.txt` golden data + the Frescox (p,n) lane-coupling regression case (input deck drafted in the DWBA design sessions).

**Wavefunction-convention note (pinned by the lax cross-engine anchor, 2026-06-12; lax `DESIGN.md` Appendix C.12):** the current engine's coefficients relate to lax's per channel as `x_c = (k_c/√a)·[(i/2)(H⁻′ − S·H⁺′)/H⁻]·χ_c` — the bracket is the source-convention conversion (jitr drives the internal solution with the matched exterior *derivative*; lax with the boundary *value* `H⁻(a)`), and `k_c/√a` is the s = k·r vs r coefficient scale. Net: `T_jitr = conv_p·conv_n·matrix_element(χp, χn, U₁)/a²`, verified to machine precision. The rewritten `tmatrix()` should simply **adopt lax's convention** (the conversion factors cancel in |T|²-type observables only up to the S-dependence, so re-derive the `xs()` prefactors once against the Phase 1 goldens rather than carrying the conversion forward).

---

## 4. Numba removal inventory (G4)

Complete site list at HEAD (grep-verified) and disposition:

| Site | What it is | Disposition |
|---|---|---|
| `xs/elastic.py` — `integral_elastic_xs`, `differential_elastic_xs` (2× `@njit`) | observable kernels | **rewrite in JAX** (§3.5) |
| `rmatrix/core.py` (4× `@njit`) | old engine internals | **deleted** with `jitr.rmatrix` (Phase 7) |
| `reactions/channel_on_grid.py` (numba `jitclass`) | old engine channel container | **deleted** with the engine; `lax.ChannelSpec` + the §3.1 per-j bookkeeping replace it |
| `utils/free_solutions.py` — `Gamow_factor` (`@njit`, recursive) | setup-time Coulomb factor | **plain NumPy** (iterative form, vectorize over l); called O(lmax) times at setup — compilation buys nothing |
| `utils/utils.py` — `complex_det`, `block` (2× `@njit`) | small linear-algebra helpers | **plain NumPy**; audit callers — if only the retired engine uses them, delete |
| `optical_potentials/dispersion.py` — `_dispersion_sum`-style inner kernel (`@njit(cache=True, fastmath=True)`) | hot inner loop of dispersive-OMP evaluation | **rewrite in JAX** (`jax.jit` over the precomputed-coefficient sum — the module already preconditions so the kernel is a pure vectorizable reduction). This also makes dispersive potentials differentiable, which they currently are not. Benchmark vs the numba version (Phase 5 exit criterion: within 2× single-eval, faster batched) |

Then: drop `numba` from `pyproject.toml` `dependencies` and the mypy overrides block; add `jax` (and pin a floor matching lax's requirement). CI: add a no-numba import check (`python -c "import jitr"` in an env without numba).

JAX vs NumPy rule of thumb applied above: JAX where the code sits inside the differentiable potential → observable pipeline or benefits from batching; NumPy for setup-time scalar/small-array work.

---

## 5. Module-level migration map

| Current | After |
|---|---|
| `jitr.rmatrix.{Solver, core}` | deleted → `lax.compile` + `Solver` |
| `jitr.quadrature.{Kernel, quadrature}` | deleted → `lax.meshes` (Legendre-x), `solver.integrate`, `lax.transforms` |
| `jitr.quadrature` Fourier–Bessel / double-FB consumers (`folding`, transforms users) | `lax.transforms.fourier` via any solver — blocked or `channels=` (blocks × momenta lifted upstream, §2.3) — or the standalone transform functions |
| `jitr.reactions.channel_on_grid` | deleted |
| `jitr.reactions.wavefunction.Wavefunctions` | rebuilt on `solver.wavefunction` / `wavefunction_grid` + external asymptotics from the boundary cache |
| `jitr.utils.free_solutions` | kept (de-numba'd); still used for Coulomb amplitude / phase shifts in `DifferentialWorkspace` |
| `jitr.xs.elastic`, `jitr.xs.quasielastic_pn` | rewritten per §3 |
| `jitr.optical_potentials.*`, `jitr.folding.*` | API-stable; potentials are now sampled on the fm grid (callers' responsibility unchanged in kind, changed in grid); dispersion kernel de-numba'd |

---

## 6. Testing strategy

Ordering principle (unchanged from v1): **characterization first, refactor second** — per subsystem: write tests → swap backend → tests pass → delete old path.

**Phase 1 golden/characterization tests (against the *current* engine, before any refactor):**
- `xs.elastic`: pin `Splus`/`Sminus`, dσ/dΩ, A_y, Q, σ_t, σ_rxn, transmission coefficients for a representative p + nucleus complex optical case, several energies, small lmax. These become the per-energy slices the vectorized engine must reproduce.
- `xs.quasielastic_pn`: pin Tpn, Sn, Sp, dσ/dΩ (the `chex_qepn_xs.txt` case).
- `folding` (`ILDAFolding`): folded potentials / volume integrals.
- Transforms: Fourier–Bessel / double-FB outputs.
- Dispersive OMP: pin the dispersion-kernel output for a DOM parameter set (guards the numba→JAX rewrite).
- Grid equivalence: `np.allclose(new_radial_grid, lax_mesh.radii)` and an explicit test that the *old* grid (k-scaled) differs — documents the break.

**Cross-engine equivalence (transition window):** lax-backed vs rmatrix-backed S-matrix at matched single (l, j, E) points, bitwise-close tolerance, deleted with the old engine.

**New-capability tests:**
- **G1:** non-local Perey–Buck-type kernel single-channel case vs an independent reference (lax's own non-local test oracle + literature S-matrix values); equivalence of `(N, N)` array path vs callable path; ℓ-dependent kernel exercises `block_dependent=True`.
- **G2:** vectorized output at grid energy eᵢ ≡ scalar-style reference at eᵢ; energy-dependent potential regime (`*_grid` path) vs per-energy independent solves.
- **G3:** blocked solver vs lmax+1 individually compiled `channels=` solvers (this is also tested upstream in lax — keep one smoke test in jitr).
- **G4:** numba-free import test; dispersion JAX-vs-pinned-numba numerics; `jax.grad` smoke test through potential → σ_rxn.
- **Upstream (in lax's suite, not jitr's):** `matrix_element` vs analytic separable case and vs jitr's current node sum; `wavefunction_grid` vs looped `wavefunction`.

**Preserved as oracles:** `tests/regression/*` (Frescox F1–F3, TALYS T1/T2 JLM/JLMB) must pass against the lax-backed workspaces within existing tolerances.

**Invalidated, rewrite to new ctor:** `tests/test_xs_optional_spin_orbit.py` (constructs `Solver(16)`), `tests/test_coupled_single_dwba.py`, `tests/test_wavefunction.py`; `examples/{local,coupled,nonlocal}.py` and notebooks importing `jitr.rmatrix`.

---

## 7. Phased implementation plan

**Phase 0 — Spike / equivalence harness.** One blocked, energy-gridded lax solver vs the current engine for a p + ⁴⁰Ca-class complex optical case: match S(l, j, E) at every grid point. Resolves residual unit/sign/grid-ordering questions in one place before any jitr code changes. *Exit: bitwise-close S over the full (l, j, E) set, both energy regimes (static V and energy-dependent V).*

**Phase 1 — Characterization tests (write first).** Land the §6 golden tests against the current engine. Nothing refactored. *Exit: behavior locked in CI.*

**Phase 2 — lax upstream. ✅ COMPLETE** at the pinned lax commit (header). `matrix_element(bra, ket, operator=None, *, conjugate)` batched over `(N_b, N_E)`, `wavefunction_grid` / `wavefunction_direct_grid`, and block-batched transforms (F1–F3; lax `DESIGN.md` §13.4, §11.2–11.3, §15.5), landed in lax with its own tests. *Exit criteria met:* the evaluator reproduces jitr's `quasielastic_pn` node sum on identical inputs (lax `tests/acceptance/`, ⁴⁸Ca(p,n) classical-kinematics fixture: S to ~8e-13, T to rtol 1e-8 after the closed-form C7 conversion — see §3.6), and `wavefunction_grid` ≡ looped `wavefunction` in lax's unit suite.

**Phase 3 — Elastic workspaces (G1 + G2 + G3, plus the `xs` half of G4).** Rewrite `IntegralWorkspace`/`DifferentialWorkspace` per §3: blocked solver, energy-grid ctor, new potential contract incl. non-local and block-dependent inputs, JAX observable kernels, `suggest_lmax` (tol ctor args removed per §3.4). *Exit: Phase 1 goldens (per-energy slices) + Frescox/TALYS regressions pass; G1/G2/G3 capability tests pass.*

**Phase 4 — Quasi-elastic (p,n) workspace.** Rewrite per §3.6 on the Phase 2 primitives. *Exit: `chex_qepn_xs.txt` + (p,n) lane-coupling regression pass; non-local distorting-potential path exercised.* (Blocked on Phase 2.)

**Phase 5 — Numba removal outside `xs` (rest of G4) + example migration.** De-numba `free_solutions`, `utils`, `dispersion` per §4; migrate single-channel examples/notebooks to the new ctor and fm-grid contract; add deprecation warnings to `rmatrix`/`quadrature` imports. *Exit: no `@njit` outside `rmatrix`/`channel_on_grid`; dispersion benchmark criterion met; examples run.*

**Phase 6 — `folding` / transforms / `wavefunction` migration.** Move remaining `quadrature` consumers onto `lax.transforms` (blocked or `channels=` solvers, or the standalone functions — blocks × momenta lifted upstream, §2.3) behind their Phase 1 goldens; rebuild `reactions.wavefunction`. *Exit: no imports of `jitr.quadrature` outside the module itself.*

**Phase 7 — Retirement.** Delete `jitr.rmatrix`, `jitr.quadrature`, `reactions.channel_on_grid`; drop `numba` from `pyproject.toml`; update `jitr.__init__`, `CHANGELOG.rst`, docs (new potential-contract page is the headline doc item); add the no-numba CI check. *Exit: G4 complete; docs published.*

---

## 8. Open questions for review

1. **Block-axis shape ambiguity in array dispatch (§3.3):** when `N_b == N_E` or `N_E == N`, shape inference is ambiguous. Proposal: explicit `energy_dependent=`/`l_dependent=` kwargs that override inference and raise on unresolvable ambiguity. Confirm, or require explicit flags always (safer, noisier).
2. **`ElasticXS` energy axis:** return `(N_E, …)` arrays in the existing dataclass, or a new `ElasticXSGrid`? Proposal: same dataclass, axis convention documented (`(N_E, N_θ)` for angular fields, `(N_E,)` for integrals) — there is no scalar mode anymore.
3. **Default `method` for complex potentials:** lax dispatches `eig` (CPU) vs `linear_solve` (GPU). The spectral path amortizes across the energy grid; `linear_solve` does not. Decide whether the workspaces force the spectral path on GPU or trust lax dispatch. Needs the Phase 0 spike timings. Note: gradient/UQ work must use `linear_solve` regardless — the `eig` spectral path is not differentiable (host-callback eigensolver; C8, lax `DESIGN.md` Appendix C.11 + §17.2) — so this choice affects plain evaluation only.
4. **`transmission_coefficients` keep as a method or fold into `ElasticXS`?** Trivial either way; pick during Phase 3 review.
5. **`mass_factor_grid` and the energy-dependent-V regime together — RESOLVED upstream:** the spectral `*_grid` path now assembles each per-energy Hamiltonian with its own μ_e (the lax C4 spectrum-kernel fix, verified numerically in lax's suite against per-energy independently compiled references at the pinned commit), and `spectrum(V)` auto-routes to the energy-batched path on non-uniform μ(E). Misconfiguration is loud, not silent: lax's C4 stubs raise if a static-regime observable is used on a non-uniform `mass_factor_grid` solver (C4; lax `DESIGN.md` Appendix C.10). Nothing left to verify in the Phase 0 spike beyond the standard equivalence run.
6. **Deprecation window:** ship one minor release with `rmatrix`/`quadrature` deprecated-but-working (Phases 5–6 span it), or cut straight to removal in one major release? Proposal: single major release, given the ctor break is unavoidable anyway.
