=============
Release Notes
=============

Below are the notes from all `jitr` releases. For more details see https://github.com/beykyle/jitr/releases

Release 4.0
-------------

jitr 4 replaces the numba engine with a JAX R-matrix core. The API changes;
see ``docs/migration-v4.md`` for a table mapping the v3 API onto v4.

* The solver is written in JAX: just-in-time compiled, runs on GPUs, and is
  differentiable from the potential through to the observables.
* Systems are solved by J^pi block. Workspaces take a grid of energies and solve
  every block at every energy in one vectorized call.
* ``jitr.xs.coupled.Workspace`` solves general coupled-channels systems of
  levels with any spin, with spin-orbit and rigid-rotor couplings and closed
  channels.
* ``jitr.xs.amplitudes``: cross sections, vector and tensor analyzing powers
  for projectiles of any spin; deuteron elastic scattering in
  ``jitr.xs.elastic``.
* ``jitr.xs.lane_pn`` is merged into ``jitr.xs.quasielastic_pn``, which
  computes both the DWBA and the coupled-channels (p,n) observables.
* ``jitr.rmatrix.Solver``, ``jitr.quadrature``, ``ProjectileTargetSystem`` and
  the channel classes are removed; ``jitr.rmatrix`` holds the mesh and the
  solution of one block.
* The built-in optical potentials evaluate over an energy grid, returning
  ``(NE, N)`` terms for a workspace over the same energies;
  ``elastic.Workspace.at_angles`` reuses a solver for new angles.
* ``block_sizes`` solves groups of J^pi blocks each padded only to its own
  largest block (opt-in; one compiled solve per group). Batches of
  potentials go through ``jax.vmap`` (documented and tested).
* No early exit in partial waves: every wave up to ``lmax`` is solved, and
  ``jitr.utils.suggest_lmax`` picks ``lmax``.
* Fixes: nonlocal interactions were divided by the channel wavenumber once too
  often; ``yamaguchi_swave_delta`` had the wrong sign of the scattering length.
* Regression tests against the benchmarks of Descouvemont, Comput. Phys.
  Commun. 200 (2016) 199, and against FRESCO and TALYS.
* numba is no longer a dependency.

Release 1.3
-------------
:Date: Aug 1, 2024

* BAND framework compatibility

Release 1.2
-------------
:Date: Jul 29, 2024

* Fix bug in coupled channels, refactor for independence of solver components and physical system Latest


Release 1.1
-------------
:Date: Mar 27, 2024

* update kernel for modularity and ease of use in parametric simulations

Release 1.0
-------------
:Date: Oct 29, 2023

This is the initial functioning release of the framework, which includes support for

  * non-local potentials
  * coupled-channels systems
  * just-in-time compilation


Release 0.3
-------------
:Date: Oct 24, 2023

* major refactors including use of numba.njit for core solver


Release 0.2
-------------

:Date: Aug 14, 2023

* add formatter by @beykyle in https://github.com/beykyle/lagrange_rmatrix/pull/6
* add dynamic versioning by @beykyle in https://github.com/beykyle/lagrange_rmatrix/pull/7


**Full Changelog**: https://github.com/beykyle/lagrange_rmatrix/compare/v0.1.5...v0.2

Release 0.1.5
-------------

:Date: Aug 9, 2023

* tests automated by github actions and `pytest`
* automatic PyPI distribution on release


Release 0.1.4 
-------------

:Date: Aug 9, 2023

* Update pypi-publish.yml by @beykyle in https://github.com/beykyle/lagrange_rmatrix/pull/5


**Full Changelog**: https://github.com/beykyle/lagrange_rmatrix/compare/v0.1.3...v0.1.4

Release 0.1.0
-------------

:Date: Aug 9, 2023

* Initial implementation and organization of project into python package
