=============
Release Notes
=============

Below are the notes from all `jitr` releases. For more details see https://github.com/beykyle/jitr/releases

Release 2.0 (unreleased)
------------------------
:Date: TBD

The internal R-matrix engine (``jitr.rmatrix``, ``jitr.quadrature``,
``jitr.reactions.channel_on_grid``) has been removed and the ``jitr.xs``
workspaces rebuilt on the external `lax <https://github.com/beykyle/lax>`_
solver package. This is a breaking release; old workspace call sites fail
loudly rather than silently misbehave.

**New capabilities**

* Non-local interaction kernels (``(N, N)`` arrays or ``f(r, r')``
  callables), energy-dependent and ℓ-dependent terms, all composable via
  ``+`` — see ``docs/potential-contract.md``.
* Energy-vectorized workspaces: ``ChannelKinematics`` fields may be
  ``(N_E,)`` arrays; one workspace solves the whole energy grid at once.
* All partial waves solved as a single vectorized block on the JAX
  backend; the potential → observable pipeline is differentiable with
  ``method="linear_solve"``.
* Semi-relativistic (energy-dependent effective mass) kinematics are
  mapped onto the solver exactly, including the interior equation.

**Migration recipe**

* ``IntegralWorkspace`` / ``DifferentialWorkspace`` /
  ``quasielastic_pn.Workspace`` constructors: the ``solver=Solver(n)``
  argument is replaced by ``nbasis=n``; pass ``channel_radius_fm`` in fm
  (was the dimensionless ``a = k·R``); kinematics may now be arrays.
* ``smatrix_abs_tol`` / ``tmatrix_abs_tol`` are removed — there is no
  per-ℓ early exit. All waves up to ``lmax`` are computed; use the new
  ``jitr.xs.elastic.suggest_lmax`` to choose ``lmax`` for the highest
  grid energy.
* ``radial_grid()`` is now documented as energy-independent physical fm.
  Numerically it is identical to the old public grid, so potential
  sampling code is unchanged — but it is now sampled once for all
  energies, not per energy.
* Axis conventions (no scalar-energy mode): partial-wave arrays are
  trailing-energy (``Splus`` is ``(lmax+1, N_E)``, ``Sminus`` is
  ``(lmax, N_E)``); observables are leading-energy (``dsdo``/``Ay``/``Q``
  are ``(N_E, N_θ)``; ``t``/``rxn`` are ``(N_E,)``). For a scalar
  ``Elab``, take ``[0]`` / ``[..., 0]`` slices.
* ``jitr.reactions.Wavefunctions`` is replaced by
  ``jitr.reactions.DistortedWaves`` (interior + exterior evaluation from
  a workspace and an interaction).
* The surviving quadrature transforms (Fourier–Bessel, double
  Fourier–Bessel, Legendre/Laguerre meshes) moved to pure NumPy in
  ``jitr.utils.transforms``; everything else from ``jitr.quadrature``
  lives upstream in ``lax``.
* ``numba`` is no longer a dependency; ``jax`` is. ``import jitr`` works
  without ``lax`` installed — only constructing an ``xs`` workspace (or
  ``DistortedWaves``) requires it. Until ``lax`` is published to PyPI,
  install it from source (``pip install -e <path-to-lax>``); tests that
  need it are marked ``requires_lax`` and auto-skip.
* Coupled-channels support was an explicit non-goal of the rewrite:
  ``examples/coupled.py`` and the solver-level/coupled-channels notebooks
  (``integration``, ``how_to_define_your_interaction``,
  ``comparison_to_Runge_Kutta``, ``test_coupled_single_dwba``) are removed
  with the engine they demonstrated; they remain available in the 1.x git
  history pending upstream ``lax`` coupled-channel support.

**Bug fixes**

* The legacy non-local solver path was incorrect (and untested); the new
  engine reproduces published Yamaguchi-potential phase shifts
  (Descouvemont, 2016) to four decimal places. Results obtained from the
  old non-local path should be regenerated.
* ``σ_l`` Coulomb phases now vary with energy across the grid (previously
  computed at a single scalar η).

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
