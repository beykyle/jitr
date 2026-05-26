# Example notebooks

`jitr` includes a growing collection of example notebooks that demonstrate how to use the library in various contexts, which live in `examples/notebooks/`.


```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

/examples/notebooks/quickstart
/examples/notebooks/reaction
/examples/notebooks/example_jlm
/examples/notebooks/builtin_omps
/examples/notebooks/builtin_omps_uq
/examples/notebooks/kduq_uq_demo
/examples/notebooks/local_omp_demo
/examples/notebooks/angular_reaction_xs
/examples/notebooks/mass_exploration
/examples/notebooks/chuq_kduq_comp
/examples/notebooks/volume_integrals
/examples/notebooks/convergence_channel_radius
```


## Start here

- [quickstart](/examples/notebooks/quickstart)
  gives a full end-to-end example for compiling a solver for a given reaction system, defining a parametric interaction potential, and calculating an elastic scattering cross section for an ensemble of potential parameters.
- [Reactions and kinematics](/examples/notebooks/reaction)
  introduces the `Reaction` class and shows how to use it to store useful information about the reaction of interest, and how to use it to compute kinematic quantities.

## Full "batteries included" example:

- [JLM and JLMB optical-potential benchmarks](/examples/notebooks/example_jlm)
  converts the original JLM example into a notebook and walks through folded
  microscopic potentials, Lane trends, and tabulated self-energy usage.
- [Compare built-in OMPs with JLM and JLMB](/examples/notebooks/builtin_omps)
  gives a deterministic `n + 208Pb` comparison of `kduq`, `chuq`, `wlh`,
  `jlm`, and `jlmb` cross sections and central volume integrals.
- [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
  walks through posterior sampling, solver setup, and interval construction for several of the built-in uncertainty quantified optical potentials in `jitr`.

## Other useful examples

- [UQ demo qith Koning-Delaroche potential](/examples/notebooks/kduq_uq_demo) 
  demos propagating the uncertainty of the built-in KDUQ interaction.
- [Optical potential interface](/examples/notebooks/local_omp_demo)
  shows how to define a custom optical potential using a common interface, and how to use it in a calculation.
- [Transmission coefficients by partial wave](/examples/notebooks/angular_reaction_xs)
  performs uncertainty quantification for partial wave transmission coefficients
- [Mass-model effects on transmission-coefficient uncertainty](/examples/notebooks/mass_exploration)
  explores how mass-model choices propagate into transmission coefficients.
- [Compare global optical-potential radial forms](/examples/notebooks/chuq_kduq_comp)
  provides a quick visual comparison of the radial forms of the built-in global potentials in `jitr`.
- [Visualize global optical potential volume integrals](/examples/notebooks/volume_integrals)
  shows how to compute and visualize volume integrals for the built-in global optical potentials in `jitr`.
- [Channel-radius convergence study for elastic scattering](/examples/notebooks/convergence_channel_radius)
  shows how to check for numerical convergence in realistic calculations.
