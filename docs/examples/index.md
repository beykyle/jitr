# Example notebooks

`jitr` includes a growing collection of example notebooks that demonstrate how to use the library in various contexts, which live in `examples/notebooks/`. This page brings them together to form the `jitr` tutorial and example gallery. 


```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

/examples/notebooks/builtin_omps_uq
/examples/notebooks/first_elastic_scattering
/examples/notebooks/local_omp_demo
/examples/notebooks/optical_potentials_and_observables
/examples/notebooks/integration
/examples/notebooks/how_to_define_your_interaction
/examples/notebooks/reaction
/examples/notebooks/convergence_channel_radius
/examples/notebooks/chuq_kduq_comp
/examples/notebooks/angular_reaction_xs
/examples/notebooks/mass_exploration
```

## Full "batteries included" example:

- [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
  is the flagship user-facing example, walking through posterior sampling, solver setup, and interval construction. It compares built-in uncertainty quantified optical potentials for your reaction of interest.

## Start here

- [first elastic observables with `xs.elastic`](/examples/notebooks/first_elastic_scattering)
   is the simplest example of how to use `jitr` to compute an observable, in this case elastic scattering. It also introduces the `xs` submodule, which provides a convenient interface for computing reaction observables.
  The submodule `xs` provides the functionality for computing reaction observables, of which elastic scattering is the simplest example.
- [Elastic scattering with `LocalOpticalPotential`](/examples/notebooks/local_omp_demo)
  `jitr` provides a convenient interface for defining your own optical potentials, which all the built in potentials ofn the previous examples follow, and provides the `LocalOpticalPotential` class as the simplest example of this interface.
- [built-in optical potentials with `xs.elastic`](/examples/notebooks/optical_potentials_and_observables)
   shows how to use the built-in optical potentials in `xs` to compute elastic scattering observables, and how to compare them to data. This example also introduces the `OpticalPotential` class, which is the base class for all optical potentials in `jitr`. 

## Solver and model-building notebooks

- [Numerical integration building blocks](/examples/notebooks/integration)
  gives lower-level context for the quadrature and integration utilities which form the backbone of the solver.
- [How to define your own interaction](/examples/notebooks/how_to_define_your_interaction)
  shows how to define a custom optical potential.
- [Reactions and kinematics](/examples/notebooks/reaction)
  introduces the `Reaction` class and shows how to use it to store useful information about the reaction of interest, and how to use it to compute kinematic quantities.
- [Channel-radius convergence study for elastic scattering](/examples/notebooks/convergence_channel_radius)
  shows how to check a key numerical convergence knob in a realistic workflow.
- [Compare global optical-potential radial forms](/examples/notebooks/chuq_kduq_comp)
  provides a quick visual comparison of the radial forms of the built-in global potentials.

## Uncertainty-propagation workflows

- [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
  mentioned above.
- [Angular reaction observables with mass-model uncertainty](/examples/notebooks/angular_reaction_xs)
  performs uncertainty quantification for partial wave transmission coefficients
- [Mass-model effects on transmission-coefficient uncertainty](/examples/notebooks/mass_exploration)
  explores how mass-model choices propagate into transmission coefficients.

