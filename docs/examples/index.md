# Example notebooks

`jitr` includes a growing collection of example notebooks that demonstrate how to use the library in various contexts, which live in `examples/notebooks/`.


```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

/examples/notebooks/quickstart
/examples/notebooks/alpha_ca_calibration
/examples/notebooks/reaction
/examples/notebooks/example_jlm
/examples/notebooks/kd03_neutron_dxs_demo.ipynb
/examples/notebooks/builtin_omps_uq
/examples/notebooks/tabulated_density_demo
/examples/notebooks/chuq_kduq_comp
/examples/notebooks/volume_integrals
/examples/notebooks/convergence_channel_radius
/examples/notebooks/dispersive_optical_model
/examples/notebooks/kduq_uq_demo
/examples/notebooks/local_omp_demo
/examples/notebooks/angular_reaction_xs
/examples/notebooks/mass_exploration
```


## Start here

- [Quickstart](/examples/notebooks/quickstart)
  compiles a solver for $\alpha$ + $^{44}$Ca, defines a parametric optical potential, and fits it to real elastic scattering data, all in under a minute.
- [Bayesian calibration](/examples/notebooks/alpha_ca_calibration)
  extends the quickstart to a full Bayesian calibration for $\alpha$ + $^{40,44,48}$Ca, using a simple error model, comparing the posterior across isotopes and checking the posterior predictive against the data.
- [Bayesian calibration with multimodal posteriors](/examples/notebooks/alpha_ca48_ambiguity.ipynb)
  extends the quickstart to a tempered Bayesian calibration for $\alpha$ + $^{48}$Ca, exploring the discrete parameter ambiguity in the posterior in a Bayesian context, and checking the posterior predictive against the data.



## Other useful examples

- [Reactions and kinematics](/examples/notebooks/reaction)
  introduces the `Reaction` class and shows how to use it to store useful information about the reaction of interest, and how to use it to compute kinematic quantities.
- [Koning-Delaroche neutron elastic scattering](/examples/notebooks/kd03_neutron_dxs_demo.ipynb)
  shows how to compute the elastic scattering cross section for neutrons on various targets using the built-in Koning-Delaroche optical potential in `jitr`.
- [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
  walks through posterior sampling, solver setup, and interval construction 
  for several of the built-in uncertainty quantified optical potentials in `jitr`.
- [Tabulated neutron and proton densities](/examples/notebooks/tabulated_density_demo)
- [JLM and JLMB semi-microscopic optical-potentials](/examples/notebooks/example_jlm)
  converts the original JLM example into a notebook and walks through folded
  microscopic potentials, Lane trends, and tabulated self-energy usage.
- [Compare global optical-potential radial forms](/examples/notebooks/chuq_kduq_comp)
  provides a quick visual comparison of the radial forms of the built-in global potentials in `jitr`.
- [Visualize global optical potential volume integrals](/examples/notebooks/volume_integrals)
  shows how to compute and visualize volume integrals for the built-in global optical potentials in `jitr`.
- [Channel-radius convergence study for elastic scattering](/examples/notebooks/convergence_channel_radius)
  shows how to check for numerical convergence in realistic calculations.
  shows how to grab the tabulated nuclear densities in `jitr`.
- [Dispersive optical model](/examples/notebooks/dispersive_optical_model)
  shows how to use the built-in dispersive optical model in `jitr`
- [UQ demo with Koning-Delaroche potential](/examples/notebooks/kduq_uq_demo) 
  demos propagating the uncertainty of the built-in KDUQ interaction.
- [Optical potential interface](/examples/notebooks/local_omp_demo)
  shows how to define a custom optical potential using a common interface, and how to use it in a calculation.
- [Transmission coefficients by partial wave](/examples/notebooks/angular_reaction_xs)
  performs uncertainty quantification for partial wave transmission coefficients
- [Mass-model effects on transmission-coefficient uncertainty](/examples/notebooks/mass_exploration)
  explores how mass-model choices propagate into transmission coefficients.
