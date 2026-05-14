# Draft tutorial: uncertainty propagation with built-in optical potentials

```{note}
This page is a narrative tutorial draft for the built-in uncertainty
workflow. It is designed to complement the flagship
[`builtin_omps_uq.ipynb`](/examples/notebooks/builtin_omps_uq) notebook
rather than replace it.
```

## Who this is for

This tutorial is for users who already understand the basic elastic
workflow and now want to compare or propagate uncertainty through the
built-in optical-potential models.

If you are brand new to `jitr`, read
[Draft tutorial: your first elastic-scattering calculation](first-elastic-scattering.md)
first.

## The workflow in one sentence

For the main built-in OMP use case, uncertainty propagation in `jitr`
means:

1. define a reaction and kinematics,
2. build a reusable solver/workspace configuration,
3. draw or load uncertain model parameters,
4. evaluate the observable repeatedly,
5. summarize the spread in the results.

This page is intentionally centered on the built-in OMP workflow because
that is likely to be one of the most common user paths through the
library.

## Step 1: define the observable and the source of uncertainty

Before you write any code, decide what is uncertain.

The main case to keep in mind here is:

- compare several built-in uncertainty-aware optical-potential families
  on the same elastic observable.

The observable should be equally explicit. For example:

- differential elastic cross section,
- ratio to Rutherford,
- angular asymmetry,
- transmission coefficients.

That matters because the cleanest reusable setup is an
observable-specific workspace paired with a library of model families,
not a one-off script tied to a single potential.

## Step 2: set up the reaction and baseline kinematics

The flagship notebook starts from a standard elastic-reaction setup:

```python
import numpy as np
import jitr

target = (54, 26)
projectile = (1, 1)
reaction = jitr.reactions.ElasticReaction(
    target=target,
    projectile=projectile,
)

Elab = 35.0
kinematics = reaction.kinematics(Elab)
angles = np.linspace(0.1, np.pi, 180)
```

The point of doing this once, up front, is that these ingredients do not
change when you switch from one built-in potential family to another.

## Step 3: build a reusable workspace

The built-in OMP workflow uses an elastic differential workspace to
capture the numerical setup:

```python
core_solver = jitr.rmatrix.Solver(40)

a = jitr.utils.interaction_range(target[0]) * kinematics.k + 2 * np.pi
channel_radius_fm = a / kinematics.k

workspace = jitr.xs.elastic.DifferentialWorkspace.build_from_system(
    reaction=reaction,
    kinematics=kinematics,
    channel_radius_fm=channel_radius_fm,
    solver=core_solver,
    lmax=50,
    angles=angles,
)
```

This is one of the most important organizational choices in the whole
workflow. It keeps:

- the reaction definition,
- the numerical basis,
- the angular grid,
- and the observable-specific settings

in one reusable object instead of scattering them across the loop over
samples.

## Step 4: choose the built-in model families to compare

In the highlighted notebook, the relevant imports are:

```python
from jitr.optical_potentials import chuq, kduq, wlh
```

This is the high-level pattern the docs should emphasize. A user is not
starting by designing a new uncertainty workflow; they are starting by
asking what the built-in model families predict for the same observable.

## Step 5: run the propagation loop

Conceptually, the loop is simple:

```python
results = []
for sample in posterior_samples:
    observable = evaluate_sample(sample, workspace)
    results.append(observable)
```

In practice, the helper used inside `evaluate_sample` depends on the
particular optical-potential family or posterior representation. The
important part of the tutorial is the division of responsibilities:

- the workspace defines the fixed calculation,
- the sample defines what changes from one evaluation to the next,
- the result array stores the observable realization.

That separation makes it much easier to compare built-in model families
fairly.

## Step 6: summarize and visualize the uncertainty

Once you have a collection of observable realizations, the next task is
usually to compute credible intervals, bands, or a model comparison
summary.

Typical questions to ask here are:

- What is the median or mean prediction?
- How wide is the interval at each angle or energy?
- Which regions are most sensitive to the uncertain input?
- Does one uncertainty model dominate another?

The current notebook already contains plotting logic for these steps. A
future polished tutorial should standardize a small set of summary plots
for the built-in OMP comparison workflow first, then generalize later.

## How the current notebooks fit together

The current uncertainty examples are best understood as a sequence:

1. [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
   is the flagship multi-model workflow and should be the default next
   stop for many users.
2. [Differential cross-section uncertainty propagation with KDUQ](/examples/notebooks/kduq_uq_demo)
   is the narrower single-model companion when a user wants to focus on
   one built-in family.
3. [Mass-model effects on transmission-coefficient uncertainty](/examples/notebooks/mass_exploration)
   and
   [Angular reaction observables with mass-model uncertainty](/examples/notebooks/angular_reaction_xs)
   extend the uncertainty story into a different modeling axis.

## Common failure modes to watch for

These are the most important things to check before trusting the
propagated uncertainty:

- the baseline deterministic calculation is already stable,
- the channel radius and basis size are adequate,
- the observable definition is consistent across samples,
- the sample source matches the built-in model being evaluated,
- and the summary statistic matches the scientific question.

Uncertainty propagation can make a workflow look sophisticated while
hiding a weak baseline setup. That is why the deterministic
elastic-scattering tutorial should come first.

## Where to go next

- For the baseline calculation:
  [Draft tutorial: first elastic observables with `xs.elastic`](first-elastic-scattering.md)
- For the main entry-point tutorial:
  [Draft tutorial: built-in optical potentials with `xs.elastic`](optical-potentials-and-observables.md)
- For the executable flagship workflow:
  [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)

## APIs worth keeping open while you read

- [Optical potentials](../api/optical-potentials.md)
- [R-matrix solver](../api/rmatrix.md)
- [Observables and workspaces](../api/xs.md)
