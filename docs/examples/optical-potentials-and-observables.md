# Draft tutorial: built-in optical potentials with `xs.elastic`

```{note}
This is a narrative draft tutorial. It is designed to accompany
[`builtin_omps_uq.ipynb`](/examples/notebooks/builtin_omps_uq), which is
one of the strongest user-facing workflows in the current documentation.
```

## Why this tutorial matters

If `jitr` is doing its job well, many users should not need to think
about the full library hierarchy on day one. They should be able to:

1. choose a reaction and observable,
2. select one or more built-in optical-potential models,
3. evaluate those models through a reusable `xs.elastic` workflow,
4. compare predictions or uncertainty bands.

That is exactly why the built-in OMP uncertainty notebook should be
treated as a highlighted workflow: it exercises a large, meaningful
portion of the library through the main user entry points.

## Who this is for

This tutorial is for users who already understand the basic elastic
workflow and now want to work productively with the library's main
high-level interfaces:

- `jitr.optical_potentials`
- `jitr.xs.elastic`

If you are completely new to the package, start with
[Draft tutorial: first elastic observables with `xs.elastic`](first-elastic-scattering.md).

## The high-level picture

For this style of calculation, the most important separation is:

- `optical_potentials` tells you **what interaction model** you are
  evaluating,
- `xs.elastic` tells you **what observable workflow** you are running.

That separation is useful because it lets you keep the observable setup
fixed while swapping optical-potential models, posterior samples, or
parameter sets.

## Step 1: define one elastic observable problem

The `builtin_omps_uq` notebook starts from a single elastic-scattering
problem:

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
angles = np.linspace(0.1, np.pi, 100)
```

Everything after this point is about evaluating model choices against
the same observable target.

## Step 2: choose the built-in optical-potential families

The flagship notebook pulls in three built-in uncertainty-aware optical
potential families:

```python
from jitr.optical_potentials import chuq, kduq, wlh
```

That is the key user-facing idea: the library gives you a set of
built-in model families that can be compared on equal footing. You do
not have to build each workflow from scratch before you can start asking
interesting scientific questions.

This is a better central tutorial than an abstraction-selection page
because it starts from a real user goal: compare credible predictions
for the same observable using the models the library already provides.

## Step 3: build the shared `xs.elastic` workflow once

The observable setup should be created once and reused across model
evaluations. In the current examples, that means building an elastic
workspace around the reaction, kinematics, solver, and angular grid.

Conceptually, it looks like this:

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

The point is not that every user must memorize this call. The point is
that `xs.elastic` is the main place where a user turns model choices
into observable calculations.

## Step 4: evaluate each model through the same observable path

Once the workspace exists, each built-in optical-potential family can be
run through the same observable machinery. That makes the comparison
fair and the code structure easy to reason about:

- the reaction is fixed,
- the energy is fixed,
- the angular grid is fixed,
- the solver settings are fixed,
- only the model family or sampled parameters change.

This is the right mental model for much of the high-level library use:
hold the observable workflow steady, and vary the model inputs you want
to compare.

## Step 5: compare predictions and uncertainty

This is the part that users usually care about most. Once the model
families have all been evaluated, the remaining tasks are:

- compute the quantity you want to compare,
- summarize central values and intervals,
- inspect where the models agree or diverge,
- decide whether the observable is sensitive enough for your question.

That is why `builtin_omps_uq.ipynb` deserves special treatment in the
docs. It is not only an uncertainty notebook; it is also a showcase for
how the library's major user-facing pieces fit together naturally.

## What this tutorial should eventually become

A polished version of this tutorial should probably be the main
follow-up to the first elastic tutorial. It should:

1. start from the same `xs.elastic` setup,
2. introduce the built-in OMP families one by one,
3. show the repeated-evaluation pattern clearly,
4. end with a comparison plot and interpretation checklist.

That gives users a direct path from “I can compute one observable” to
“I can compare serious model workflows using the built-in tools.”

## Where to go next

- For the shortest path to an elastic observable:
  [Draft tutorial: first elastic observables with `xs.elastic`](first-elastic-scattering.md)
- For the executable flagship workflow:
  [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
- For a narrower single-model uncertainty workflow:
  [Draft tutorial: uncertainty propagation with built-in optical potentials](uncertainty-propagation.md)

## APIs worth keeping open while you read

- [Optical potentials](../api/optical-potentials.md)
- [Observables and workspaces](../api/xs.md)
- [R-matrix solver](../api/rmatrix.md)
