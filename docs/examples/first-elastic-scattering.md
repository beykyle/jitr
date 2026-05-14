# Draft tutorial: first elastic observables with `xs.elastic`

```{note}
This is a documentation draft, not a fully executed notebook. It is meant
to give new users a clear first-run path through `jitr` and point them to
the richer example notebooks once the overall workflow makes sense.
```

## Who this is for

This tutorial is for a new user who wants one complete success case:
define an elastic reaction, build an `xs.elastic` workflow, and compute
an observable without first learning every lower-level piece.

## What you should know first

- You have `jitr` installed.
- You are comfortable with a short Python script or notebook.
- You do **not** need to know the full solver stack yet.

If you want the executable notebook that complements this page, start
with
[Elastic scattering with `LocalOpticalPotential`](/examples/notebooks/local_omp_demo).

## What you will build

The shortest useful elastic-scattering workflow in `jitr` has four
moving parts:

1. A reaction definition.
2. Channel kinematics at a chosen beam energy.
3. A built-in optical-potential model or parameterization.
4. An `xs.elastic` workspace that turns those ingredients into
   observables.

The point of this page is to show the main user path through the library:
reaction -> kinematics -> `xs.elastic` -> observable.

## Step 1: define the reaction

In `jitr`, a reaction is represented by a `Reaction`-family object. For a
simple elastic case, the target and projectile are often enough.

```python
from jitr.reactions.reaction import Reaction

target = (208, 82)      # 208Pb
projectile = (1, 1)     # proton

reaction = Reaction(target=target, projectile=projectile, process="el")
```

At this point, you have a compact object that knows the entrance
channel, the compound system, and the metadata needed by later solver
steps.

**Why start here?** Because the rest of the library assumes you can
describe the system cleanly before you worry about basis size, channel
radii, or uncertainty propagation.

## Step 2: compute the kinematics

Most of the remaining workflow depends on channel kinematics at a
particular laboratory energy.

```python
energy_lab = 80.0  # MeV
kinematics = reaction.kinematics(energy_lab)

print(kinematics.Ecm, kinematics.k, kinematics.eta)
```

The returned `ChannelKinematics` object packages the quantities that show
up repeatedly throughout the rest of the calculation:

- center-of-mass energy,
- reduced mass,
- wave number,
- Coulomb parameter.

You do not need to compute or track those pieces by hand.

## Step 3: choose an optical-potential model

For a first calculation, a built-in optical-potential interface is the
most natural starting point. One common entry point is
`LocalOpticalPotential`, which gives you a standard local
single-channel optical-model interface.

```python
from jitr.optical_potentials import LocalOpticalPotential

optical_model = LocalOpticalPotential()
print(optical_model.params)
```

The parameter list tells you which real, imaginary, spin-orbit, and
Coulomb terms the model expects. In a real analysis, those values might
come from a published parameterization, a fit, or a posterior sample.

For onboarding, the important idea is simpler: the optical potential is
the object that turns your reaction and kinematics into the interaction
terms the solver consumes.

## Step 4: build the `xs.elastic` workspace

This is the real center of the onboarding path. If your goal is “get one
observable quickly,” the `xs.elastic` workspace is the important
high-level entry point:

```python
import numpy as np
import jitr

angles = np.linspace(0.1, np.pi, 180)
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

You can think of this workspace as the object that bundles:

- the reaction definition,
- the computed kinematics,
- the numerical solver,
- the angular grid,
- and the observable-specific settings

into one reusable elastic-scattering workflow.

## Step 5: treat this as a reusable observable pipeline

Once the workspace exists, you have the structure you need for most
high-level user workflows. You can now:

- evaluate one parameter set,
- compare multiple optical potentials,
- loop over posterior samples,
- and inspect how the observable changes

without redesigning the calculation from scratch each time.

## Step 6: interpret the result

The first thing to inspect is not necessarily the prettiest plot. It is
whether the workflow itself is coherent:

- Did the reaction object match the physical system you intended?
- Is the chosen energy in the regime you care about?
- Is the channel radius large enough for the interaction you selected?
- Does the observable behave sensibly before you start fitting or
  propagating uncertainty?

That mindset matters because `jitr` is designed for reusable workflows.
Once the first calculation is trustworthy, you can swap in a different
potential, increase basis size, add uncertainty samples, or move into
the flagship built-in OMP uncertainty workflow without rewriting the
whole analysis.

## Where to go next

The strongest next step for most users is:

- [Draft tutorial: built-in optical potentials with `xs.elastic`](optical-potentials-and-observables.md)
  for the main user-facing workflow built around library entry points.

Then, depending on your goal:

- read [Compare built-in uncertainty-quantified optical potentials](/examples/notebooks/builtin_omps_uq)
  for the flagship executable workflow,
- or read [Elastic scattering with `LocalOpticalPotential`](/examples/notebooks/local_omp_demo)
  for the lower-level notebook companion to this page.

## APIs worth keeping open while you read

- [Core package](../api/core.md)
- [Reactions](../api/reactions.md)
- [Optical potentials](../api/optical-potentials.md)
- [Observables and workspaces](../api/xs.md)
