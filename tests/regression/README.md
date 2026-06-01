# External-reference regression tests

This package holds end-to-end regression tests that compare `jitr` against
committed outputs from independent reaction codes.

## Current status

- The landed Frescox elastic slice now covers one upstream proton baseline plus
  additional proton and neutron elastic energies in `frescox/reference/`.
- The harness is intentionally small: one manifest, one loader, one builder
  module, and one parametrized pytest entrypoint.
- The design document's `test/regression/` layout is adapted here to
  `tests/regression/` so it works with the repository's existing pytest and CI
  configuration.

## Layout

- `manifest.json`: committed case inventory
- `_readers.py`: manifest and reference-data loader
- `_builders.py`: API-specific workspace/input assembly
- `test_regression.py`: end-to-end regression assertion
- `frescox/`: committed Frescox inputs, references, and parser
- `talys/`: TALYS scaffolding and sample-selection notes

## Notes

- Only committed CSV/JSON references are read in pytest; neither Frescox nor
  TALYS is required in CI.
- Frescox cases use integer-amu input masses (`m = A * AMU`) plus classical
  kinematics so the regression matches the upstream deck conventions exactly.
- Later DWBA and TALYS slices still require public APIs that do not yet exist
  on this branch.
- Neutral elastic regressions now omit the Coulomb matrix entirely in the
  builder, so neutron metadata can stay physically literal instead of carrying
  a dummy Coulomb term for execution.
