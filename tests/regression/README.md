# Regression harness

End-to-end tests comparing `jitr` against committed reference outputs from
Frescox (F1–F10) and TALYS (T1–T4). See the
[regression-tests documentation](../../docs/regression-tests.md) for a case
overview.

## Layout

- `manifest.json` — committed case inventory (all active cases)
- `_readers.py` — manifest and reference-data loader
- `_builders.py` — API-specific workspace/input assembly
- `conftest.py` — parametrizes pytest over `manifest.json`
- `test_regression.py` — Frescox case assertions
- `test_talys_reference.py` — TALYS parser smoke tests
- `frescox/` — Frescox inputs, reference CSVs/JSON, and parser
- `talys/` — TALYS inputs, reference CSVs/JSON, and parser

## Design notes

- Only committed CSV/JSON references are read in pytest; neither Frescox nor
  TALYS is required in CI.
- Frescox cases use integer-AMU masses (`m = A × AMU`) and classical
  kinematics, matching the upstream deck convention exactly.
- Neutral elastic cases omit the Coulomb matrix in the builder; neutron
  metadata stays physically literal.
- The `(p,n)` cases (F9, F10) use `observable_type: "quasielastic_pn"` and
  build `jitr.xs.quasielastic_pn.Workspace`. Both channels take their lab
  energy straight from the deck, so jitR and Frescox see identical
  kinematics rather than jitR's Q-value tables. `U1_central` is left to
  jitR's default isovector difference, which is what the deck's tabulated
  form factor holds; `U1_spin_orbit` is zero because a Frescox `KIND=1` form
  factor is central only.
