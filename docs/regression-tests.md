# Regression tests

The `tests/regression/` package holds end-to-end regression tests that compare
`jitr` against committed outputs from independent reaction codes.

## Overview

The harness is intentionally small: one manifest, one loader, one builder
module, and one parametrised pytest entrypoint.  Only committed CSV/JSON
references are read during `pytest`; neither Frescox nor TALYS is required in
CI.

### Layout

- `tests/regression/manifest.json` — committed case inventory
- `tests/regression/_readers.py` — manifest and reference-data loader
- `tests/regression/_builders.py` — API-specific workspace/input assembly
- `tests/regression/test_regression.py` — end-to-end regression assertions
- `tests/regression/frescox/` — committed Frescox inputs, references, and
  parser
- `tests/regression/talys/` — TALYS inputs, references, parser, and
  provenance notes

### Running the regression suite

```bash
uv run pytest tests/regression/
```

The test file that drives the parametrised Frescox cases is
`test_regression.py`; TALYS cases live in `test_talys_reference.py`.

---

## Frescox

- regression tests are based on the examples published at <http://www.fresco.org.uk/examples/>.

### Landed cases

- `F1_p_ni78_elastic`: proton elastic scattering on `78Ni` at `Elab = 6.9 MeV`
  from the first energy block in `B1-example-el.out`
- `F2_p_ni78_elastic_11MeV`: proton elastic scattering on `78Ni` at
  `Elab = 11.0 MeV` from the second energy block in `B1-example-el.out`
- `F3_p_ni78_elastic_49p35MeV`: proton elastic scattering on `78Ni` at
  `Elab = 49.35 MeV` from the third energy block in `B1-example-el.out`
- `F4_p_ni78_elastic_100MeV` and `F5_p_ni78_elastic_200MeV`: proton elastic
  scattering on `78Ni` from the repo-local `inputs/B1-high-el.in` deck and its
  committed `outputs/B1-high-el.out`
- `F6_n_ni78_elastic_49p35MeV`, `F7_n_ni78_elastic_100MeV`, and
  `F8_n_ni78_elastic_200MeV`: neutron elastic scattering on `78Ni` from the
  repo-local `inputs/B1_n-high-el.in` deck and its committed
  `outputs/B1_n-high-el.out`

### Building Frescox locally

When Frescox is not already installed in the environment, the following build
works on this repository's Linux development setup:

```bash
mkdir -p /tmp/jitr-frescox
cd /tmp/jitr-frescox
git clone --depth=1 https://github.com/LLNL/Frescox.git
cd Frescox/source
make MACH=gfortran -j2
```

The resulting executable is `./frescox`.

### Regenerating Frescox CSV references

1. Download the upstream input and output files, for example:

   ```bash
   mkdir -p /tmp/jitr-frescox/runs
   curl -fsSL http://www.fresco.org.uk/examples/B1-example-el.out \
      -o /tmp/jitr-frescox/runs/B1-example-el.out
   ```

2. Run the parser:

   ```bash
   uv run python tests/regression/frescox/tools/parse_frescox.py \
       --output /tmp/jitr-frescox/runs/B1-example-el.out \
       --metadata tests/regression/frescox/reference/F1_p_ni78_elastic.json \
       --csv-out tests/regression/frescox/reference/F1_p_ni78_elastic.csv \
       --case-index 0 \
       --min-angle-deg 1.0
   ```

3. For the repo-local high-energy decks, run Frescox first:

   ```bash
   /tmp/jitr-frescox/Frescox/source/frescox \
      < tests/regression/frescox/inputs/B1-high-el.in \
      > tests/regression/frescox/outputs/B1-high-el.out

   /tmp/jitr-frescox/Frescox/source/frescox \
      < tests/regression/frescox/inputs/B1_n-high-el.in \
      > tests/regression/frescox/outputs/B1_n-high-el.out
   ```

4. Then parse the committed output blocks into CSV references:

   ```bash
   uv run python tests/regression/frescox/tools/parse_frescox.py \
      --output tests/regression/frescox/outputs/B1-high-el.out \
      --metadata tests/regression/frescox/reference/F4_p_ni78_elastic_100MeV.json \
      --csv-out tests/regression/frescox/reference/F4_p_ni78_elastic_100MeV.csv \
      --case-index 1 \
      --min-angle-deg 1.0

   uv run python tests/regression/frescox/tools/parse_frescox.py \
      --output tests/regression/frescox/outputs/B1_n-high-el.out \
      --metadata tests/regression/frescox/reference/F8_n_ni78_elastic_200MeV.json \
      --csv-out tests/regression/frescox/reference/F8_n_ni78_elastic_200MeV.csv \
      --case-index 2 \
      --min-angle-deg 1.0
   ```

### Notes

- Frescox cases use the literal input-deck mass convention in `jitr`: each
  particle mass is set to `A * AMU`, and the builder uses classical kinematics.
  This matches the upstream examples more closely than the default tabulated
  mass model.
- Frescox's `elab`/`nlab` NAMELIST in the current LLNL build only supports up
  to four energies in one deck, so the higher-energy proton/neutron ladders are
  split into dedicated `B1-high-el` and `B1_n-high-el` inputs instead of
  overloading `B1-example-el.in`.
- B2 and B5 are not landed yet because this branch still lacks a public
  `jitr.xs.dwba` workspace.

---

## TALYS

The TALYS slice currently carries the following elastic reference case:
`T2_jlmb_elastic`, a `120Sn(n,n)` calculation using the JLMB optical potential at 10 MeV.


### Source sample

The committed input adapts the TALYS sample
`talys/samples/n-Sn120-omp-JLM/new/talys.inp` by:

- reducing the energy list to a single 10 MeV point,
- setting `jlmmode 2` for the stronger JLMB modification,
- enabling `outangle y` and `fileelastic y` so TALYS emits an elastic
  angular-distribution file with `xs`, `direct`, and `compound` columns.

### Regenerating the TALYS CSV reference

From a temporary work directory that contains `T2_jlmb_elastic.inp` and
`T2_jlmb_elastic.energies`, with `REPO_ROOT` set to the repository root:

```bash
REPO_ROOT=/path/to/jitr
"$REPO_ROOT"/talys/bin/talys < T2_jlmb_elastic.inp > talys.out
uv run python "$REPO_ROOT"/tests/regression/talys/tools/parse_talys.py \
    --output nn0010.000ang.L00 \
    --metadata "$REPO_ROOT"/tests/regression/talys/reference/T2_jlmb_elastic.json \
    --csv-out "$REPO_ROOT"/tests/regression/talys/reference/T2_jlmb_elastic.csv \
    --column direct
```

The parser reads the `direct` column from the YANDF angle file, not
`talys.out`, because the angular table is only emitted in the separate
`fileelastic` artifact.
