# TALYS regression data

This directory now carries the first committed TALYS elastic reference case:
`T2_jlmb_elastic`, a `120Sn(n,n)` JLMB run at 10 MeV.

## Current status

- The committed TALYS slice is **reference-only** for now: input deck, metadata,
  parser support, and normalized CSV are landed.
- The case is **not** in `tests/regression/manifest.json` yet because the repo
  still lacks the `jitr` builder path that evaluates JLM/JLMB elastic potentials
  end to end.
- The design note that `jlmomp` alone selects the JLM variant was stale. In the
  mirrored TALYS source, `jlmomp` enables the semi-microscopic OMP and
  `jlmmode` selects the JLMB-style imaginary normalization.

## Source sample

The committed input adapts the mirrored TALYS sample
`talys/samples/n-Sn120-omp-JLM/new/talys.inp` by:

- reducing the energy list to a single 10 MeV point,
- setting `jlmmode 2` for the stronger JLMB modification,
- enabling `outangle y` and `fileelastic y` so TALYS emits an elastic
  angular-distribution file with `xs`, `direct`, and `compound` columns.

## Regeneration

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

The parser intentionally reads the `direct` column from the YANDF angle file,
not `talys.out`, because the angular table is only emitted in the separate
`fileelastic` artifact.
