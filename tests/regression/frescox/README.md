# Frescox regression references

The first committed Frescox regression case is adapted from the upstream
`B1-example-el` example published at <http://www.fresco.org.uk/examples/>.

## Landed case

- `F1_p_ni78_elastic`: proton elastic scattering on `78Ni` at `Elab = 6.9 MeV`
  from the first energy block in `B1-example-el.out`

## Regenerating the CSV

1. Download the upstream input and output files.
2. Run:

   ```bash
   uv run python tests/regression/frescox/tools/parse_frescox.py \
       --output tests/regression/frescox/B1-example-el.out \
       --metadata tests/regression/frescox/reference/F1_p_ni78_elastic.json \
       --csv-out tests/regression/frescox/reference/F1_p_ni78_elastic.csv \
       --case-index 0 \
       --min-angle-deg 1.0
   ```

The parser only reads the committed external output and never edits the
reference metadata by hand.

## Notes

- Frescox cases use the literal input-deck mass convention in `jitr`: each
  particle mass is set to `A * AMU`, and the builder uses classical
  kinematics. This matches the upstream examples more closely than the default
  tabulated mass model.
- B2 and B5 are not landed yet because this branch still lacks a public
  `jitr.xs.dwba` workspace.
