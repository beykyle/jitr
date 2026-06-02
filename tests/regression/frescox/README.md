# Frescox regression references

Cases F1–F8 are adapted from the upstream `B1-example-el` example published at
<http://www.fresco.org.uk/examples/> (F1–F3) and repo-local high-energy decks
(F4–F8).

## Building Frescox locally

```bash
mkdir -p /tmp/jitr-frescox
cd /tmp/jitr-frescox
git clone --depth=1 https://github.com/LLNL/Frescox.git
cd Frescox/source
make MACH=gfortran -j2
```

The resulting executable is `./frescox`.

## Regenerating the CSV references

**Upstream cases (F1–F3):** download the output and parse:

```bash
mkdir -p /tmp/jitr-frescox/runs
curl -fsSL http://www.fresco.org.uk/examples/B1-example-el.out \
   -o /tmp/jitr-frescox/runs/B1-example-el.out

uv run python tests/regression/frescox/tools/parse_frescox.py \
    --output /tmp/jitr-frescox/runs/B1-example-el.out \
    --metadata tests/regression/frescox/reference/F1_p_ni78_elastic.json \
    --csv-out tests/regression/frescox/reference/F1_p_ni78_elastic.csv \
    --case-index 0 --min-angle-deg 1.0
```

Repeat with `--case-index 1` / `F2_p_ni78_elastic_11MeV` and
`--case-index 2` / `F3_p_ni78_elastic_49p35MeV`.

**High-energy cases (F4–F8):** run Frescox first, then parse:

```bash
FRESCOX=/tmp/jitr-frescox/Frescox/source/frescox

$FRESCOX < tests/regression/frescox/inputs/B1-high-el.in \
         > tests/regression/frescox/outputs/B1-high-el.out

$FRESCOX < tests/regression/frescox/inputs/B1_n-high-el.in \
         > tests/regression/frescox/outputs/B1_n-high-el.out

uv run python tests/regression/frescox/tools/parse_frescox.py \
   --output tests/regression/frescox/outputs/B1-high-el.out \
   --metadata tests/regression/frescox/reference/F4_p_ni78_elastic_100MeV.json \
   --csv-out tests/regression/frescox/reference/F4_p_ni78_elastic_100MeV.csv \
   --case-index 0 --min-angle-deg 1.0

# repeat for F5 (--case-index 1), F6/F7/F8 from B1_n-high-el.out
```

## Notes

- Frescox's `elab`/`nlab` NAMELIST supports at most four energies per deck, so
  the high-energy proton and neutron ladders are split into separate
  `B1-high-el` and `B1_n-high-el` input decks.
- B2 and B5 are not landed yet (requires a public `jitr.xs.dwba` workspace).
