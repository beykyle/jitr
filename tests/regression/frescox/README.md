# Frescox regression references

Cases F1–F8 are adapted from the upstream `B1-example-el` example published at
<http://www.fresco.org.uk/examples/> (F1–F3) and repo-local high-energy decks
(F4–F8). Cases F9–F10 are quasi-elastic `48Ca(p,n)48Sc(IAS)` DWBA decks
contributed by Jin Lei (Tongji University), re-run locally.

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

## Regenerating the (p,n) IAS cases (F9–F10)

The deck reads its transition potential from `fort.4` in the working
directory. Write it from the KD02 parameters in the case metadata, run the
deck, and parse the outgoing-neutron block:

```bash
FRESCOX=/tmp/jitr-frescox/Frescox/source/frescox

for E in 25 35; do
  CASE=$([ "$E" = 25 ] && echo F9 || echo F10)_p_ca48_pn_ias_${E}MeV

  uv run python tests/regression/frescox/tools/make_pn_formfactor.py \
      --metadata tests/regression/frescox/reference/$CASE.json \
      --out tests/regression/frescox/inputs/Ca48_pn_IAS_${E}MeV.formfactor

  workdir=$(mktemp -d)
  cp tests/regression/frescox/inputs/Ca48_pn_IAS_${E}MeV.formfactor $workdir/fort.4
  (cd $workdir && $FRESCOX) \
      < tests/regression/frescox/inputs/Ca48_pn_IAS_${E}MeV.in \
      > tests/regression/frescox/outputs/Ca48_pn_IAS_${E}MeV.out

  uv run python tests/regression/frescox/tools/parse_frescox.py \
      --output tests/regression/frescox/outputs/Ca48_pn_IAS_${E}MeV.out \
      --metadata tests/regression/frescox/reference/$CASE.json \
      --csv-out tests/regression/frescox/reference/$CASE.csv \
      --case-index 1 --min-angle-deg 1.0
done
```

`--case-index 1` selects the outgoing-neutron partition; block 0 is proton
elastic.

### Two Frescox conventions worth knowing

**A malformed form-factor header fails silently.** Frescox first reads the
`fort.4` header expecting trailing `LOP` and `DER` integers, and only falls
back to the shorter historical header on an I/O error. That fallback re-reads
after the failed record, swallowing the first data line, and the form factor
is then dropped with no diagnostic: the `(p,n)` cross section comes out
identically zero. `make_pn_formfactor.py` always writes `LOP = DER = -1`
explicitly. If a charge-exchange deck returns exactly zero at every angle,
suspect this first.

**`FSCALE = sqrt(2) * sqrt(4 pi)` is not a fudge factor.** For a local
`KIND=1` form factor with `IP3=0`, `INTER` scales the table by
`ASCALE = FSCALE * R4PI` with `R4PI = 1/sqrt(4 pi)` (`frxx7a.f`, `globx7.f`),
so the `sqrt(4 pi)` cancels `R4PI`. The remaining `sqrt(2)` is
`sqrt(2 j_p + 1)` for the spin-1/2 projectile: Frescox reads the table as a
reduced matrix element, and the coupling coefficient it multiplies
(`frxx4.f`, `IP3=0` branch) evaluates to exactly `1/sqrt(2)` for every
`(l, j)` in this deck. The two cancel, so Frescox's matrix element is the
plain `U1` and jitR applies no such factor. This was verified two ways: by
evaluating that coupling coefficient symbolically, and by jitR reproducing
the Frescox cross section absolutely.

## Notes

- Frescox's `elab`/`nlab` NAMELIST supports at most four energies per deck, so
  the high-energy proton and neutron ladders are split into separate
  `B1-high-el` and `B1_n-high-el` input decks.
- B2 and B5 are not landed yet (requires a public `jitr.xs.dwba` workspace).
- F9/F10 guard the spin-flip terms of the `(p,n)` amplitude: with the
  Clebsch-Gordan arguments in their pre-fix order, jitR falls below the
  Frescox reference by up to a factor 15 at 25 MeV and 84 at 35 MeV in the
  backward hemisphere, and both cases fail.
- Our run of the F9/F10 decks reproduces the angular distributions supplied
  with them (`fresco_dsdo.dat`, FRES 3.4 on macOS/ARM) to a ratio of
  1.000000 at every angle, so the two Frescox lineages agree exactly here.
