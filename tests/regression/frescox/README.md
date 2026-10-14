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
   --case-index 1 --min-angle-deg 1.0

# repeat for F5 (B1-high-el.out, --case-index 2) and F6/F7/F8
# (B1_n-high-el.out, --case-index 0/1/2)
```

Both decks run `elab = 49.35, 100, 200` MeV, so the cross-section blocks are
indexed 0, 1, 2 in that order; each case's `reference_run.case_index` records
its block. Block 0 of `B1-high-el.out` is unused: it repeats F3 (49.35 MeV
protons) and agrees with the upstream F3 output to 0.2%.

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

## Spin-1 tensor analyzing powers (F11–F14)

Deuteron elastic scattering on `58Ni` at 22 and 56 MeV, from two repo-local
decks that differ only in the projectile spin-orbit term:

- `d_ni58-tensor-el.in` (`Vso = 6` MeV): F11 (22 MeV, block 0) and F12
  (56 MeV, block 1), with `dsdo` and the tensor analyzing powers
  `iT11, T20, T21, T22`.
- `d_ni58-noso-el.in` (`Vso = 0`): F13 and F14, `dsdo` only. With no spin
  dependence `dsdo` is the same for any projectile spin, so these pass against
  today's spin-0 treatment of the deuteron and pin the deck conventions
  (surface absorption, `r At^{1/3}` radii from `ap=0`, Coulomb) independently
  of the spin-1 observables.

**F11 and F12 are staged, not yet in `manifest.json`.** Their CSVs carry four
extra columns, which `_readers.load_case` does not accept yet, and jitR has no
spin-1 elastic observables. Register them once both land; the per-observable
tolerances (`atol 5e-3` on each `T_kq`, the 5-decimal precision Frescox prints
being the floor) are already in the metadata.

Regenerate with (Frescox writes `fort.*` files to the working directory):

```bash
FRESCOX=/tmp/jitr-frescox/Frescox/source/frescox
for deck in tensor noso; do
  $FRESCOX < tests/regression/frescox/inputs/d_ni58-$deck-el.in \
           > tests/regression/frescox/outputs/d_ni58-$deck-el.out
done

P=tests/regression/frescox
uv run python $P/tools/parse_frescox.py --output $P/outputs/d_ni58-tensor-el.out \
    --metadata $P/reference/F11_d_ni58_elastic_tensor_22MeV.json \
    --csv-out $P/reference/F11_d_ni58_elastic_tensor_22MeV.csv \
    --case-index 0 --min-angle-deg 1.0 --polarizations
# F12: --case-index 1; F13/F14: d_ni58-noso-el.out without --polarizations
```

The references were made with Frescox `7.2-20-ga7f491` (gfortran).

### What Frescox computes

`kqmax=2` asks for analyzing powers up to rank 2, `pp=0` for the projectile
(beam polarization), and `koords=0` for the Madison frame (z along the beam,
y along `k_in × k_out`). Each angle line ends with
`& pols = T10 iT11 T20 T21 T22`; `T10` is zero for elastic scattering, and
`--polarizations` checks that as a format guard. With `M_{m' m}(θ)` the
amplitude from projectile state `m` to `m'` (Coulomb included on the
diagonal), `frxx4.f` (subroutine `CRISS`) evaluates

```
T_kq = sqrt(2k+1) Σ_{m,m'} M_{m', m+q} M*_{m', m} <s m k q | s m+q> / Σ_{m,m'} |M_{m' m}|^2
```

times `i` for odd `k` (so `iT11` is printed as a real number). For spin 1 the
vector analyzing power is `A_y = (2/√3) iT11`.

### Spin-orbit convention

Frescox `TYPE=3`, shape 0, is `-CONLS P1 E / [(1+E)^2 a r]` with
`CONLS = 2.000`, multiplied by `j(j+1) - l(l+1) - s(s+1) = 2 l·s` for any
projectile spin. That is jitR's Thomas form (`omp.spin_orbit`) acting through
`Potential.spin_orbit`, which also multiplies `2 l·s`, so `Vso` carries over
unchanged from spin 1/2 to spin 1.

### Step size

The decks use `hcm=0.02`. At `hcm=0.05` Frescox's 56 MeV cross section in
the deep minimum near 168° (`~5e-3` mb/sr) is 0.8% low, enough to fail F14
at `rtol 5e-3`; at `hcm=0.02` jitR and Frescox agree to 0.3% at every angle,
and the `T_kq` move by at most `2e-4` between the two step sizes.

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
