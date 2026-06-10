# TALYS regression references

Four [TALYS](https://github.com/arjankoning1/talys) 2.2 elastic-scattering
cases for `120Sn` at 10 MeV are committed and active in the manifest.  All
adapt samples from `talys/samples/n-Sn120-omp-JLM/`.

| Case | Projectile | `jlmmode` | Source input |
|------|-----------|-----------|-------------|
| T1   | n | 0 | `inputs/T1_jlm_elastic.inp` |
| T2   | n | 2 | `inputs/T2_jlmb_elastic.inp` |
| T3   | p | 0 | `inputs/T3_jlmb_proton_elastic.inp` |
| T4   | p | 2 | `inputs/T4_jlmb2_proton_elastic.inp` |

All inputs enable `outangle y` and `fileelastic y`; the parser reads the
`direct` column from the resulting `[np][np]0010.000ang.L00` YANDF file.

## Building TALYS locally

```bash
git clone --depth=1 https://github.com/arjankoning1/talys.git ${PATH_TO_TALYS}
cd ${PATH_TO_TALYS}
make
```

The resulting executable is `${PATH_TO_TALYS}/source/talys`.

```
export PATH="${PATH_TO_TALYS}/source:$PATH"
which talys
```

should print the path to the executable.

## Regenerating a CSV reference

From a temporary work directory, with `REPO_ROOT` pointing to the repo root:

```bash
REPO_ROOT=/path/to/jitr
CASE=T2_jlmb_elastic          # change for each case
OUTPUT=nn0010.000ang.L00       # nn* for neutron, pp* for proton

cp "$REPO_ROOT"/tests/regression/talys/inputs/${CASE}.inp .
cp "$REPO_ROOT"/tests/regression/talys/inputs/${CASE}.energies .
talys < ${CASE}.inp > talys.out

uv run python "$REPO_ROOT"/tests/regression/talys/tools/parse_talys.py \
    --output "$OUTPUT" \
    --metadata "$REPO_ROOT"/tests/regression/talys/reference/${CASE}.json \
    --csv-out  "$REPO_ROOT"/tests/regression/talys/reference/${CASE}.csv \
    --column direct
```

Substitute the appropriate `CASE` and `OUTPUT` filename for each of T1–T4.
