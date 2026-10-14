# Regression tests

End-to-end tests compare `jitr` against committed outputs from Frescox and
TALYS. Reference CSVs are committed; neither external code is required in CI.

## Running

```bash
uv run pytest tests/regression/
```

## Frescox cases (F1–F10)

Ten cases against LLNL [Frescox](https://github.com/LLNL/Frescox): eight
elastic for p/n + `78Ni`, and two quasi-elastic `(p,n)` to the isobaric
analog state of `48Ca`.

| Case | Reaction | Projectile | E\_lab (MeV) | Source deck |
|------|----------|-----------|-------------|-------------|
| F1   | elastic | p | 6.9   | `B1-example-el.out` (block 1) |
| F2   | elastic | p | 11.0  | `B1-example-el.out` (block 2) |
| F3   | elastic | p | 49.35 | `B1-example-el.out` (block 3) |
| F4   | elastic | p | 100   | `B1-high-el.in` (block 1) |
| F5   | elastic | p | 200   | `B1-high-el.in` (block 2) |
| F6   | elastic | n | 49.35 | `B1_n-high-el.in` (block 1) |
| F7   | elastic | n | 100   | `B1_n-high-el.in` (block 2) |
| F8   | elastic | n | 200   | `B1_n-high-el.in` (block 3) |
| F9   | `48Ca(p,n)48Sc(IAS)` | p | 25 | `Ca48_pn_IAS_25MeV.in` |
| F10  | `48Ca(p,n)48Sc(IAS)` | p | 35 | `Ca48_pn_IAS_35MeV.in` |

F9 and F10 exercise `jitr.xs.quasielastic_pn` rather than
`jitr.xs.elastic`. Frescox does its own channel-spin algebra, so they are an
independent check on the spin-flip terms of the `(p,n)` amplitude, which the
elastic cases cannot see. jitR reproduces both to better than 1.3e-4 at every
angle.

## TALYS cases (T1–T4)

Four JLM/JLMB elastic-scattering cases against TALYS 2.2 for `120Sn` at
10 MeV:

| Case | Projectile | `jlmmode` | Notes |
|------|-----------|-----------|-------|
| T1   | n | 0 | standard JLMB normalization |
| T2   | n | 2 | stronger JLMB normalization |
| T3   | p | 0 | Coulomb code path |
| T4   | p | 2 | Coulomb + stronger normalization |

```{toctree}
:maxdepth: 1
:caption: Harness details

../tests/regression/README
../tests/regression/frescox/README
../tests/regression/talys/README
```
