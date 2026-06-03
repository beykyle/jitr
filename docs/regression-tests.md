# Regression tests

End-to-end tests compare `jitr` against committed outputs from Frescox and
TALYS. Reference CSVs are committed; neither external code is required in CI.

## Running

```bash
uv run pytest tests/regression/
```

## Frescox cases (F1–F8)

Eight elastic-scattering cases against LLNL
[Frescox](https://github.com/LLNL/Frescox), all for p/n + `78Ni`:

| Case | Projectile | E\_lab (MeV) | Source deck |
|------|-----------|-------------|-------------|
| F1   | p | 6.9   | `B1-example-el.out` (block 1) |
| F2   | p | 11.0  | `B1-example-el.out` (block 2) |
| F3   | p | 49.35 | `B1-example-el.out` (block 3) |
| F4   | p | 100   | `B1-high-el.in` (block 1) |
| F5   | p | 200   | `B1-high-el.in` (block 2) |
| F6   | n | 49.35 | `B1_n-high-el.in` (block 1) |
| F7   | n | 100   | `B1_n-high-el.in` (block 2) |
| F8   | n | 200   | `B1_n-high-el.in` (block 3) |

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
