# TALYS regression scaffolding

The TALYS elastic cases from the design are not landed yet.

## Current blocker

This branch does not currently expose a public JLM/JLMB elastic builder under
`jitr.xs.elastic`, so TALYS-backed reference cases cannot run end-to-end yet.

## Candidate upstream sample

The most promising upstream sample discovered during implementation is
`samples/n-Sn120-omp-JLM`, which already exercises TALYS with `jlmomp y` and
ships both input and reference output directories in the TALYS repository.

## Next step

Once JLM/JLMB support exists in `jitr`, clone the upstream TALYS input into this
directory, generate a `direct`-column CSV with `tools/parse_talys.py`, and add
T1/T2 manifest entries.
