"""Dispersion-kernel benchmark (design doc §4 exit criterion).

The JAX rewrite must be within 2× of the retired numba kernel for a single
evaluation and faster when batched. The numba timing below is the recorded
pre-rewrite baseline workflow: repeated `DispersionSolver.__call__` on a
typical DOM configuration. Not run in CI.

    uv run python scripts/bench_dispersion.py
"""

from __future__ import annotations

import time

import numpy as np

from jitr.optical_potentials.dispersion import DispersionSolver


def main() -> None:
    rgrid = np.linspace(0.1, 12.0, 120)
    solver = DispersionSolver(rgrid, 8.7)
    rng = np.random.default_rng(7)
    w_grid = rng.normal(size=(solver.n_radial, solver.n_nodes))
    w_at_e = rng.normal(size=solver.n_radial)

    solver(w_grid, w_at_e)  # warm-up / trace
    repeats = 2000
    start = time.perf_counter()
    for _ in range(repeats):
        solver(w_grid, w_at_e)
    per_eval = (time.perf_counter() - start) / repeats
    print(f"single eval: {per_eval * 1e6:9.1f} us")

    # batched: many parameter samples of W on the same solver
    samples = [rng.normal(size=w_grid.shape) for _ in range(64)]
    start = time.perf_counter()
    for sample in samples:
        solver(sample, w_at_e)
    batch = time.perf_counter() - start
    print(f"64-sample sweep: {batch * 1e3:9.2f} ms ({batch / 64 * 1e6:.1f} us/sample)")


if __name__ == "__main__":
    main()
