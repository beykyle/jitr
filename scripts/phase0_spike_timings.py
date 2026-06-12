"""Phase 0 spike timings (design doc §8 Q3): eig-spectral vs linear_solve.

Times compile (mpmath-dominated for charged channels) and per-evaluation cost
of the full blocked S-matrix pipeline for N_E ∈ {1, 8, 64}, lmax ∈ {15, 20}.
Not run in CI.

    uv run python scripts/phase0_spike_timings.py
"""

from __future__ import annotations

import time

import lax
import numpy as np

NBASIS = 40
RADIUS = 12.0
MASS_FACTOR = 22.0  # ~ ħ²/2μ for p + 48Ca, MeV·fm²


def build(lmax: int, n_e: int, method: str) -> tuple[lax.Solver, object]:
    energies = np.linspace(8.0, 50.0, n_e)
    t0 = time.perf_counter()
    solver = lax.compile(
        mesh=lax.MeshSpec("legendre", "x", n=NBASIS, scale=RADIUS),
        blocks=[
            [lax.ChannelSpec(l=ell, threshold=0.0, mass_factor=MASS_FACTOR)]
            for ell in range(lmax + 1)
        ],
        solvers=(
            ("rmatrix_direct",) if method == "linear_solve" else ("spectrum", "smatrix")
        ),
        energies=energies,
        z1z2=(1, 20),
        V_is_complex=True,
        method=method,
    )
    compile_s = time.perf_counter() - t0
    rgrid = np.asarray(solver.mesh.radii)
    v = solver.interaction_from_array(
        local=[(-45.0 - 4.0j) * np.exp(-((rgrid / 4.0) ** 2))]
    )
    return solver, (compile_s, v)


def time_eval(solver: lax.Solver, v, method: str, repeats: int = 5) -> float:
    import jax

    def run():
        if method == "linear_solve":
            return solver.smatrix_direct(v)
        return solver.smatrix(solver.spectrum(v))

    out = run()
    jax.block_until_ready(out)  # warm-up / trace
    t0 = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(run())
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    print(
        f"{'lmax':>5} {'N_E':>4} {'method':>13} {'compile [s]':>12} {'eval [ms]':>10}"
    )
    for lmax in (15, 20):
        for n_e in (1, 8, 64):
            for method in ("eig", "linear_solve"):
                solver, (compile_s, v) = build(lmax, n_e, method)
                eval_s = time_eval(solver, v, method)
                print(
                    f"{lmax:>5} {n_e:>4} {method:>13} {compile_s:>12.2f} "
                    f"{eval_s * 1e3:>10.2f}"
                )


if __name__ == "__main__":
    main()
