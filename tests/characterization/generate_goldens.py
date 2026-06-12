"""Generate the characterization goldens from the *current* engine.

Run once before the lax-core rewrite (design doc §6, Phase 1):

    uv run python tests/characterization/generate_goldens.py [case ...]

With no arguments, regenerates every golden in ``data/``.
"""

from __future__ import annotations

import sys

import numpy as np
from _cases import CASES, DATA_DIR


def main(argv: list[str]) -> None:
    names = argv or sorted(CASES)
    unknown = set(names) - set(CASES)
    if unknown:
        raise SystemExit(
            f"unknown case(s) {sorted(unknown)}; choose from {sorted(CASES)}"
        )
    DATA_DIR.mkdir(exist_ok=True)
    for name in names:
        print(f"generating {name} ...", flush=True)
        arrays = CASES[name]()
        np.savez_compressed(DATA_DIR / f"{name}.npz", **arrays)
        print(f"  wrote data/{name}.npz ({sorted(arrays)})")


if __name__ == "__main__":
    main(sys.argv[1:])
