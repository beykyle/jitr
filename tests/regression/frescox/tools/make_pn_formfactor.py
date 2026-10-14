"""Write the Lane (p,n) transition form factor read by the Frescox IAS decks.

The decks couple the p + 48Ca and n + 48Sc(IAS) partitions with a single local
``KIND=1`` form factor read from ``fort.4``. This writes that table, which is

    U1(r) = -(U_n^nuc(r) - U_p^nuc(r)) * sqrt(|N - Z|) / (N - Z - 1)

built from the same KD02 central potentials as the decks, with their Coulomb
and spin-orbit parts removed. That is exactly ``U1_central`` of
:mod:`jitr.xs.quasielastic_pn`, so the regression case compares the same
operator on both sides. A ``KIND=1`` form factor cannot carry an ``l.s`` term,
which is why the case passes ``U1_spin_orbit = 0``.

Two Frescox conventions are baked into the header:

``FSCALE = sqrt(2) * sqrt(4 pi)``
    For a local ``KIND=1`` form factor with ``IP3=0``, ``INTER`` scales the
    table by ``ASCALE = FSCALE * R4PI`` with ``R4PI = 1/sqrt(4 pi)``
    (``frxx7a.f``, ``globx7.f``), so the ``sqrt(4 pi)`` cancels ``R4PI``. The
    remaining ``sqrt(2)`` is ``sqrt(2 j_p + 1)`` for the spin-1/2 projectile:
    Frescox reads the table as a reduced matrix element, and the coupling
    coefficient it multiplies (``frxx4.f``, ``IP3=0`` branch) is exactly
    ``1/sqrt(2)`` for every ``(l, j)`` here. The two cancel, so Frescox's
    matrix element is the plain ``U1`` and jitr needs no such factor.

``LOP = DER = -1``, written explicitly
    Frescox first reads the header expecting these two integers and only falls
    back to the shorter form on an I/O error. That fallback re-reads after the
    failed record, consuming the first data line, and the form factor is then
    dropped **silently**: the (p,n) cross section comes out identically zero
    with no error message. Writing them explicitly avoids the fallback.

Usage::

    uv run python tests/regression/frescox/tools/make_pn_formfactor.py \\
        --metadata tests/regression/frescox/reference/F9_p_ca48_pn_ias_25MeV.json \\
        --out tests/regression/frescox/inputs/Ca48_pn_IAS_25MeV.formfactor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FSCALE = np.sqrt(2.0) * np.sqrt(4.0 * np.pi)
LTR = PTR = TTR = 0
IB = IA = 1
LOP = DER = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rmax-fm", type=float, default=20.0)
    parser.add_argument("--step-fm", type=float, default=0.02)
    return parser.parse_args()


def woods_saxon(r: np.ndarray, R: float, a: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((r - R) / a))


def woods_saxon_derivative(r: np.ndarray, R: float, a: float) -> np.ndarray:
    x = np.exp((r - R) / a)
    return -x / (a * (1.0 + x) ** 2)


def central_potential(r: np.ndarray, p: dict[str, float], A: int) -> np.ndarray:
    """Nuclear central KD02 potential: volume real, volume and surface imaginary."""
    A13 = A ** (1.0 / 3.0)
    return (
        -p["V"] * woods_saxon(r, p["rv"] * A13, p["av"])
        - 1j * p["W"] * woods_saxon(r, p["rw"] * A13, p["aw"])
        - 1j
        * p["Wd"]
        * (-4.0 * p["avd"])
        * woods_saxon_derivative(r, p["rvd"] * A13, p["avd"])
    )


def transition_potential(metadata: dict, r: np.ndarray) -> np.ndarray:
    potential = metadata["optical_potential"]
    A = int(metadata["reaction"]["target"]["A"])
    Z = int(metadata["reaction"]["target"]["Z"])
    N = A - Z
    isovector_factor = np.sqrt(abs(N - Z)) / (N - Z - 1)
    U_p = central_potential(r, potential["proton"], A)
    U_n = central_potential(r, potential["neutron"], A)
    return -(U_n - U_p) * isovector_factor


def write_formfactor(path: Path, U1: np.ndarray, step_fm: float) -> None:
    header = (
        f"{U1.size:4d}{step_fm:8.4f}{0.0:8.4f}{FSCALE:8.4f}"
        f"{LTR:4d}{PTR:4.0f}{TTR:4.0f}{IB:4d}{IA:4d}{LOP:4d}{DER:4d}"
        "Lane U1 central"
    )
    rows = [f"{value.real: .10e} {value.imag: .10e}" for value in U1]
    path.write_text("\n".join([header, *rows]) + "\n")


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    r = np.arange(0.0, args.rmax_fm + 0.5 * args.step_fm, args.step_fm)
    write_formfactor(args.out, transition_potential(metadata, r), args.step_fm)
    print(f"wrote {args.out} ({r.size} points to {r[-1]:.2f} fm)")


if __name__ == "__main__":
    main()
