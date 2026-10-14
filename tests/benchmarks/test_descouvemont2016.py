"""Benchmarks from P. Descouvemont, Comput. Phys. Commun. 200 (2016) 199.

The published examples are solved with a single R-matrix interval at the
published channel radius ``a``; the reference values are the paper's output
tables (``data/descouvemont2016.json``). Units follow the paper: the
interaction enters through ``hbar^2 / 2 mu`` in MeV fm^2 and ``e^2 = 1.44``
MeV fm, so the symmetric mass factor of :func:`jitr.lagrange.solve` is
``M = 1 / (hbar^2 / 2 mu)``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from jitr import lagrange
from jitr.utils.free_solutions import coulomb_hankel_table

DATA = json.loads(
    (Path(__file__).parent / "data" / "descouvemont2016.json").read_text()
)
E2 = DATA["e2"]


def hankel(k, eta, a, l):
    return np.array(coulomb_hankel_table(k * a, eta, l)[:])[:, l]


def solve_open(mesh, l, mass_factor, E, z12, V):
    """All channels open at the same threshold; ``V`` is (nch, nch, N, N)."""
    l = np.atleast_1d(l)
    k = np.sqrt(E / mass_factor)
    eta = z12 * E2 / (2 * mass_factor * k)
    H = np.array([hankel(k, eta, mesh.a, li) for li in l]).T
    nch = l.size
    return np.asarray(
        lagrange.solve(
            mesh,
            l,
            np.full(nch, k**2),
            np.ones((nch, nch)) / mass_factor,
            V,
            H,
            np.zeros(nch),
            np.ones(nch, bool),
        )
    )


@pytest.mark.parametrize("ref", DATA["ex1_alpha_pb208"]["references"])
def test_example1_alpha_pb208(ref):
    """alpha + 208Pb, l = 20: Woods-Saxon optical potential plus point Coulomb."""
    d = DATA["ex1_alpha_pb208"]
    m = lagrange.mesh(ref["N"], ref["a"])
    r = m.r
    ws = 1 / (1 + np.exp((r - d["R"]) / d["a"]))
    v = -complex(d["V"], d["W"]) * ws + d["z1z2"] * E2 / r
    S = [
        solve_open(
            m, d["l"], d["hbar2_over_2mu"], E, d["z1z2"], np.diag(v)[None, None]
        )[0, 0]
        for E in ref["energies"]
    ]
    expected = np.array(ref["S_re"]) + 1j * np.array(ref["S_im"])
    np.testing.assert_allclose(S, expected, rtol=1e-4, atol=1e-4)


def reid_triplet(r):
    """Reid soft-core 3S1-3D1 central, tensor and spin-orbit terms in MeV."""
    h, x = 10.463, 0.7 * r
    e1, e2, e4, e6 = (np.exp(-n * x) / x for n in (1, 2, 4, 6))
    central = -h * e1 + 105.468 * e2 - 3187.8 * e4 + 9924.3 * e6
    tensor = (
        -h * (1 + 3 / x + 3 / x**2) * e1
        + (h * (12 / x + 3 / x**2) + 351.77) * e4
        - 1673.5 * e6
    )
    spin_orbit = 708.91 * e4 - 2713.1 * e6
    return central, tensor, spin_orbit


@pytest.mark.parametrize("ref", DATA["ex2_np_reid_j1"]["references"])
def test_example2_np_reid_j1(ref):
    """n-p, J = 1+: 3S1 and 3D1 coupled by the tensor force.

    In the (l=0, l=2) basis the operators are the identity (central),
    S12 = [[0, 2 sqrt 2], [2 sqrt 2, -2]] (tensor) and L.S = diag(0, -3).
    The paper tabulates the diagonal phases arg(S_ii)/2 and |S_12|. The
    potential is singular only as 1/r at the origin (the 1/r^3 tensor terms
    cancel), so the x-regularized mesh converges quickly; every published
    mesh is compared with one interval of N = 60 points at the same radius.
    """
    d = DATA["ex2_np_reid_j1"]
    m = lagrange.mesh(60, ref["a"])
    central, tensor, spin_orbit = reid_triplet(m.r)
    s8 = 2 * np.sqrt(2)
    couplings = [np.eye(2), np.array([[0, s8], [s8, -2]]), np.diag([0.0, -3.0])]
    V = sum(
        C[:, :, None, None] * np.diag(v)
        for C, v in zip(couplings, (central, tensor, spin_orbit), strict=True)
    )
    for i, E in enumerate(ref["energies"]):
        S = solve_open(m, d["l"], d["hbar2_over_2mu"], E, 0, V)
        np.testing.assert_allclose(np.angle(S[0, 0]) / 2, ref["phase_11"][i], atol=5e-5)
        np.testing.assert_allclose(np.angle(S[1, 1]) / 2, ref["phase_22"][i], atol=5e-5)
        np.testing.assert_allclose(abs(S[0, 1]), ref["eta_12"][i], atol=5e-5)


@pytest.mark.parametrize("ref", DATA["ex5_yamaguchi"]["references"])
def test_example5_yamaguchi_nonlocal(ref):
    """Separable nonlocal potential, l = 0. The a = 15 fm, N = 20 entry is the
    exact large-radius phase shift."""
    d = DATA["ex5_yamaguchi"]
    m = lagrange.mesh(ref["N"], ref["a"])
    r = m.r
    alpha, beta, mf = d["alpha"], d["beta"], d["hbar2_over_2mu"]
    kernel = -2 * beta * (alpha + beta) ** 2 * np.exp(-beta * np.add.outer(r, r)) * mf
    V = lagrange.interaction(m, nonlocal_=kernel)[None, None]
    delta = [
        np.degrees(np.angle(solve_open(m, 0, mf, E, 0, V)[0, 0]) / 2)
        for E in ref["energies"]
    ]
    np.testing.assert_allclose(delta, ref["delta_deg"], atol=ref["atol_deg"])
