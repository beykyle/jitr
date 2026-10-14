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
from scipy.integrate import solve_ivp

from jitr import lagrange
from jitr.reactions import Level, rotor
from jitr.utils.constants import HBARC
from jitr.utils.free_solutions import coulomb_hankel_table
from jitr.xs import coupled

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


class RotorCase:
    """Rigid-rotor coupled channels of Examples 3 and 4 at one J^pi.

    Woods-Saxon ``U = -(V + iW) f`` plus a uniformly charged sphere on the
    diagonal, and the first-order deformation coupling ``-beta R_c U'``
    times the rotor geometry.
    """

    def __init__(self, name, a, N, energies):
        d = self.d = DATA[name]
        self.mf = DATA["hbar2_over_amu"] / (d["Ap"] * d["At"] / (d["Ap"] + d["At"]))
        levels = [Level(I) for I, _ in d["levels"]]
        Ex = np.array([E for _, E in d["levels"]])
        k2 = (np.asarray(energies, float)[:, None] - Ex) / self.mf
        self.z12 = d["Zp"] * d["Zt"] * E2
        self.ws = coupled.Workspace(
            a,
            N,
            levels,
            d["J"] + 2 * d["lam"],
            k2,
            np.full_like(k2, HBARC**2 / (2 * self.mf)),
            self.z12 / (2 * self.mf * np.sqrt(np.abs(k2))),
            J=[d["J"]],
            parity=d["parity"],
        )
        self.geometry = rotor(self.ws.blocks, levels, d["lam"])
        diagonal, coupling = self.forms(self.ws.r)
        S, is_open = self.ws.smatrix(
            local=[(np.eye(self.ws.nch), diagonal), (self.geometry, coupling)]
        )
        self.S, self.open = np.asarray(S[:, 0]), np.asarray(is_open[:, 0])

    def forms(self, r):
        d = self.d
        x = np.exp((r - d["R"]) / d["a_ws"])
        depth = complex(d["V"], d["W"])
        Rc = d["R_coul"]
        coulomb = np.where(
            r <= Rc, self.z12 * (3 - (r / Rc) ** 2) / (2 * Rc), self.z12 / r
        )
        dU = depth * x / (d["a_ws"] * (1 + x) ** 2)
        return -depth / (1 + x) + coulomb, -d["beta"] * d["R_coupling"] * dU

    def first_column(self, e):
        n = int(self.open[e].sum())
        return np.abs(self.S[e, :n, 0]), np.angle(self.S[e, :n, 0]) / 2

    def integrate(self, e):
        """S from an independent solution of the same problem: integrate the
        regular solutions of the coupled radial equations outward to ``a``
        and match them to Coulomb-Hankel functions in open channels and to
        decaying Whittaker functions (log-derivative B_c) in closed ones."""
        ws = self.ws
        active = ws.blocks.active[0]
        l, nch, a = ws.blocks.l[0][active], int(active.sum()), ws.mesh.a
        k2 = np.asarray(ws.k2[e, 0])[active]
        C = self.geometry[0][np.ix_(active, active)]

        def rhs(r, y):
            u, du = y[: nch * nch].reshape(nch, nch), y[nch * nch :].reshape(nch, nch)
            diagonal, coupling = self.forms(np.array(r))
            U = (np.eye(nch) * diagonal + C * coupling) / self.mf
            d2u = (np.diag(l * (l + 1) / r**2 - k2) + U) @ u
            return np.concatenate([du.ravel(), d2u.ravel()])

        r0 = 1e-3
        u0 = np.diag(r0 ** (l + 1.0)).astype(complex)
        du0 = np.diag((l + 1.0) * r0**l).astype(complex)
        y = solve_ivp(
            rhs,
            (r0, a),
            np.concatenate([u0.ravel(), du0.ravel()]),
            rtol=1e-11,
            atol=1e-30,
            method="DOP853",
        ).y[:, -1]
        u, du = y[: nch * nch].reshape(nch, nch), y[nch * nch :].reshape(nch, nch)
        is_open = np.asarray(ws.open[e, 0])[active]
        H = np.asarray(ws.H[e, 0])[:, active]
        Bc = np.asarray(ws.Bc[e, 0])[active]
        k = np.sqrt(np.abs(k2))
        # unknowns: combination x of regular solutions, then S_c0 of open c
        n_open = int(is_open.sum())
        A = np.zeros((nch + n_open, nch + n_open), complex)
        rhs_vec = np.zeros(nch + n_open, complex)
        row = 0
        for c in range(nch):
            if is_open[c]:
                j = nch + int(is_open[:c].sum())
                for value, scale, h in ((u, k[c] ** -0.5, 0), (du, k[c] ** 0.5, 2)):
                    A[row, :nch] = value[c]
                    A[row, j] = 0.5j * scale * H[h, c]
                    rhs_vec[row] = 0.5j * scale * H[h + 1, c] * (c == 0)
                    row += 1
            else:
                A[row, :nch] = du[c] - Bc[c] / a * u[c]
                row += 1
        return np.linalg.solve(A, rhs_vec)[nch:]


@pytest.mark.parametrize("ref", DATA["ex3_o16_ca44"]["references"])
def test_example3_o16_ca44_rotor(ref):
    """16O + 44Ca (0+, 2+), J = 30+, four open channels."""
    case = RotorCase("ex3_o16_ca44", ref["a"], 100, ref["energies"])
    for e in range(len(ref["energies"])):
        amp, phase = case.first_column(e)
        np.testing.assert_allclose(amp, ref["amplitudes"][e], atol=5e-5)
        np.testing.assert_allclose(phase, ref["phases"][e], atol=5e-5)


def test_example4_alpha_c12_open_channels():
    """alpha + 12C (0+, 2+, 4+), J = 3-, at 16 and 20 MeV, where all eight
    channels are open."""
    ref = next(r for r in DATA["ex4_alpha_c12"]["references"] if r["a"] == 11.0)
    case = RotorCase("ex4_alpha_c12", 11.0, 80, ref["energies"][3:])
    for e in range(2):
        amp, phase = case.first_column(e)
        np.testing.assert_allclose(amp, ref["amplitudes"][3 + e], atol=5e-6)
        np.testing.assert_allclose(phase, ref["phases"][3 + e], atol=5e-4)


def test_example4_alpha_c12_closed_channels():
    """Below 14.08 MeV some channels of Example 4 are closed.

    The published values at 4, 8 and 12 MeV are not used: computed with
    subinterval propagation they depend on the partition into intervals (at
    4 MeV the same propagation gives 0.6153, 0.6125 or 0.6246 for different
    partitions, while all agree to 1e-6 at 16 MeV, where every channel is
    open). Instead the single-interval S is checked against an independent
    integration of the coupled equations, for convergence in N, and for
    independence of the channel radius once the potential has vanished.
    """
    E = [4.0, 8.0, 12.0]
    case = RotorCase("ex4_alpha_c12", 11.0, 80, E)
    finer = RotorCase("ex4_alpha_c12", 11.0, 120, E)
    np.testing.assert_allclose(case.S, finer.S, atol=1e-6)
    for e in range(len(E)):
        n = int(case.open[e].sum())
        np.testing.assert_allclose(case.S[e, :n, 0], case.integrate(e), atol=1e-6)
    near, far = (
        RotorCase("ex4_alpha_c12", 12.0, 110, E),
        RotorCase("ex4_alpha_c12", 14.0, 130, E),
    )
    np.testing.assert_allclose(near.S, far.S, atol=1e-4)
