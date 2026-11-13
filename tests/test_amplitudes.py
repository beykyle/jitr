"""General amplitudes and observables for any projectile and target spins.

Conventions are pinned by validated numbers (the elastic and (p,n) goldens go
through this path) and by the independent checks below: a plane-wave Born
limit for transitions with L' != L, exact identities of the analyzing powers,
the optical theorem, flux conservation, and Thompson's own rotor formula.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from jitr.optical_potentials.potential_forms import thomas_safe, woods_saxon_safe
from jitr.reactions import Level, Reaction, jpi_blocks, rotor, spin_orbit
from jitr.reactions.system import _wigner
from jitr.utils.constants import HBARC
from jitr.xs import amplitudes, coupled
from jitr.xs.amplitudes import clebsch_gordan, kq_labels
from jitr.xs.coupled import Potential, potential_terms
from jitr.xs.elastic import Workspace as ElasticWorkspace

ANGLES = np.linspace(0.02, np.pi - 0.02, 120)
MF = 20.736 / 3.0  # hbar^2 / 2 mu in MeV fm^2


def neutral_workspace(levels, E, Ex, lmax=16, a=12.0, N=60, **kw):
    """Uncharged projectile on the levels, one energy, hbar^2/2mu = MF."""
    k2 = (np.atleast_1d(E)[:, None] - np.asarray(Ex)) / MF
    mu = np.full_like(k2, HBARC**2 / (2 * MF))
    return coupled.Workspace(a, N, levels, lmax, k2, mu, 0 * k2, **kw)


def test_plane_wave_born_limit_of_rotor_excitation():
    """0+ -> 2+ through a weak coupling eps F(r) Y_2.Q_2 and no other
    potential. In first order ⟨2M'|V|00⟩ = eps F(r) Y*_2M'(r̂)/sqrt5, so
    dσ/dΩ = (k'/k) eps^2 I(q)^2 / (4π mf^2) with I(q) = ∫ j_2(qr) F r^2 dr and
    q^2 = k^2 + k'^2 - 2kk' cos θ. This pins transitions with L' != L: the
    i^{l'-l} basis, the flux normalization and the CG/Y conventions."""
    levels, E, Ex, eps = [Level(0), Level(2)], 10.0, 1.5, 1e-3
    ws = neutral_workspace(levels, E, [0.0, Ex])
    F = np.exp(-((ws.r / 2.0) ** 2))
    S, _ = ws.smatrix(local=[(rotor(ws.blocks, levels, 2), eps * F)])
    xs = amplitudes.observables(
        S,
        amplitudes.transition_tables(
            ws, amplitudes.transition_geometry(ws, 0, 1), ANGLES
        ),
    )
    k, kp = np.sqrt(E / MF), np.sqrt((E - Ex) / MF)
    q = np.sqrt(k**2 + kp**2 - 2 * k * kp * np.cos(ANGLES))
    integral = [
        quad(
            lambda r, q=q: spherical_jn(2, q * r) * np.exp(-((r / 2) ** 2)) * r**2,
            0,
            12,
        )[0]
        for q in q
    ]
    born = 10 * (kp / k) * eps**2 * np.array(integral) ** 2 / (4 * np.pi * MF**2)
    np.testing.assert_allclose(xs.dsdo[0], born, rtol=1e-4, atol=1e-6 * born.max())
    sigma = 2 * np.pi * np.trapezoid(born * np.sin(ANGLES), ANGLES)
    np.testing.assert_allclose(xs.sigma[0], sigma, rtol=2e-3)


def test_spin_half_ay_is_sqrt2_iT11():
    p = Reaction(target=(40, 20), projectile=(1, 1), process="El")
    ws = ElasticWorkspace(
        p, p.kinematics(np.array([15.0, 40.0])), 12.0, 20, 40, angles=ANGLES
    )
    r = ws.r
    xs = ws.xs(
        Potential(
            central=(-48 - 9j) * woods_saxon_safe(r, 4.2, 0.65),
            spin_orbit=(5.5 - 0.5j) * thomas_safe(r, 4.0, 0.6),
            coulomb=20 * 1.44 / np.maximum(r, 4.4),
        )
    )
    assert kq_labels(0.5) == [(1, 0), (1, 1)]
    np.testing.assert_allclose(xs.Ay, np.sqrt(2) * np.real(1j * xs.T[:, 1]), atol=1e-12)
    np.testing.assert_allclose(xs.T[:, 0], 0, atol=1e-12)


def elastic_observables(s, central, so_strength=0.0, E=12.0, angles=ANGLES):
    """Neutral spin-s projectile on 0+: observables of the elastic channel."""
    levels = [Level(0, 1, s)]
    ws = neutral_workspace(levels, E, [0.0], lmax=14)
    r = ws.r
    pot = Potential(
        central=central(r),
        spin_orbit=so_strength * thomas_safe(r, 3.8, 0.6) if s else None,
    )
    S, _ = ws.smatrix(
        *potential_terms(pot, np.eye(ws.nch), spin_orbit(ws.blocks, levels))
    )
    xs = amplitudes.observables(
        S,
        amplitudes.transition_tables(
            ws, amplitudes.transition_geometry(ws, 0, 0), angles
        ),
    )
    return ws, S, xs, amplitudes.integrated_cross_sections(ws, S)


def central(r):
    return (-45 - 8j) * woods_saxon_safe(r, 4.0, 0.6)


@pytest.mark.parametrize("s", [0.5, 1.0, 1.5])
def test_spin_independent_interaction(s):
    """Without spin-dependent forces the spin is a spectator: every T_kq
    vanishes and the cross sections are those of a spin-0 projectile."""
    _, _, xs0, (t0, r0) = elastic_observables(0.0, central)
    _, _, xs, (t, r) = elastic_observables(s, central)
    assert xs.T.shape[1] == len(kq_labels(s))
    np.testing.assert_allclose(xs.T, 0, atol=1e-10)
    np.testing.assert_allclose(
        xs.dsdo, xs0.dsdo, rtol=1e-10, atol=1e-12 * xs0.dsdo.max()
    )
    np.testing.assert_allclose(r, r0, rtol=1e-10)
    np.testing.assert_allclose(t, t0, rtol=1e-10)


@pytest.mark.parametrize("s", [0.5, 1.0])
def test_optical_theorem(s):
    """σ_T = (4π / k) Im Tr f(0) / (2s+1) for a neutral projectile."""
    angles = np.array([0.0, 0.3])
    _, _, xs, (t, _) = elastic_observables(s, central, so_strength=6.0, angles=angles)
    k = np.sqrt(12.0 / MF)
    forward = np.trace(np.asarray(xs.f)[0, :, :, 0])
    np.testing.assert_allclose(
        t[0], 10 * 4 * np.pi / k * forward.imag / (2 * s + 1), rtol=1e-10
    )


@pytest.mark.parametrize("s", [1.0, 1.5])
def test_analyzing_power_symmetries(s):
    """Reflection in the scattering plane: T_kq is real for even k and
    imaginary for odd k, and T_k0 vanishes for odd k."""
    _, _, xs, _ = elastic_observables(s, central, so_strength=6.0)
    T = np.asarray(xs.T)
    scale = np.abs(T).max()
    assert scale > 1e-2
    for n, (k, q) in enumerate(kq_labels(s)):
        if k % 2:
            np.testing.assert_allclose(T[:, n].real, 0, atol=1e-12 * scale)
            if q == 0:
                np.testing.assert_allclose(T[:, n], 0, atol=1e-12 * scale)
        else:
            np.testing.assert_allclose(T[:, n].imag, 0, atol=1e-12 * scale)


def rotor_case(E, s=0.0, real=False):
    """A light rotor (0+, 2+, 4+) at one energy with deformation coupling."""
    levels = [Level(0, 1, s), Level(2, 1, s), Level(4, 1, s)]
    ws = neutral_workspace(levels, E, [0.0, 1.5, 4.0], lmax=14)
    x = np.exp((ws.r - 4.0) / 0.5)
    depth = 60.0 + (0 if real else 12j)
    U, dU = -depth / (1 + x), depth * x / (0.5 * (1 + x) ** 2)
    S, _ = ws.smatrix(
        local=[(np.eye(ws.nch), U), (rotor(ws.blocks, levels, 2), -0.3 * 3.0 * dU)]
    )
    return ws, S


def test_flux_conservation_and_integrated_cross_sections():
    """With real potentials and every channel open, the non-elastic cross
    section is the sum of the inelastic ones, and each integrated cross
    section is the angular integral of its dσ/dΩ."""
    angles = np.linspace(1e-3, np.pi - 1e-3, 721)
    ws, S = rotor_case(12.0, real=True)
    _, rxn = amplitudes.integrated_cross_sections(ws, S)
    inelastic = 0
    for exit in (1, 2):
        xs = amplitudes.observables(
            S,
            amplitudes.transition_tables(
                ws, amplitudes.transition_geometry(ws, 0, exit), angles
            ),
        )
        integral = 2 * np.pi * np.trapezoid(xs.dsdo[0] * np.sin(angles), angles)
        np.testing.assert_allclose(integral, xs.sigma[0], rtol=1e-4)
        inelastic += xs.sigma[0]
    np.testing.assert_allclose(rxn[0], inelastic, rtol=1e-9)


def test_closed_exit_level():
    """A closed exit level has zero cross section and finite T_kq."""
    ws, S = rotor_case(3.0, s=1.0)  # the 4+ level at 4 MeV is closed
    xs = amplitudes.observables(
        S,
        amplitudes.transition_tables(
            ws, amplitudes.transition_geometry(ws, 0, 2), ANGLES
        ),
    )
    np.testing.assert_array_equal(xs.dsdo, 0)
    assert np.all(np.isfinite(xs.T))


def test_padded_and_closed_slots_do_not_contribute():
    """Elastic observables of a spin-1/2 projectile are unchanged by an
    uncoupled closed level, which pads every block and adds closed slots."""
    _, _, xs, (t, r) = elastic_observables(0.5, central, so_strength=6.0)
    levels = [Level(0, 1, 0.5), Level(2, 1, 0.5)]
    ws = neutral_workspace(levels, 12.0, [0.0, 30.0], lmax=14)
    rr = ws.r
    pot = Potential(central=central(rr), spin_orbit=6.0 * thomas_safe(rr, 3.8, 0.6))
    S2, _ = ws.smatrix(
        *potential_terms(pot, np.eye(ws.nch), spin_orbit(ws.blocks, levels))
    )
    xs2 = amplitudes.observables(
        S2,
        amplitudes.transition_tables(
            ws, amplitudes.transition_geometry(ws, 0, 0), ANGLES
        ),
    )
    t2, r2 = amplitudes.integrated_cross_sections(ws, S2)
    np.testing.assert_allclose(xs2.dsdo, xs.dsdo, rtol=1e-10)
    np.testing.assert_allclose(xs2.T, xs.T, atol=1e-12)
    np.testing.assert_allclose([t2, r2], [t, r], rtol=1e-10)


def racah(a, b, c, d, e, f):
    """Racah W(abcd; ef) = (-1)^{a+b+c+d} {a b e; d c f} (zero off triangle)."""
    d2 = [round(2 * x) for x in (a, b, e, d, c, f)]
    return (-1) ** round(a + b + c + d) * _wigner("6j", *d2)


def thompson_rotor(blocks, levels, lam, b):
    """Thompson Eqs. 48-49 for target excitation (K = 0), with the nuclear
    form factor V^N_λ replaced by 1/sqrt(4π) (its geometry):
    ⟨(L s)j, I; J|V|(L' s)j', I'; J⟩ = (-1)^{j-j'-L+L'} ĵ ĵ' Σ_J2 (2J2+1)
    W(s L J I; j J2) W(s L' J I'; j' J2) X^{J2 λ}_{LI:L'I'},
    X^{Jλ}_{LI:L'I'} = L̂ L̂' (-1)^{J-I'-L+L'} W(L L' I I'; λ J)
    ⟨L0L'0|λ0⟩ Î'⟨I'0λ0|I0⟩."""
    h = lambda x: np.sqrt(2 * x + 1)  # noqa: E731
    l, j, lv, J = blocks.l[b], blocks.j[b], blocks.level[b], blocks.J[b]
    n = int(blocks.active[b].sum())
    out = np.zeros((n, n))
    for c in range(n):
        for d in range(n):
            L, Lp, jj, jp = l[c], l[d], j[c], j[d]
            I, Ip, s = levels[lv[c]].I, levels[lv[d]].I, levels[lv[c]].s

            def X(J2, L=L, Lp=Lp, I=I, Ip=Ip):  # noqa: E741
                return (
                    h(L)
                    * h(Lp)
                    * (-1) ** round(J2 - Ip - L + Lp)
                    * (racah(L, Lp, I, Ip, lam, J2))
                    * clebsch_gordan(L, 0, Lp, 0, lam, 0)
                    * h(Ip)
                    * clebsch_gordan(Ip, 0, lam, 0, I, 0)
                )

            J2s = np.arange(abs(J - s), J + s + 1)
            out[c, d] = (
                (-1) ** round(jj - jp - L + Lp)
                * h(jj)
                * h(jp)
                * sum(
                    (2 * J2 + 1)
                    * (racah(s, L, J, I, jj, J2))
                    * (racah(s, Lp, J, Ip, jp, J2))
                    * X(J2)
                    for J2 in J2s
                )
                / np.sqrt(4 * np.pi)
            )
    return out


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_rotor_matches_thompson(s):
    """`rotor` equals i^{L'-L} times Thompson's target-excitation coupling,
    with no free constant."""
    levels = [Level(0, 1, s), Level(2, 1, s), Level(4, 1, s)]
    blocks = jpi_blocks(levels, 10)
    C = rotor(blocks, levels, 2)
    for b in range(0, blocks.l.shape[0], 3):
        n = int(blocks.active[b].sum())
        l = blocks.l[b][:n]
        phase = (1j ** (l[None, :] - l[:, None])).real
        np.testing.assert_allclose(
            C[b][:n, :n], phase * thompson_rotor(blocks, levels, 2, b), atol=1e-12
        )


def test_transition_without_channel_pairs_is_rejected():
    """Keeping only J^pi = 0+, the 4- level would need l = 4 and so parity
    -1: no block holds channels of both levels."""
    levels = [Level(0), Level(4, -1)]
    ws = neutral_workspace(levels, 12.0, [0.0, 4.0], J=[0], parity=1)
    with pytest.raises(ValueError, match="no block"):
        amplitudes.transition_geometry(ws, 0, 1)


def test_closed_charged_entrance_has_no_coulomb_amplitude():
    """Elastic scattering in a level that is closed at this energy is zero,
    including the Coulomb amplitude."""
    levels = [Level(0), Level(2)]
    k2 = np.array([[12.0 / MF, (12.0 - 20.0) / MF]])
    mu = np.full_like(k2, HBARC**2 / (2 * MF))
    eta = np.array([[2.0, 3.0]])
    ws = coupled.Workspace(12.0, 40, levels, 6, k2, mu, eta)
    S, _ = ws.smatrix(local=[(np.eye(ws.nch), -50 * woods_saxon_safe(ws.r, 4.0, 0.6))])
    g = amplitudes.transition_geometry(ws, 1, 1)
    xs = amplitudes.observables(S, amplitudes.transition_tables(ws, g, ANGLES))
    np.testing.assert_array_equal(xs.dsdo, 0)


def test_spin_half_polarization_checks_its_assumptions():
    """Ay and Q from two amplitudes need spin 1/2 on spin 0 at both ends."""
    _, _, xs, _ = elastic_observables(1.0, central)
    with pytest.raises(ValueError, match="spin-1/2"):
        amplitudes.spin_half_polarization(xs)
