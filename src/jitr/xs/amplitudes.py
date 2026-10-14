r"""Scattering amplitudes and observables for any projectile and target spins.

This follows I. J. Thompson, "Methods of Direct Reaction Theories", ch. 3.2 in
*Scattering* (R. Pike, P. Sabatier eds., Academic Press, 2002), Sec. 3.2.6,
whose coupling order ``L + J_p = J, J + J_t = J_T`` is that of
:func:`jitr.reactions.system.jpi_blocks` (``(l, s) j`` coupled with ``I``).

For an entrance level ``α0`` (projectile spin ``s``, target spin ``I``) and an
exit level ``α`` (``s'``, ``I'``), the amplitude for spin projections
``(m, M) -> (m', M')`` is (Thompson Eq. 38)

.. math::
    f_{m'M':mM}(θ) = δ_{αα_0}δ_{m'm}δ_{M'M} F_c(θ) + \sum_{J^π}\sum_{c c_0}
        \langle L_0 0\, s\, m | j_0 m\rangle \langle j_0 m\, I M | J\, M_T\rangle
        \langle L' M_{L'}\, s' m' | j' m'_j\rangle
        \langle j' m'_j\, I' M' | J\, M_T\rangle
        \frac{\sqrt{4π(2L_0+1)}}{k_0} e^{i(σ_{c_0} + σ_c)}
        \frac{i}{2}(δ_{cc_0} - S_{cc_0}) Y_{L'}^{M_{L'}}(θ, 0),

with ``M_T = m + M`` and ``M_{L'} = M_T - m' - M'`` (the superscript of ``Y``
must agree with the Clebsch-Gordan coefficient beside it). Conventions:

* the couplings carry ``i^{l'-l}`` (as :func:`~jitr.reactions.system.rotor`
  does), so the radial functions are Thompson's ``f_α`` of Eq. 32 and
  Eq. 38 applies to our S directly;
* S is flux-normalized, so ``f`` carries ``1/k_0`` only and
  ``dσ/dΩ = Σ|f|^2 / ((2s+1)(2I+1))`` with no velocity ratio (Thompson's
  Eq. 41 carries it because his S is a ratio of wave amplitudes);
* Coulomb phases appear as ``e^{i(σ_{c_0}+σ_c)}`` and the Coulomb amplitude
  (Eq. 40) as ``F_c e^{2iσ_0}``, a common phase of every term;
* the spherical tensor analyzing powers are
  ``T_kq = Tr(f τ_kq f^†) / Tr(f f^†)`` with ``f`` indexed
  ``[final, initial]`` and τ_kq acting on the initial projectile spin with
  ``\langle m''|τ_kq|m\rangle = \hat k \langle s m\, k q | s m''\rangle``
  (Thompson's ``(τ_kq)_{m m''}``, so it enters the trace transposed), in the
  Madison convention with scattering in the ``xz`` plane; for spin 1/2,
  ``A_y = \sqrt 2\, iT_{11}``.

Only channels that are active and open enter (closed and padded slots of S are
zero, and must not contribute a ``δ_{cc_0}``). Identical particles are not
symmetrized.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma, sph_harm_y

from ..reactions.system import _wigner


def clebsch_gordan(j1, m1, j2, m2, j, m) -> float:
    r""":math:`\langle j_1 m_1\, j_2 m_2 | j m\rangle` from a 3j symbol."""
    d = lambda x: round(2 * x)  # noqa: E731
    phase = (-1) ** round(j1 - j2 + m)
    return (
        phase
        * np.sqrt(2 * j + 1)
        * _wigner("3j", d(j1), d(j2), d(j), d(m1), d(m2), d(-m))
    )


def projections(spin) -> np.ndarray:
    return np.arange(-spin, spin + 1)


class Tables(NamedTuple):
    """Geometry of one entrance -> exit transition at fixed angles and energies.

    Channel pairs ``p`` are (block, exit channel, entrance channel).
    """

    W: jax.Array  # (n_final, n_initial, P): Clebsch-Gordan products sqrt(2L0+1)
    Y: jax.Array  # (n_final, n_initial, L, Nθ): Y_{L'}^{M_L'}(θ, 0)
    onehot: jax.Array  # (P, L): pair's exit l
    pair: jax.Array  # (3, P): block, exit channel, entrance channel
    delta: jax.Array  # (P,): δ_{c c0}
    weight: jax.Array  # (P,): (2J+1) / ((2s+1)(2I+1))
    phase: jax.Array  # (NE, P): open * sqrt(4π) e^{i(σ+σ')} (i/2) / k0
    is_open: jax.Array  # (NE, P)
    coulomb: jax.Array  # (NE, n_final, n_initial, Nθ): δ F_c e^{2iσ0}
    tau: jax.Array  # (n_kq, n_initial, n_initial): <a|τ_kq|b>
    k0: jax.Array  # (NE,)
    norm: float  # 1 / ((2s+1)(2I+1))


class XS(NamedTuple):
    """``dsdo`` (mb/sr) and ``T`` (``T_kq`` for ``kq_labels``) of shape
    ``(NE, ·, Nθ)``, the integrated exit cross section ``sigma`` (mb), and the
    amplitudes ``f`` (fm), ``(NE, n_final, n_initial, Nθ)``."""

    dsdo: jax.Array
    T: jax.Array
    sigma: jax.Array
    f: jax.Array


def kq_labels(spin) -> list[tuple[int, int]]:
    """Order of the analyzing powers ``T_kq``: ``1 <= k <= 2s``, ``0 <= q <= k``."""
    return [(k, q) for k in range(1, round(2 * spin) + 1) for q in range(k + 1)]


class Geometry(NamedTuple):
    """The angle- and energy-independent part of a transition: channel pairs
    ``(block, exit channel, entrance channel)``, their Clebsch-Gordan
    products and weights, and the ``τ_kq`` matrices."""

    entrance: int
    exit: int
    initial: list  # (m, M) of the entrance level
    final: list  # (m', M') of the exit level
    pair: np.ndarray  # (3, P)
    W: np.ndarray  # (n_final, n_initial, P)
    tau: np.ndarray  # (n_kq, n_initial, n_initial): <a|τ_kq|b>
    norm: float  # 1 / ((2s+1)(2I+1))


def transition_geometry(ws, entrance: int, exit: int) -> Geometry:
    """Clebsch-Gordan tables of the ``entrance -> exit`` transition of a
    :class:`jitr.xs.coupled.Workspace`, built once per workspace."""
    b, lv = ws.blocks, ws.levels
    s0, I0, s1, I1 = lv[entrance].s, lv[entrance].I, lv[exit].s, lv[exit].I
    initial = [(m, M) for m in projections(s0) for M in projections(I0)]
    final = [(m, M) for m in projections(s1) for M in projections(I1)]
    pairs = [
        (bb, c, c0)
        for bb in range(b.l.shape[0])
        for c in np.flatnonzero(b.active[bb] & (b.level[bb] == exit))
        for c0 in np.flatnonzero(b.active[bb] & (b.level[bb] == entrance))
    ]
    if not pairs:
        raise ValueError(
            f"no block of the workspace holds channels of both level {entrance} "
            f"and level {exit} (check the J and parity filters and lmax)"
        )
    pb, pc, pc0 = pairs = np.array(pairs).T
    L, J = b.l[pb, pc], b.J[pb]
    L0, j0, j1 = b.l[pb, pc0], b.j[pb, pc0], b.j[pb, pc]
    W = np.zeros((len(final), len(initial), pairs.shape[1]))
    for f, (m1, M1) in enumerate(final):
        for i, (m, M) in enumerate(initial):
            ML = m + M - m1 - M1
            for p in range(pairs.shape[1]):
                if abs(ML) > L[p] or abs(m + M) > J[p]:
                    continue
                W[f, i, p] = (
                    clebsch_gordan(L0[p], 0, s0, m, j0[p], m)
                    * clebsch_gordan(j0[p], m, I0, M, J[p], m + M)
                    * clebsch_gordan(L[p], ML, s1, m1, j1[p], ML + m1)
                    * clebsch_gordan(j1[p], ML + m1, I1, M1, J[p], m + M)
                    * np.sqrt(2 * L0[p] + 1)
                )
    tau = np.array(
        [
            [
                [
                    np.sqrt(2 * k + 1) * clebsch_gordan(s0, m2, k, q, s0, m) * (M == M2)
                    for (m2, M2) in initial
                ]
                for (m, M) in initial
            ]
            for k, q in kq_labels(s0)
        ]
    ).reshape(-1, len(initial), len(initial))
    norm = 1 / ((2 * s0 + 1) * (2 * I0 + 1))
    return Geometry(entrance, exit, initial, final, pairs, W, tau, norm)


def transition_tables(ws, g: Geometry, angles, energies=slice(None)) -> Tables:
    """Tables for :func:`observables` at ``angles`` (radians) and the selected
    workspace energies, from the transition's :class:`Geometry`."""
    b = ws.blocks
    pb, pc, pc0 = g.pair
    L = b.l[pb, pc]
    lmax = int(b.l.max())
    Y = np.zeros((len(g.final), len(g.initial), lmax + 1, len(angles)), complex)
    for f, (m1, M1) in enumerate(g.final):
        for i, (m, M) in enumerate(g.initial):
            ML = round(m + M - m1 - M1)
            for Lp in range(abs(ML), lmax + 1):
                Y[f, i, Lp] = sph_harm_y(Lp, ML, angles, 0)
    k0 = np.sqrt(np.abs(ws.k2_level[energies, g.entrance]))
    is_open = ws.open[energies][:, pb, pc] & ws.open[energies][:, pb, pc0]
    sigma = ws.sigma[energies]
    phase = (
        is_open
        * np.exp(1j * (sigma[:, pb, pc0] + sigma[:, pb, pc]))
        * np.sqrt(4 * np.pi)
        * 0.5j
        / k0[:, None]
    )
    n = len(g.initial)
    coulomb = np.zeros((len(k0), len(g.final), n, len(angles)), complex)
    eta0 = ws.eta_level[energies, g.entrance]
    elastic = g.exit == g.entrance
    if elastic and np.any(eta0 != 0):
        sin2 = np.sin(np.asarray(angles) / 2) ** 2
        sigma0 = loggamma(1 + 1j * eta0).imag[:, None]
        Fc = (
            -eta0[:, None]
            / (2 * k0[:, None] * sin2)
            * np.exp(2j * sigma0 - 2j * eta0[:, None] * np.log(np.sqrt(sin2)))
        )
        is_open0 = (ws.k2_level[energies, g.entrance] > 0)[:, None]  # closed: no Fc
        coulomb = np.eye(n)[None, :, :, None] * (is_open0 * Fc)[:, None, None, :]
    return Tables(
        W=jnp.asarray(g.W),
        Y=jnp.asarray(Y),
        onehot=jnp.asarray(L[:, None] == np.arange(lmax + 1)),
        pair=jnp.asarray(g.pair),
        delta=jnp.asarray(pc == pc0) * elastic,
        weight=jnp.asarray((2 * b.J[pb] + 1) * g.norm),
        phase=jnp.asarray(phase),
        is_open=jnp.asarray(is_open),
        coulomb=jnp.asarray(coulomb),
        tau=jnp.asarray(g.tau),
        k0=jnp.asarray(k0),
        norm=g.norm,
    )


@jax.jit
def observables(S, t: Tables) -> XS:
    """Observables from the S-matrix ``(NE, NB, nch, nch)`` of the
    workspace (and energies) the tables were built for."""
    pb, pc, pc0 = t.pair
    X = t.phase * (t.delta - S[:, pb, pc, pc0])
    B = jnp.einsum("fip,ep,pl->efil", t.W, X, t.onehot)
    f = jnp.einsum("efil,filt->efit", B, t.Y) + t.coulomb
    total = jnp.sum(jnp.abs(f) ** 2, axis=(1, 2))
    traces = jnp.einsum("efat,kab,efbt->ekt", f, t.tau, jnp.conj(f))
    sigma = (
        jnp.pi
        / t.k0**2
        * jnp.sum(
            t.weight * t.is_open * jnp.abs(t.delta - S[:, pb, pc, pc0]) ** 2, axis=1
        )
    )
    return XS(
        dsdo=10 * t.norm * total,
        T=traces / jnp.maximum(total, 1e-30)[:, None],
        sigma=10 * sigma,
        f=f,
    )


def spin_half_polarization(xs: XS):
    """``(A_y, Q)`` for a spin-1/2 projectile on a spin-0 target.

    With the amplitude matrix ``A + B σ·n̂`` in the scattering plane,
    ``A = f_{++}`` and the spin-flip ``f_{-+} = iB``, so
    ``A_y = 2 Im(A^* f_{-+}) / Σ``, ``Q = 2 Re(A^* f_{-+}) / Σ`` with
    ``Σ = |A|^2 + |B|^2``; ``A_y = \\sqrt 2\\, iT_{11}``.
    """
    if xs.f.shape[1:3] != (2, 2):
        raise ValueError("A_y and Q need a spin-1/2 projectile on spin-0 states")
    A, flip = xs.f[:, 1, 1], xs.f[:, 0, 1]
    denom = jnp.maximum(0.5 * jnp.sum(jnp.abs(xs.f) ** 2, axis=(1, 2)), 1e-30)
    return 2 * jnp.imag(jnp.conj(A) * flip) / denom, 2 * jnp.real(
        jnp.conj(A) * flip
    ) / denom


def integrated_cross_sections(ws, S, entrance: int = 0, energies=slice(None)):
    """``(σ_T, σ_R)`` in mb for incidence in level ``entrance``: total (for
    neutral projectiles) and non-elastic (absorption plus every other
    level), from block sums with weights ``(2J+1)/((2s+1)(2I+1))``."""
    return _integrals(S, *integral_weights(ws, entrance, energies))


def integral_weights(ws, entrance: int = 0, energies=slice(None)):
    """``(π w / k^2, open)`` per (energy, block, channel) of the entrance level,
    ``w = (2J+1)/((2s+1)(2I+1))``."""
    b, lv = ws.blocks, ws.levels[entrance]
    is_open = ws.open[energies] & b.active & (b.level == entrance)
    weight = (2 * b.J + 1)[:, None] / ((2 * lv.s + 1) * (2 * lv.I + 1))
    return np.pi * weight / ws.k2_level[energies, entrance][:, None, None], is_open


@jax.jit
def _integrals(S, weight, is_open):
    total = 2 * weight * is_open * (1 - jnp.diagonal(S, axis1=2, axis2=3).real)
    out = jnp.sum(jnp.abs(S) ** 2 * is_open[..., :, None], axis=2)  # in the level
    nonelastic = weight * is_open * (1 - out)
    return 10 * total.sum(axis=(1, 2)), 10 * nonelastic.sum(axis=(1, 2))
