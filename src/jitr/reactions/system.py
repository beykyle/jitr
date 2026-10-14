"""J^pi blocks of channels, their couplings and boundary values."""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

from ..utils.free_solutions import coulomb_hankel_table, whittaker_log_derivative

# ---------------------------------------------------------------------------
# J^pi blocks
#
# A channel is (level alpha, l, j): the projectile with spin s_alpha and
# orbital angular momentum l couple to j, which couples with the spin I_alpha
# of the target state to the total J. Parity is pi_alpha (-1)^l. Total J and
# parity are conserved, so each J^pi block is an independent set of coupled
# radial equations.
# ---------------------------------------------------------------------------


class Level(NamedTuple):
    """A target state (and projectile) defining one kind of channel.

    Attributes:
        I: Spin of the target state.
        parity: Its parity, +1 or -1 (times that of the projectile).
        s: Spin of the projectile in this partition.
    """

    I: float  # noqa: E741
    parity: int = 1
    s: float = 0.0


class Blocks(NamedTuple):
    """The channels of each J^pi block, padded to a common channel count.

    ``J`` and ``parity`` have shape ``(NB,)``; ``l``, ``j``, ``level`` and
    ``active`` have shape ``(NB, nch)``. Padded slots are inactive, with
    ``l = 0`` and ``level = 0``; the workspace makes them decoupled closed
    channels, so they never change the active channels (``jax.vmap`` needs
    every block to have the same shape).
    """

    J: np.ndarray
    parity: np.ndarray
    l: np.ndarray
    j: np.ndarray
    level: np.ndarray
    active: np.ndarray


def jpi_blocks(levels, lmax, J=None, parity=None) -> Blocks:
    """Enumerate the J^pi blocks of channels with ``l <= lmax``.

    Args:
        levels: Sequence of :class:`Level`.
        lmax: Maximum orbital angular momentum of any channel.
        J: Optional sequence of total angular momenta to keep.
        parity: Optional parity (+1 or -1) to keep.

    Returns:
        The blocks, ordered by (J, parity), with channels ordered by
        (level, l, j) so the first level's channels come first.
    """
    # work in doubled angular momenta so half-integers are exact
    s2 = [round(2 * lv.s) for lv in levels]
    I2 = [round(2 * lv.I) for lv in levels]
    J2max = 2 * lmax + max(s2) + max(I2)
    wanted = None if J is None else {round(2 * x) for x in np.atleast_1d(J)}
    blocks = []
    for J2 in range(J2max + 1):
        if wanted is not None and J2 not in wanted:
            continue
        for pi in (1, -1):
            if parity is not None and pi != parity:
                continue
            channels = [
                (alpha, l2 // 2, j2 / 2)
                for alpha, lv in enumerate(levels)
                for j2 in range(abs(J2 - I2[alpha]), J2 + I2[alpha] + 1, 2)
                for l2 in range(abs(j2 - s2[alpha]), j2 + s2[alpha] + 1, 2)
                if l2 % 2 == 0
                and l2 // 2 <= lmax
                and lv.parity * (-1) ** (l2 // 2) == pi
            ]
            if channels:
                blocks.append((J2 / 2, pi, sorted(channels)))
    if not blocks:
        raise ValueError("no J^pi block has a channel: check the J and parity filters")
    nch = max(len(ch) for _, _, ch in blocks)
    NB = len(blocks)
    l = np.zeros((NB, nch), int)
    j = np.zeros((NB, nch))
    level = np.zeros((NB, nch), int)
    active = np.zeros((NB, nch), bool)
    for b, (_, _, channels) in enumerate(blocks):
        for c, (alpha, lc, jc) in enumerate(channels):
            level[b, c], l[b, c], j[b, c], active[b, c] = alpha, lc, jc, True
    return Blocks(
        J=np.array([x[0] for x in blocks]),
        parity=np.array([x[1] for x in blocks]),
        l=l,
        j=j,
        level=level,
        active=active,
    )


def spin_orbit(blocks: Blocks, levels) -> np.ndarray:
    r"""``2 l.s = j(j+1) - l(l+1) - s(s+1)`` on the diagonal of each block.

    For a spin-1/2 projectile this is :math:`l\cdot\sigma`, the convention of
    jitr's spin-orbit potentials.
    """
    s = np.array([lv.s for lv in levels])[blocks.level]
    ls = blocks.j * (blocks.j + 1) - blocks.l * (blocks.l + 1) - s * (s + 1)
    return ls[:, :, None] * np.eye(ls.shape[1])


@functools.cache
def _wigner(kind: str, *args2: int) -> float:
    """3j or 6j symbol with doubled integer arguments."""
    args = [Rational(a, 2) for a in args2]
    return float(wigner_3j(*args) if kind == "3j" else wigner_6j(*args))


def rotor(blocks: Blocks, levels, lam: int) -> np.ndarray:
    r"""Geometry of the first-order rotational coupling of multipole ``lam``.

    A rigid axial rotor deformed as ``R(θ') = R (1 + β Y_{λ0}(θ'))`` in its
    body frame has, to first order in β, the potential
    ``U(r) - β R U'(r) Y_λ(r̂)·Q_λ``, where ``Q_λ`` rotates between the
    states of a K = 0 band. Its matrix element between channels
    ``((l s)j, I) J`` and ``((l' s)j', I') J`` is (Edmonds, Eqs. 7.1.6-7.1.8)

    .. math::
        i^{l'-l} (-1)^{j'+I+J} \begin{Bmatrix} j & I & J \\ I' & j' & λ \end{Bmatrix}
        \langle (ls)j \| Y_λ \| (l's)j' \rangle \langle I \| Q_λ \| I' \rangle,

    with :math:`\langle (ls)j\|Y_λ\|(l's)j'\rangle = (-1)^{l+s+j'+λ}
    \hat j \hat j' \{l\,j\,s;\, j'\,l'\,λ\} \langle l\|Y_λ\|l'\rangle`,
    :math:`\langle l\|Y_λ\|l'\rangle = (-1)^l \hat l \hat λ \hat l'
    (l\,λ\,l';0\,0\,0)/\sqrt{4π}` and
    :math:`\langle I\|Q_λ\|I'\rangle = (-1)^I \hat I \hat I' (I\,λ\,I';0\,0\,0)`
    (:math:`\hat x = \sqrt{2x+1}`). The factor :math:`i^{l'-l}` makes the
    matrix real and symmetric. For a spin-0 projectile this is the coupling
    of Descouvemont, Comput. Phys. Commun. 200 (2016) 199, Examples 3 and 4.

    Returns:
        The coupling matrix of each block, ``(NB, nch, nch)``; multiply by
        the form factor ``-β R U'(r)``.
    """
    I = np.array([lv.I for lv in levels])  # noqa: E741
    s = np.array([lv.s for lv in levels])
    NB, nch = blocks.l.shape
    C = np.zeros((NB, nch, nch))
    d = lambda x: round(2 * x)  # noqa: E731
    hat = lambda x: np.sqrt(2 * x + 1)  # noqa: E731
    for b, c, cp in np.argwhere(blocks.active[:, :, None] & blocks.active[:, None, :]):
        a, ap = blocks.level[b, c], blocks.level[b, cp]
        l, j, lp, jp = blocks.l[b, c], blocks.j[b, c], blocks.l[b, cp], blocks.j[b, cp]
        Ia, Ib, sa, J = I[a], I[ap], s[a], blocks.J[b]
        if s[ap] != sa or (lp - l) % 2:
            continue
        geometry = (
            _wigner("6j", d(j), d(Ia), d(J), d(Ib), d(jp), d(lam))
            * _wigner("6j", d(l), d(j), d(sa), d(jp), d(lp), d(lam))
            * _wigner("3j", d(l), d(lam), d(lp), 0, 0, 0)
            * _wigner("3j", d(Ia), d(lam), d(Ib), 0, 0, 0)
        )
        if geometry == 0:
            continue
        phase = (-1) ** round((lp - l) / 2 + jp + Ia + J + l + sa + jp + lam + l + Ia)
        norm = hat(j) * hat(jp) * hat(l) * hat(lam) * hat(lp) * hat(Ia) * hat(Ib)
        C[b, c, cp] = phase * norm * geometry / np.sqrt(4 * np.pi)
    return C


def asymptotics(blocks: Blocks, k2, eta, a: float):
    """Boundary values at the channel radius ``a`` (fm), host-side.

    Args:
        blocks: The J^pi blocks.
        k2: Squared wavenumber of each level, fm^-2, shape ``(NE, nlev)``;
            negative below threshold.
        eta: Sommerfeld parameter of each level (computed with ``|k|``).
        a: Channel radius in fm.

    Returns:
        ``(H, Bc, is_open)``: ``H`` of shape ``(NE, NB, 4, nch)`` holds
        ``(H+, H-, H+', H-')`` of open channels at ``rho = k a``, ``Bc`` of
        shape ``(NE, NB, nch)`` the closed-channel Bloch constants and
        ``is_open`` the open mask. Inactive slots are closed with ``Bc = 0``.
    """
    NE, nlev = k2.shape
    NB, nch = blocks.l.shape
    H = np.zeros((NE, NB, 4, nch), complex)
    H[:, :, :2] = 1
    Bc = np.zeros((NE, NB, nch))
    is_open = (k2[:, blocks.level] > 0) & blocks.active
    for alpha in range(nlev):
        bs, cs = np.nonzero(blocks.active & (blocks.level == alpha))
        if bs.size == 0:
            continue
        ls = blocks.l[bs, cs]
        for e in range(NE):
            k = np.sqrt(abs(k2[e, alpha]))
            if k2[e, alpha] > 0:
                problem = (
                    f"level {alpha} at energy index {e} (Sommerfeld parameter "
                    f"{eta[e, alpha]:.3g}) is too close to its threshold for Coulomb "
                    "functions at the channel radius"
                )
                if eta[e, alpha] > 250:
                    raise ValueError(problem)
                table = np.array(
                    coulomb_hankel_table(k * a, eta[e, alpha], ls.max())[:]
                )
                if not np.all(np.isfinite(table)):
                    raise ValueError(problem)
                H[e, bs, :, cs] = table[:, ls].T
            else:
                Bc[e, bs, cs] = [
                    whittaker_log_derivative(int(l), eta[e, alpha], 2 * k * a)
                    for l in ls
                ]
    return H, Bc, is_open
