r"""Coupled-channels workspace: every J^pi block at every energy in one call.

The channels are enumerated from a list of :class:`~jitr.reactions.Level`
(see :func:`~jitr.reactions.system.jpi_blocks`); each J^pi block is solved
with :func:`jitr.rmatrix.solve`, vectorized with ``jax.vmap`` over blocks and
energies. The interaction is a sum of terms ``C ⊗ v``: a coupling matrix
``C`` per block, shape ``(NB, nch, nch)`` (or ``(nch, nch)`` for all blocks),
times a radial form ``v`` on the mesh, in MeV. Forms come in three layouts,
told apart by ``ndim`` alone:

* ``(N,)`` or ``(N, N)``: the same for every energy and block;
* ``(NE, N)`` or ``(NE, N, N)``: energy dependent (``NE`` may be 1);
* ``(L, NE, N)`` or ``(L, NE, N, N)``: also depending on the channel's ``l``
  (``v[l_c]``); allowed only for couplings that conserve ``l``.

Couplings are constant geometry, host-side numpy arrays such as
:func:`~jitr.reactions.system.rotor`, :func:`~jitr.reactions.system.spin_orbit`
or level projectors, so they cannot carry gradients. A parameter to
differentiate, such as a deformation ``β``, goes in the form:
``(rotor(blocks, levels, 2), beta * R * dU)``.

Results are energy first: ``S`` has shape ``(NE, NB, nch, nch)``. Blocks
with fewer channels are padded with decoupled closed channels; their rows and
columns of ``S``, like those of closed channels, are zero (so the diagonal of
``S`` there is 0, not 1: use the ``open`` mask in any sum over channels).

By default every block is solved padded to the largest one, ``nch``. When
block sizes differ a lot, ``block_sizes`` solves groups of blocks, each padded
only to its own largest block; every group is compiled separately, so a few
groups pay off and many cost compile time. The results are the same.

A batch of potentials (e.g. one per parameter sample) is solved in one call
with :func:`jax.vmap`, the forms carrying a leading sample axis. Each step of
the solver then holds ``batch_size`` systems per sample, so for large blocks
build the workspace with a smaller ``batch_size``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

from .. import rmatrix
from ..reactions.system import Blocks, asymptotics, jpi_blocks
from ..utils.constants import HBARC


class Potential(NamedTuple):
    """The terms of an optical potential, each optional, on a workspace mesh.

    Local terms are ``(N,)`` or ``(NE, N)`` in MeV; nonlocal terms are
    ``(N, N)`` or ``(NE, N, N)`` in MeV fm^-1. Either kind may carry a leading
    ``l`` axis (see the module docstring). The spin-orbit terms multiply
    ``2 l.s`` (``l.sigma`` for spin 1/2).
    """

    central: Any = None
    spin_orbit: Any = None
    coulomb: Any = None
    nonlocal_central: Any = None
    nonlocal_spin_orbit: Any = None


def potential_terms(potential: Potential, C, spin_orbit) -> tuple[list, list]:
    """``(local, nonlocal_)`` term lists of a potential acting with coupling
    ``C`` (``spin_orbit`` is the ``2 l.s`` coupling of the blocks)."""
    p = potential
    local = [(C, p.central), (C, p.coulomb), (C * spin_orbit, p.spin_orbit)]
    nonlocal_ = [(C, p.nonlocal_central), (C * spin_orbit, p.nonlocal_spin_orbit)]
    return (
        [t for t in local if t[1] is not None],
        [t for t in nonlocal_ if t[1] is not None],
    )


def _groups(counts, block_sizes) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Blocks grouped by padded size: ``((block indices), size)`` per group.

    ``counts`` is the number of active channels of each block. ``block_sizes``
    is ``None`` (one group, padded to the largest block), the largest number
    of groups (sizes chosen to minimize the padded cost, the sum over blocks
    of size cubed), or the padded sizes themselves.
    """
    counts = np.asarray(counts)
    if block_sizes is None:
        return ((tuple(range(counts.size)), int(counts.max())),)
    if isinstance(block_sizes, (int, np.integer)):
        sizes, mult = np.unique(counts, return_counts=True)
        # cost[g, i]: cheapest padding of sizes[:i] in g groups (dynamic program)
        m, k = sizes.size, min(int(block_sizes), sizes.size)
        cost = np.full((k + 1, m + 1), np.inf)
        cost[0, 0], cut = 0.0, np.zeros((k + 1, m + 1), int)
        for g in range(1, k + 1):
            for i in range(1, m + 1):
                for j in range(i):
                    c = cost[g - 1, j] + sizes[i - 1] ** 3 * mult[j:i].sum()
                    if c < cost[g, i]:
                        cost[g, i], cut[g, i] = c, j
        g, i, chosen = int(np.argmin(cost[:, m])), m, []
        while i:
            chosen.append(int(sizes[i - 1]))
            i, g = cut[g, i], g - 1
        block_sizes = chosen
    padded = np.sort(np.asarray(block_sizes, int))
    if padded[-1] < counts.max():
        raise ValueError(
            f"block_sizes {list(padded)} cannot hold a block of {counts.max()} channels"
        )
    size = padded[np.searchsorted(padded, counts)]
    return tuple(
        (tuple(int(b) for b in np.flatnonzero(size == n)), int(n))
        for n in np.unique(size)
    )


def promote(v, is_nonlocal: bool, NE: int, lmax: int, N: int):
    """A form in one of the three layouts, as ``(L, NE or 1, N[, N])``."""
    base = 1 + is_nonlocal
    v = jnp.asarray(v)
    if not base <= v.ndim <= base + 2:
        raise ValueError(f"form must have {base} to {base + 2} axes, got {v.shape}")
    if v.shape[v.ndim - base :] != (N,) * base:
        raise ValueError(f"form {v.shape} does not end in the mesh size {(N,) * base}")
    v = v.reshape((1,) * (base + 2 - v.ndim) + v.shape)
    if v.shape[1] not in (1, NE):
        raise ValueError(f"energy axis of form {v.shape} must be 1 or NE = {NE}")
    if v.shape[0] != 1 and v.shape[0] <= lmax:
        raise ValueError(f"l-dependent form {v.shape} needs L > lmax = {lmax}")
    return v


class Workspace:
    """Precomputed channels, mesh and boundary values for a set of levels.

    Args:
        a: Channel radius in fm.
        nbasis: Number of mesh points.
        levels: Sequence of :class:`~jitr.reactions.Level`.
        lmax: Maximum orbital angular momentum.
        k2: Squared wavenumber of each level in fm^-2, ``(NE, nlev)``,
            negative for closed channels.
        mu: Reduced mass of each level in MeV, ``(NE, nlev)``.
        eta: Sommerfeld parameter of each level (computed with ``|k|``).
        J: Optional total angular momenta to keep.
        parity: Optional parity to keep.
        batch_size: Number of (energy, block) systems solved at once. The
            default keeps about 256 MB of matrices in flight (each system
            needs about three copies of its matrix): everything at once for
            single-channel problems, about ten 720 x 720 blocks at a time.
        block_sizes: How blocks are padded (see the module docstring):
            ``None`` pads every block to the largest; an ``int`` is the most
            groups of blocks to pad separately, with sizes chosen to minimize
            the work; a sequence gives the padded sizes themselves.
    """

    def __init__(
        self,
        a,
        nbasis,
        levels,
        lmax,
        k2,
        mu,
        eta,
        J=None,
        parity=None,
        batch_size=None,
        block_sizes=None,
    ):
        k2, mu, eta = (np.atleast_2d(np.asarray(x, float)) for x in (k2, mu, eta))
        at_threshold = np.argwhere(np.abs(k2) < 1e-12)
        if at_threshold.size:
            e, level = at_threshold[0]
            raise ValueError(
                f"level {level} is exactly at threshold at energy index {e} "
                "(k^2 = 0): shift the energy grid off the threshold"
            )
        self.levels = list(levels)
        self.mesh = rmatrix.mesh(nbasis, a)
        self.blocks: Blocks = jpi_blocks(self.levels, lmax, J, parity)
        b = self.blocks
        self.H, self.Bc, self.open = asymptotics(b, k2, eta, a)
        self.k2_level, self.eta_level = k2, eta
        # Coulomb phases arg Γ(1 + l + iη) of the open channels
        eta_c = eta[:, b.level]
        self.sigma = np.where(self.open, loggamma(1 + b.l + 1j * eta_c).imag, 0.0)
        self.k2 = np.where(b.active, k2[:, b.level], -1.0)
        m = np.sqrt(mu[:, b.level])
        self.M = 2 * m[..., :, None] * m[..., None, :] / HBARC**2
        self.active2 = b.active[:, :, None] & b.active[:, None, :]
        self.same_l = b.l[:, :, None] == b.l[:, None, :]
        self._batch_size = batch_size
        self.groups = _groups(b.active.sum(1), block_sizes)

    def _batch_sizes(self):
        """Systems solved at once in each group: ``batch_size``, or about
        256 MB of matrices in flight (three copies of each)."""
        if self._batch_size:
            return tuple(self._batch_size for _ in self.groups)
        NE, N = self.k2.shape[0], self.mesh.x.size
        sizes = []
        for idx, n in self.groups:
            systems = NE * len(idx)
            budget = min(systems, max(1, 2**28 // (3 * 16 * (n * N) ** 2)))
            # equal batches: a small remainder batch would cost a whole step
            sizes.append(-(-systems // -(-systems // budget)))
        return tuple(sizes)

    @property
    def batch_size(self):
        """Systems solved at once (in the largest group)."""
        return self._batch_sizes()[-1]

    def channel(self, c: int) -> Workspace:
        """The one-level workspace of channel ``c`` of every block, sliced
        from this one (mesh and boundary values are reused, not recomputed).
        Channel ``c`` must be active, and of one level, in every block."""
        b = self.blocks
        level = int(b.level[0, c])
        if not (b.active[:, c].all() and (b.level[:, c] == level).all()):
            raise ValueError(f"channel {c} is not one level active in every block")
        sl = slice(c, c + 1)
        ws = object.__new__(Workspace)
        ws.__dict__.update(self.__dict__)
        ws.levels = [self.levels[level]]
        ws.blocks = Blocks(
            b.J, b.parity, b.l[:, sl], b.j[:, sl], 0 * b.level[:, sl], b.active[:, sl]
        )
        ws.H, ws.Bc, ws.open = self.H[..., sl], self.Bc[..., sl], self.open[..., sl]
        ws.k2, ws.sigma, ws.M = (
            self.k2[..., sl],
            self.sigma[..., sl],
            self.M[..., sl, sl],
        )
        ws.k2_level = self.k2_level[:, level : level + 1]
        ws.eta_level = self.eta_level[:, level : level + 1]
        ws.active2 = ws.blocks.active[:, :, None] & ws.blocks.active[:, None, :]
        ws.same_l = ws.blocks.l[:, :, None] == ws.blocks.l[:, None, :]
        ws.groups = _groups(ws.blocks.active.sum(1), None)
        return ws

    @property
    def r(self):
        """Mesh points in fm."""
        return self.mesh.r

    @property
    def l(self):  # noqa: E743
        return self.blocks.l

    @property
    def j(self):
        return self.blocks.j

    @property
    def nch(self):
        return self.blocks.l.shape[1]

    def _terms(self, local, nonlocal_):
        """Masked couplings, forms promoted to ``(L, NE, ...)``, static spec."""
        NE = self.k2.shape[0]
        lmax = self.blocks.l[self.blocks.active].max()
        couplings, forms, spec = [], [], []
        for is_nonlocal, terms in ((False, local), (True, nonlocal_)):
            for C, v in terms:
                if np.shape(C)[-2:] != (self.nch, self.nch):
                    raise ValueError(
                        f"coupling {np.shape(C)} must be ({self.nch}, {self.nch}) "
                        "or one per block"
                    )
                C = np.broadcast_to(C, self.active2.shape) * self.active2
                v = promote(v, is_nonlocal, NE, lmax, self.mesh.x.size)
                if v.shape[0] > 1 and np.any((C != 0) & ~self.same_l):
                    raise ValueError("l-dependent forms need couplings that conserve l")
                couplings.append(C)
                forms.append(v)
                spec.append((is_nonlocal, v.shape[0] > 1, v.shape[1] > 1))
        return couplings, forms, tuple(spec)

    def _solve(self, local, nonlocal_, weights=None):
        couplings, forms, spec = self._terms(local, nonlocal_)
        return _solve_all(
            self.mesh,
            self.blocks.l,
            self.k2,
            self.M,
            self.H,
            self.Bc,
            self.open,
            couplings,
            forms,
            spec,
            weights,
            self.groups,
            self._batch_sizes(),
        )

    def smatrix(self, local=(), nonlocal_=()):
        """Flux-normalized S of every block at every energy.

        Args:
            local: Sequence of ``(C, v)`` with local forms ``v(r_n)``; the
                couplings ``C`` are constants (parameters go in ``v``).
            nonlocal_: Sequence of ``(C, v)`` with nonlocal forms
                ``v(r_n, r_m)`` in MeV fm^-1.

        Returns:
            ``(S, open)`` of shapes ``(NE, NB, nch, nch)`` and ``(NE, NB, nch)``.
        """
        return self._solve(local, nonlocal_), self.open

    def wavefunctions(self, local=(), nonlocal_=(), weights=None):
        r"""Interior wavefunctions on the mesh and at the channel radius.

        Args:
            local, nonlocal_: The interaction, as in :meth:`smatrix`.
            weights: Flux-normalized incoming amplitudes ``(nch, nw)``;
                defaults to a unit incoming wave in each channel.

        Returns:
            ``(S, u, u_a)``, with ``u`` of shape ``(NE, NB, nw, nch, N)`` and
            ``u_a`` of shape ``(NE, NB, nw, nch)``. These are the symmetrized,
            flux-normalized waves :math:`\tilde u_c = u_c / \sqrt{\mu_c}` up to
            a constant: outside the potential
            :math:`\tilde u_c = k_c^{-1/2} (i/2)[w_c H^-_c - (S w)_c H^+_c]`.
        """
        if weights is None:
            weights = np.eye(self.nch)
        S, c = self._solve(local, nonlocal_, jnp.asarray(weights, complex))
        m = self.mesh
        return S, c / jnp.sqrt(m.a * m.w), c @ m.b / jnp.sqrt(m.a)


@partial(jax.jit, static_argnames=("spec", "groups", "batch_sizes"))
def _solve_all(
    mesh, l, k2, M, H, Bc, is_open, couplings, forms, spec, weights, groups, batch_sizes
):
    """Solve every (energy, block): each group of blocks padded to its own
    size, the results padded back to ``nch`` channels."""
    if len(groups) == 1 and groups[0][1] == l.shape[1]:
        return _solve_group(
            mesh,
            l,
            k2,
            M,
            H,
            Bc,
            is_open,
            couplings,
            forms,
            spec,
            weights,
            batch_sizes[0],
        )
    NE, NB, nch = k2.shape
    S = jnp.zeros((NE, NB, nch, nch), complex)
    c = (
        None
        if weights is None
        else jnp.zeros((NE, NB, weights.shape[1], nch, mesh.x.size), complex)
    )
    for (idx, n), batch_size in zip(groups, batch_sizes, strict=True):
        b = np.array(idx)
        out = _solve_group(
            mesh,
            l[b, :n],
            k2[:, b, :n],
            M[:, b, :n, :n],
            H[:, b, :, :n],
            Bc[:, b, :n],
            is_open[:, b, :n],
            [C[b, :n, :n] for C in couplings],
            forms,
            spec,
            None if weights is None else weights[:n],
            batch_size,
        )
        if weights is None:
            S = S.at[:, b, :n, :n].set(out)
        else:
            S = S.at[:, b, :n, :n].set(out[0])
            c = c.at[:, b, :, :n].set(out[1])
    return S if weights is None else (S, c)


def _solve_group(
    mesh, l, k2, M, H, Bc, is_open, couplings, forms, spec, weights, batch_size
):
    """Map :func:`rmatrix.solve` over every (energy, block), ``batch_size``
    at a time (vectorized within a batch); the interaction of each one is
    assembled inside the map, so only a batch of matrices is ever in memory.
    Each solve is checkpointed: reverse-mode autodiff stores only its inputs
    and recomputes the solve in the backward pass, so gradients stay within
    the same memory budget instead of keeping every LU factorization."""
    NE, NB = k2.shape[:2]

    def one(e, b):
        lb = l[b]
        V = 0
        for C, v, (is_nonlocal, by_l, by_e) in zip(couplings, forms, spec, strict=True):
            vc = v[lb if by_l else 0, e if by_e else 0]  # (nch, N) or (nch, N, N)
            if not by_l:
                vc = jnp.broadcast_to(vc, (lb.shape[0], *vc.shape))
            if is_nonlocal:
                Vc = rmatrix.interaction(mesh, nonlocal_=vc)
            else:
                Vc = rmatrix.interaction(mesh, local=vc)
            V = V + C[b][:, :, None, None] * Vc[:, None]
        return rmatrix.solve(
            mesh, lb, k2[e, b], M[e, b], V, H[e, b], Bc[e, b], is_open[e, b], weights
        )

    one = jax.checkpoint(one)
    # whole batches only (the last system repeated): lax.map's separate
    # remainder step can cost several full steps
    n = NE * NB
    i = jnp.minimum(jnp.arange(-(-n // batch_size) * batch_size), n - 1)
    out = jax.lax.map(lambda i: one(i // NB, i % NB), i, batch_size=batch_size)
    return jax.tree.map(lambda x: x[:n].reshape(NE, NB, *x.shape[1:]), out)
