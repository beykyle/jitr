r"""Coupled-channels workspace: every J^pi block at every energy in one call.

The channels are enumerated from a list of :class:`~jitr.reactions.Level`
(see :func:`~jitr.reactions.system.jpi_blocks`); each J^pi block is solved
with :func:`jitr.lagrange.solve`, vectorized with ``jax.vmap`` over blocks and
energies. The interaction is a sum of terms ``C ⊗ v``: a coupling matrix
``C`` per block, shape ``(NB, nch, nch)`` (or ``(nch, nch)`` for all blocks),
times a radial form ``v`` on the mesh, in MeV. Forms come in three layouts,
told apart by ``ndim`` alone:

* ``(N,)`` or ``(N, N)``: the same for every energy and block;
* ``(NE, N)`` or ``(NE, N, N)``: energy dependent (``NE`` may be 1);
* ``(L, NE, N)`` or ``(L, NE, N, N)``: also depending on the channel's ``l``
  (``v[l_c]``); allowed only for couplings that conserve ``l``.

Results are energy first: ``S`` has shape ``(NE, NB, nch, nch)``. Blocks
with fewer channels are padded with decoupled closed channels; their rows and
columns of ``S``, like those of closed channels, are zero (so the diagonal of
``S`` there is 0, not 1: use the ``open`` mask in any sum over channels).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .. import lagrange
from ..reactions.system import Blocks, asymptotics, jpi_blocks
from ..utils.constants import HBARC


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
    """

    def __init__(
        self, a, nbasis, levels, lmax, k2, mu, eta, J=None, parity=None, batch_size=None
    ):
        k2, mu, eta = (np.atleast_2d(np.asarray(x, float)) for x in (k2, mu, eta))
        self.levels = list(levels)
        self.mesh = lagrange.mesh(nbasis, a)
        self.blocks: Blocks = jpi_blocks(self.levels, lmax, J, parity)
        b = self.blocks
        self.H, self.Bc, self.open = asymptotics(b, k2, eta, a)
        self.k2 = np.where(b.active, k2[:, b.level], -1.0)
        m = np.sqrt(mu[:, b.level])
        self.M = 2 * m[..., :, None] * m[..., None, :] / HBARC**2
        self.active2 = b.active[:, :, None] & b.active[:, None, :]
        self.same_l = b.l[:, :, None] == b.l[:, None, :]
        matrix_bytes = 16 * (self.nch * nbasis) ** 2
        n = self.k2.shape[0] * self.k2.shape[1]
        self.batch_size = batch_size or int(min(n, max(1, 2**28 // (3 * matrix_bytes))))

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
        couplings, forms, spec = [], [], []
        for is_nonlocal, terms in ((False, local), (True, nonlocal_)):
            for C, v in terms:
                C = np.broadcast_to(C, self.active2.shape) * self.active2
                base = 1 + is_nonlocal
                v = jnp.asarray(v)
                if v.ndim == base:
                    v = v[None, None]
                elif v.ndim == base + 1:
                    v = v[None]
                    assert v.shape[1] in (1, NE), "energy axis must be 1 or NE"
                else:
                    assert v.ndim == base + 2, f"form must have {base}-{base + 2} axes"
                    assert v.shape[1] in (1, NE), "energy axis must be 1 or NE"
                    lmax = self.blocks.l[self.blocks.active].max()
                    assert v.shape[0] > lmax, "l-dependent form needs L > lmax"
                    assert not np.any((C != 0) & ~self.same_l), (
                        "l-dependent forms need couplings that conserve l"
                    )
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
            self.batch_size,
        )

    def smatrix(self, local=(), nonlocal_=()):
        """Flux-normalized S of every block at every energy.

        Args:
            local: Sequence of ``(C, v)`` with local forms ``v(r_n)``.
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


@partial(jax.jit, static_argnames=("spec", "batch_size"))
def _solve_all(
    mesh, l, k2, M, H, Bc, is_open, couplings, forms, spec, weights, batch_size
):
    """Map :func:`lagrange.solve` over every (energy, block), ``batch_size``
    at a time (vectorized within a batch); the interaction of each one is
    assembled inside the map, so only a batch of matrices is ever in memory.
    Each solve is checkpointed: reverse-mode autodiff stores only its inputs
    and recomputes the solve in the backward pass, so gradients stay within
    the same memory budget instead of keeping every LU factorization."""
    NE, NB = k2.shape[:2]
    weight_matrix = mesh.a * jnp.sqrt(jnp.outer(mesh.w, mesh.w))

    def one(e, b):
        lb = l[b]
        V = 0
        for C, v, (is_nonlocal, by_l, by_e) in zip(couplings, forms, spec, strict=True):
            vc = v[lb if by_l else 0, e if by_e else 0]  # (nch, N) or (nch, N, N)
            if not by_l:
                vc = jnp.broadcast_to(vc, (lb.shape[0], *vc.shape))
            Vc = (
                vc * weight_matrix
                if is_nonlocal
                else vc[..., None] * jnp.eye(vc.shape[-1])
            )
            V = V + C[b][:, :, None, None] * Vc[:, None]
        return lagrange.solve(
            mesh, lb, k2[e, b], M[e, b], V, H[e, b], Bc[e, b], is_open[e, b], weights
        )

    one = jax.checkpoint(one)
    out = jax.lax.map(
        lambda i: one(i // NB, i % NB), jnp.arange(NE * NB), batch_size=batch_size
    )
    return jax.tree.map(lambda x: x.reshape(NE, NB, *x.shape[1:]), out)
