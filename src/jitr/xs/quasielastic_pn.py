r"""Quasi-elastic ``(p,n)`` scattering to the isobaric analog state.

The proton (entrance) and neutron (exit, isobaric analog) channels are coupled
by the isovector Lane potential :math:`U_1`, a central plus spin-orbit term.
It conserves ``l`` and ``j``, so each J^pi block of a spin-1/2 projectile on
a 0+ target holds one proton and one neutron channel with the same ``(l, j)``:

.. math::
    \left[T_l + U_p - E_p\right] u_p + U_1 u_n = 0, \qquad
    \left[T_l + U_n - E_n\right] u_n + U_1 u_p = 0 .

:meth:`Workspace.cc` solves these coupled channels exactly; the (p,n) element
``S_np`` of each block's flux-normalized S-matrix gives the observables.
:meth:`Workspace.dwba` treats :math:`U_1` to first order: with proton and
neutron distorted waves ``u_p``, ``u_n`` (unit incoming amplitude, outgoing
boundary conditions, no complex conjugation),

.. math::
    T = \int u_p U_1 u_n \, dr, \qquad
    S_{np} = -\frac{4 i \sqrt{\mu_p \mu_n}}{\hbar^2 \sqrt{k_p k_n}}\, T,

the first-order term of the coupled-channels ``S_np`` (from the outgoing
Green's function of the neutron channel). Both methods then share the
amplitudes of :mod:`jitr.xs.amplitudes` (entrance proton, exit neutron),
giving ``dσ/dΩ = (1/2) Σ |f_m'm|^2``, ``A_y``, ``Q`` and the integrated cross
section ``(π/k_p^2) Σ_b (2j+1)/2 |S_np|^2``. When :math:`U_1` is not given, each of
its terms defaults to ``-(U_n - U_p) sqrt|N-Z| / (N - Z - 1)``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..reactions import Level, Reaction, spin_orbit
from ..utils.constants import HBARC
from ..utils.kinematics import ChannelKinematics
from . import amplitudes, coupled
from .coupled import Potential, potential_terms, promote

PROTON, NEUTRON = 0, 1


class QuasielasticPnXS(NamedTuple):
    """(p,n) observables: ``dsdo`` (mb/sr), ``Ay`` and ``Q`` of shape
    ``(NE, Nθ)``, the integrated cross section ``rxn_pn`` (mb), ``(NE,)``, and
    the amplitudes ``f[e, m', m, θ]`` (fm) of :mod:`jitr.xs.amplitudes`."""

    dsdo: jax.Array
    Ay: jax.Array
    Q: jax.Array
    rxn_pn: jax.Array
    f: jax.Array


def isovector_factor(reaction: Reaction) -> float:
    r"""Return :math:`\sqrt{|N-Z|}/(N-Z-1)` for the target of ``reaction``."""
    N, Z = reaction.target.A - reaction.target.Z, reaction.target.Z
    if N - Z == 1:
        raise ValueError(
            f"the (p,n) isovector factor diverges for N - Z = 1 targets like "
            f"{reaction.target}; supply U1 explicitly"
        )
    return float(np.sqrt(abs(N - Z)) / (N - Z - 1))


class Workspace:
    """(p,n) to the isobaric analog state at one or many energies.

    Args:
        reaction: The (p,n) reaction (target, projectile, product, residual).
        kinematics_entrance: Proton-channel kinematics (scalar or arrays).
        kinematics_exit: Neutron-channel kinematics, same shape.
        channel_radius_fm: Channel radius in fm, shared by both channels.
        lmax: Maximum orbital angular momentum.
        nbasis: Number of Lagrange mesh points.
        angles: Optional c.m. angles in radians for the angular observables.
        batch_size: Passed to :class:`jitr.xs.coupled.Workspace`.
    """

    def __init__(
        self,
        reaction: Reaction,
        kinematics_entrance: ChannelKinematics,
        kinematics_exit: ChannelKinematics,
        channel_radius_fm: float,
        lmax: int,
        nbasis: int,
        angles=None,
        batch_size=None,
    ):
        if reaction.residual is None or reaction.product is None:
            raise ValueError("(p,n) needs a reaction with product and residual")
        self.reaction = reaction
        self.kinematics_entrance = kinematics_entrance
        self.kinematics_exit = kinematics_exit
        self.lmax = lmax
        level = Level(0, 1, 0.5)
        kin = (kinematics_entrance, kinematics_exit)
        k, mu, eta = (
            np.stack(
                [
                    np.broadcast_to(getattr(x, q), np.shape(kinematics_entrance.k))
                    for x in kin
                ],
                axis=-1,
            ).reshape(-1, 2)
            for q in ("k", "mu", "eta")
        )
        args = (channel_radius_fm, nbasis)
        # DWBA distorted waves: one-channel blocks per particle; CC: both
        self.p = coupled.Workspace(
            *args,
            [level],
            lmax,
            k[:, :1] ** 2,
            mu[:, :1],
            eta[:, :1],
            batch_size=batch_size,
        )
        self.n = coupled.Workspace(
            *args,
            [level],
            lmax,
            k[:, 1:] ** 2,
            mu[:, 1:],
            eta[:, 1:],
            batch_size=batch_size,
        )
        self.lane = coupled.Workspace(
            *args, [level, level], lmax, k**2, mu, eta, batch_size=batch_size
        )
        b = self.p.blocks
        assert np.array_equal(b.l[:, 0], self.lane.blocks.l[:, 0])
        assert np.array_equal(b.j[:, 0], self.lane.blocks.j[:, 1])
        self.l, self.j = b.l[:, 0], b.j[:, 0]
        self.ls = spin_orbit(b, [level])  # (NB, 1, 1): l.sigma of each block
        self.k, self.mu, self.eta = k, mu, eta
        self.born = (
            -4j * np.sqrt(mu[:, 0] * mu[:, 1]) / (HBARC**2 * np.sqrt(k[:, 0] * k[:, 1]))
        )
        self.angles = angles
        if angles is not None:
            self._tables = amplitudes.transition_tables(
                self.lane, PROTON, NEUTRON, angles
            )

    @property
    def r(self):
        """Mesh points in fm, the grid on which potentials are evaluated."""
        return self.p.r

    def transition(self, Up: Potential, Un: Potential, U1: Potential | None = None):
        """The Lane coupling ``U1``: given terms as is, missing ones from the
        isovector difference of the neutron and proton potentials."""
        if Un.central is None:
            raise TypeError("the neutron central potential Un.central is required")
        U1 = U1 or Potential()
        terms = {}
        for name in (
            "central",
            "spin_orbit",
            "nonlocal_central",
            "nonlocal_spin_orbit",
        ):
            given, p, n = getattr(U1, name), getattr(Up, name), getattr(Un, name)
            if given is None and (p is not None or n is not None):
                given = -(
                    (0 if n is None else n) - (0 if p is None else p)
                ) * isovector_factor(self.reaction)
            terms[name] = given
        return Potential(**terms)

    def smatrix(self, Up: Potential, Un: Potential, U1: Potential | None = None):
        """Coupled-channels S-matrix ``(NE, NB, 2, 2)``, channels ``(p, n)``."""
        U1 = self.transition(Up, Un, U1)
        P, N, X = (
            np.diag([1.0, 0.0]),
            np.diag([0.0, 1.0]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        local, nonlocal_ = [], []
        for potential, C in (
            (Up, P),
            (Un._replace(coulomb=None), N),
            (U1._replace(coulomb=None), X),
        ):
            lo, nl = potential_terms(potential, C, self.ls)
            local += lo
            nonlocal_ += nl
        return self.lane.smatrix(local, nonlocal_)[0]

    def tmatrix(self, Up: Potential, Un: Potential, U1: Potential | None = None):
        """DWBA ``(T, Sp, Sn)``, each ``(NE, NB)``: the transition amplitude
        ``∫ u_p U1 u_n dr`` (MeV fm) and the elastic S of each particle."""
        U1 = self.transition(Up, Un, U1)
        one = np.ones((1, 1))
        Sp, up, _ = self.p.wavefunctions(
            *potential_terms(Up, one, self.ls), weights=one
        )
        Sn, un, _ = self.n.wavefunctions(
            *potential_terms(Un._replace(coulomb=None), one, self.ls), weights=one
        )
        up, un = up[:, :, 0, 0], un[:, :, 0, 0]  # (NE, NB, N), flux-normalized
        m, NE = self.p.mesh, up.shape[0]
        lmax = int(self.l.max())
        T = 0
        for v, C, is_nonlocal in (
            (U1.central, 1.0, False),
            (U1.spin_orbit, self.ls[:, 0, 0], False),
            (U1.nonlocal_central, 1.0, True),
            (U1.nonlocal_spin_orbit, self.ls[:, 0, 0], True),
        ):
            if v is None:
                continue
            v = promote(v, is_nonlocal, NE, lmax)  # (L, NE or 1, N[, N])
            v = jnp.swapaxes(v[self.l] if v.shape[0] > 1 else v, 0, 1)
            if is_nonlocal:
                ww = jnp.outer(m.w, m.w)
                integral = m.a**2 * jnp.sum(
                    up[..., :, None] * ww * v * un[..., None, :], (-2, -1)
                )
            else:
                integral = m.a * jnp.sum(m.w * up * v * un, axis=-1)
            T = T + C * integral
        # flux-normalized waves are k^{-1/2} times unit-amplitude ones
        T = jnp.sqrt(self.k[:, :1] * self.k[:, 1:]) * T
        return T, Sp[:, :, 0, 0], Sn[:, :, 0, 0]

    def cc(self, Up: Potential, Un: Potential, U1: Potential | None = None):
        """Observables from the exact coupled-channels solution."""
        return self.xs_from_smatrix(self.smatrix(Up, Un, U1)[:, :, NEUTRON, PROTON])

    def dwba(self, Up: Potential, Un: Potential, U1: Potential | None = None):
        """Observables to first order in ``U1``."""
        T, _, _ = self.tmatrix(Up, Un, U1)
        return self.xs_from_smatrix(self.born[:, None] * T)

    def xs_from_smatrix(self, S_np, angles=None) -> QuasielasticPnXS:
        """Observables from the (p,n) element of S, ``(NE, NB)``, at the
        workspace angles or at ``angles``."""
        if angles is None and self.angles is None:
            raise ValueError("no angles: pass them here or to the workspace")
        tables = (
            self._tables
            if angles is None
            else amplitudes.transition_tables(self.lane, PROTON, NEUTRON, angles)
        )
        return _pn_xs(S_np, tables)


@jax.jit
def _pn_xs(S_np, tables) -> QuasielasticPnXS:
    # S[exit, entrance]: the neutron row of the proton column
    S = jnp.zeros((*S_np.shape, 2, 2), complex).at[..., NEUTRON, PROTON].set(S_np)
    xs = amplitudes.observables(S, tables)
    Ay, Q = amplitudes.spin_half_polarization(xs)
    return QuasielasticPnXS(dsdo=xs.dsdo, Ay=Ay, Q=Q, rxn_pn=xs.sigma, f=xs.f)
