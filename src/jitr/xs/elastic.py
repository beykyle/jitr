r"""Elastic scattering of a projectile of any spin on a spin-0 target.

The S-matrix of each J^pi block comes from :class:`jitr.xs.coupled.Workspace`
and the observables from :mod:`jitr.xs.amplitudes` (Thompson, Eqs. 38-41):
the differential cross section, the spherical tensor analyzing powers
``T_kq`` and, for spin 1/2, the analyzing power ``A_y = \sqrt 2\, iT_{11}`` and
the spin-rotation function ``Q``. With block weights ``(2J+1)/(2s+1)``,

.. math::
    σ_R = \frac{π}{k^2} \sum_c \frac{2J+1}{2s+1} \Big(1 - \sum_{c'} |S_{c'c}|^2\Big), \qquad
    σ_T = \frac{2π}{k^2} \sum_c \frac{2J+1}{2s+1} (1 - \mathrm{Re}\, S_{cc}).

Projectile spins: 1/2 for nucleons, tritons and helions, 1 for deuterons, 0
otherwise. The spin-orbit potential multiplies ``2 l.s`` (``l.σ`` for spin
1/2). Cross sections are in mb (``10 fm^2``).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..reactions import Level, Reaction, spin_orbit
from ..utils.kinematics import ChannelKinematics, level_kinematics
from . import amplitudes, coupled
from .coupled import Potential, potential_terms

#: projectile spins by (A, Z); all others are taken as spin 0
SPINS = {(1, 0): 0.5, (1, 1): 0.5, (3, 1): 0.5, (3, 2): 0.5, (2, 1): 1.0}


class ElasticXS(NamedTuple):
    """Elastic observables; angular ones have shape ``(NE, Nθ)``, integrated
    ones ``(NE,)``.

    Attributes:
        dsdo: Differential cross section in mb/sr.
        Ay: Analyzing power (spin 1/2; zero otherwise).
        Q: Spin-rotation function (spin 1/2; zero otherwise).
        t: Total cross section in mb (meaningful for neutral projectiles).
        rxn: Reaction cross section in mb.
        T: Analyzing powers ``T_kq``, ``(NE, n_kq, Nθ)``, ordered as
            :func:`jitr.xs.amplitudes.kq_labels`.
    """

    dsdo: jax.Array
    Ay: jax.Array
    Q: jax.Array
    t: jax.Array
    rxn: jax.Array
    T: jax.Array


class Workspace:
    """Elastic scattering at one or many energies.

    Args:
        reaction: The elastic reaction.
        kinematics: Entrance kinematics, scalar or arrays over ``NE`` energies.
        channel_radius_fm: Channel radius in fm, shared by every energy.
        lmax: Maximum orbital angular momentum.
        nbasis: Number of Lagrange mesh points.
        angles: Optional c.m. angles in radians for the angular observables.
        batch_size: Passed to :class:`jitr.xs.coupled.Workspace`.
    """

    def __init__(
        self,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        channel_radius_fm: float,
        lmax: int,
        nbasis: int,
        angles=None,
        batch_size=None,
    ):
        projectile = (reaction.projectile.A, reaction.projectile.Z)
        if projectile == (reaction.target.A, reaction.target.Z):
            raise NotImplementedError("identical particles are not symmetrized")
        self.s = SPINS.get(projectile, 0.0)
        self.reaction = reaction
        self.kinematics = kinematics
        self.lmax = lmax
        k2, mu, eta = level_kinematics(kinematics, [0.0])
        self.solver = coupled.Workspace(
            channel_radius_fm,
            nbasis,
            [Level(0, 1, self.s)],
            lmax,
            k2,
            mu,
            eta,
            batch_size=batch_size,
        )
        self.spin_orbit = spin_orbit(self.solver.blocks, self.solver.levels)
        self.k, self.eta = np.sqrt(k2[:, 0]), eta[:, 0]
        self.angles = angles
        self.rutherford = None
        self._geometry = amplitudes.transition_geometry(self.solver, 0, 0)
        if angles is not None:
            self._tables = amplitudes.transition_tables(
                self.solver, self._geometry, angles
            )
            sin2 = np.sin(np.asarray(angles) / 2) ** 2
            self.rutherford = (
                10 * (self.eta[:, None] / (2 * self.k[:, None] * sin2)) ** 2
                if np.any(self.eta)
                else None
            )

    @property
    def r(self):
        """Mesh points in fm, the grid on which potentials are evaluated."""
        return self.solver.r

    @property
    def blocks(self):
        return self.solver.blocks

    def smatrix(self, potential: Potential):
        """S-matrix of each block, ``(NE, NB, nch, nch)`` (``nch = 1`` for
        spin 0 and 1/2)."""
        if self.s == 0 and (
            potential.spin_orbit is not None
            or potential.nonlocal_spin_orbit is not None
        ):
            raise ValueError("a spin-0 projectile has no spin-orbit interaction")
        local, nonlocal_ = potential_terms(
            potential, np.eye(self.solver.nch), self.spin_orbit
        )
        return self.solver.smatrix(local, nonlocal_)[0]

    def xs(self, potential: Potential) -> ElasticXS:
        """Observables at the workspace angles."""
        return self.xs_from_smatrix(self.smatrix(potential))

    def xs_from_smatrix(self, S, angles=None):
        """Observables from ``S`` at the workspace angles, at ``angles`` (an
        array) or, for a list of one angle array per energy, a list of
        :class:`ElasticXS` (one per energy, each with ``NE = 1``)."""
        if isinstance(angles, list):
            if len(angles) != S.shape[0]:
                raise ValueError("pass one angle array per energy")
            return [
                self._xs(
                    S[e : e + 1],
                    amplitudes.transition_tables(
                        self.solver, self._geometry, a, slice(e, e + 1)
                    ),
                    slice(e, e + 1),
                )
                for e, a in enumerate(angles)
            ]
        if angles is None and self.angles is None:
            raise ValueError("no angles: pass them here or to the workspace")
        if angles is None:
            return self._xs(S, self._tables)
        return self._xs(
            S, amplitudes.transition_tables(self.solver, self._geometry, angles)
        )

    def to_lj(self, S):
        """``S`` (spin 0 or 1/2) indexed by ``[energy, l, j]`` with
        ``j = l + 1/2, l - 1/2`` (the undefined ``l = 0, j = -1/2`` entry is
        1), or by ``[energy, l]`` for spin 0. Transmission coefficients are
        ``1 - |to_lj(S)|^2``. Larger spins have several channels per block
        and are not supported."""
        if self.s > 0.5:
            raise ValueError("to_lj supports projectile spin 0 and 1/2 only")
        l, j = self.blocks.l[:, 0], self.blocks.j[:, 0]
        S = S[:, :, 0, 0]
        if self.s == 0:
            return S[:, np.argsort(l)]
        out = jnp.ones((S.shape[0], self.lmax + 1, 2), S.dtype)
        return out.at[:, l, (j < l).astype(int)].set(S)

    def _xs(self, S, tables, energies=slice(None)) -> ElasticXS:
        weights = amplitudes.integral_weights(self.solver, 0, energies)
        return _elastic_xs(S, tables, weights, spin_half=self.s == 0.5)


@partial(jax.jit, static_argnames="spin_half")
def _elastic_xs(S, tables, weights, spin_half) -> ElasticXS:
    xs = amplitudes.observables(S, tables)
    t, rxn = amplitudes._integrals(S, *weights)
    if spin_half:
        Ay, Q = amplitudes.spin_half_polarization(xs)
    else:
        Ay = Q = jnp.zeros_like(xs.dsdo)
    return ElasticXS(dsdo=xs.dsdo, Ay=Ay, Q=Q, t=t, rxn=rxn, T=xs.T)
