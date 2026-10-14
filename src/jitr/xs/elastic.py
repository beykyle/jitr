r"""Elastic scattering of a spin-0 or spin-1/2 projectile on a spin-0 target.

Every J^pi block holds a single channel ``(l, j)``, so the elastic S-matrix is
one number per block, ``S_b`` (shape ``(NE, NB)``). With the Coulomb amplitude
``f_c`` and Coulomb phases ``σ_l``, the amplitudes are

.. math::
    A(θ) = f_c(θ) + \sum_b \frac{2j+1}{2s+1}\,
        \frac{e^{2iσ_l}}{2ik} (S_b - 1) P_l(\cos θ), \qquad
    B(θ) = \sum_b \pm \frac{e^{2iσ_l}}{2ik} S_b P^1_l(\cos θ),

with ``+`` for ``j = l + 1/2``, ``-`` for ``j = l - 1/2`` and ``B = 0`` for a
spin-0 projectile; for spin 1/2 these are the familiar
``(l+1)(S_+ - 1) + l (S_- - 1)`` and ``S_+ - S_-`` sums. Then
``dσ/dΩ = |A|^2 + |B|^2``, ``A_y = 2 Im(A^* B) / dσ/dΩ`` and
``Q = 2 Re(A^* B) / dσ/dΩ``, and with the same block weights

.. math::
    σ_R = \frac{π}{k^2} \sum_b \frac{2j+1}{2s+1} (1 - |S_b|^2), \qquad
    σ_T = \frac{2π}{k^2} \sum_b \frac{2j+1}{2s+1} (1 - \mathrm{Re}\, S_b).

Cross sections are in mb (``10 fm^2``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import eval_legendre, gamma, lpmv

from ..reactions import Level, Reaction, spin_orbit
from ..utils.kinematics import ChannelKinematics, level_kinematics
from . import coupled
from .coupled import Potential, potential_terms

#: projectile spins by (A, Z): nucleons, triton and helion are spin 1/2
SPIN_HALF = {(1, 0), (1, 1), (3, 1), (3, 2)}


class ElasticXS(NamedTuple):
    """Elastic observables; angular ones have shape ``(NE, Nθ)``, integrated
    ones ``(NE,)``.

    Attributes:
        dsdo: Differential cross section in mb/sr.
        Ay: Analyzing power.
        Q: Spin-rotation function.
        t: Total cross section in mb.
        rxn: Reaction cross section in mb.
    """

    dsdo: jax.Array
    Ay: jax.Array
    Q: jax.Array
    t: jax.Array
    rxn: jax.Array


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
        self.s = 0.5 if projectile in SPIN_HALF else 0.0
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
        b = self.solver.blocks
        self.l, self.j = b.l[:, 0], b.j[:, 0]
        self.k, self.eta = np.sqrt(k2[:, 0]), eta[:, 0]
        self.spin_orbit = spin_orbit(b, self.solver.levels)
        self.weight = (2 * self.j + 1) / (2 * self.s + 1)
        self.sign = np.sign(self.j - self.l) if self.s else 0 * self.l
        self.sigma_l = np.angle(gamma(1 + self.l + 1j * self.eta[:, None]))
        self.zz = reaction.target.Z * reaction.projectile.Z
        self.angles = None
        if angles is not None:
            self.angles, self._tables = angles, self._angular_tables(angles)
            self.rutherford = self._tables[3]

    @property
    def r(self):
        """Mesh points in fm, the grid on which potentials are evaluated."""
        return self.solver.r

    def smatrix(self, potential: Potential):
        """Elastic S-matrix of each block, shape ``(NE, NB)``."""
        if self.s == 0 and (
            potential.spin_orbit is not None
            or potential.nonlocal_spin_orbit is not None
        ):
            raise ValueError("a spin-0 projectile has no spin-orbit interaction")
        local, nonlocal_ = potential_terms(potential, np.eye(1), self.spin_orbit)
        S, _ = self.solver.smatrix(local, nonlocal_)
        return S[:, :, 0, 0]

    def xs(self, potential: Potential) -> ElasticXS:
        """Observables at the workspace angles."""
        return self.xs_from_smatrix(self.smatrix(potential))

    def xs_from_smatrix(self, S, angles=None):
        """Observables from ``S`` at the workspace angles, at ``angles`` (an
        array) or, for a list of one angle array per energy, a list of
        :class:`ElasticXS` (one per energy, each with ``NE = 1``)."""
        if isinstance(angles, list):
            return [
                _observables(
                    S[e : e + 1], *self._angular_tables(a, e)[:3], *self._integral(e)
                )
                for e, a in enumerate(angles)
            ]
        if angles is None and self.angles is None:
            raise ValueError("no angles: pass them here or to the workspace")
        tables = self._tables if angles is None else self._angular_tables(angles)
        return _observables(S, *tables[:3], *self._integral())

    def to_lj(self, S):
        """``S`` indexed by ``[energy, l, j]`` with ``j = l + 1/2, l - 1/2``
        (spin 1/2; the undefined ``l = 0, j = -1/2`` entry is 1), or by
        ``[energy, l]`` for spin 0. Transmission coefficients are
        ``1 - |to_lj(S)|^2``."""
        if self.s == 0:
            return S[:, np.argsort(self.l)]
        out = jnp.ones((S.shape[0], self.lmax + 1, 2), S.dtype)
        return out.at[:, self.l, (self.j < self.l).astype(int)].set(S)

    def _integral(self, e=None):
        sel = slice(None) if e is None else slice(e, e + 1)
        return self.k[sel], self.weight

    def _angular_tables(self, angles, e=None):
        """``(A, B)`` partial-wave coefficients times Legendre functions,
        ``(NE, NB, Nθ)``, plus ``f_c`` and the Rutherford cross section."""
        sel = slice(None) if e is None else slice(e, e + 1)
        k, eta, sigma = self.k[sel, None], self.eta[sel, None], self.sigma_l[sel]
        x = np.cos(angles)
        phase = np.exp(2j * sigma) / (2j * k)
        P = eval_legendre(self.l[:, None], x)
        P1 = lpmv(1, self.l[:, None], x)
        A = (self.weight * phase)[..., None] * P
        B = (self.sign * phase)[..., None] * P1
        sin2 = np.sin(angles / 2) ** 2
        if self.zz == 0:
            return A, B, np.zeros((len(k), len(x))), None
        f_c = (
            -eta
            / (2 * k * sin2)
            * np.exp(2j * sigma[:, :1] - 2j * eta * np.log(np.sqrt(sin2)))
        )
        return A, B, f_c, 10 * eta**2 / (4 * k**2 * sin2**2)


@jax.jit
def _observables(S, A, B, f_c, k, weight):
    a = f_c + jnp.einsum("eb,ebt->et", S - 1, A)
    b = jnp.einsum("eb,ebt->et", S, B)
    dsdo = jnp.abs(a) ** 2 + jnp.abs(b) ** 2
    denom = jnp.maximum(dsdo, 1e-30)
    rxn = jnp.pi / k**2 * jnp.sum(weight * (1 - jnp.abs(S) ** 2), axis=1)
    t = 2 * jnp.pi / k**2 * jnp.sum(weight * (1 - S.real), axis=1)
    return ElasticXS(
        dsdo=10 * dsdo,
        Ay=2 * jnp.imag(jnp.conj(a) * b) / denom,
        Q=2 * jnp.real(jnp.conj(a) * b) / denom,
        t=10 * t,
        rxn=10 * rxn,
    )


def check_angles(angles) -> None:
    """Validate that the angle grid is one-dimensional and lies on ``[0, π]``."""
    if angles.ndim != 1:
        raise ValueError("angles must be a 1D array")
    if angles[0] < 0 or angles[-1] > np.pi:
        raise ValueError("angles must a grid in radians on [0,pi)")
