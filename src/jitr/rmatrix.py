r"""R-matrix solution of coupled radial equations on a Lagrange-Legendre mesh.

One :class:`Mesh` of ``N`` points ``r_n = a x_n`` on ``[0, a]`` (``a`` in fm)
serves every channel and every energy. The basis functions are the
x-regularized Lagrange-Legendre functions of Baye, Phys. Rep. 565 (2015),
Eq. 3.122, normalized on ``[0, a]``:

.. math::
    \varphi_n(r_m) = \frac{\delta_{nm}}{\sqrt{a w_n}}, \qquad
    \varphi_n(a) = \frac{b_n}{\sqrt a}, \qquad
    b_n = \frac{(-1)^{N-n}}{\sqrt{x_n (1 - x_n)}},

with ``(x_n, w_n)`` the Gauss-Legendre nodes and weights on ``(0, 1)``.

**Interior problem.** In a block of channels ``c`` with orbital angular
momentum ``l_c``, wavenumber ``k_c`` (``k_c^2 < 0`` when closed) and reduced
mass ``mu_c``, the radial equations are

.. math::
    \left[-\frac{d^2}{dr^2} + \frac{l_c(l_c+1)}{r^2} - k_c^2\right] u_c
    + \sum_{c'} \frac{2\mu_c}{\hbar^2} V_{cc'} u_{c'} = 0 .

Writing ``u_c = sqrt(mu_c) ũ_c`` makes the coupling symmetric,
``M_cc' = 2 sqrt(mu_c mu_c') / (ħc)^2``. Adding the Bloch operator
``L_c = δ(r - a)(d/dr - B_c/a)`` to both sides (Descouvemont, Comput. Phys.
Commun. 200 (2016) 199, Sec. 2) and projecting on the basis gives

.. math::
    \tilde A c = \frac{b}{\sqrt a}\,\left(\tilde u'(a) - \frac{B}{a}\tilde u(a)\right),
    \qquad
    \tilde A_{cc'} = \delta_{cc'}\left[\frac{F}{a^2}
        + \frac{l_c(l_c+1)}{r^2} - k_c^2 - \frac{B_c}{a^2} b b^T\right]
        + M_{cc'} \mathcal V_{cc'} ,

where ``F`` is the matrix of ``-d^2/dx^2 + δ(x - 1) d/dx`` at ``a = 1`` (Baye,
Eqs. 3.128-3.129), exact under ``r = a x`` because both terms scale as
``1/a^2``. ``𝒱`` is the Lagrange matrix of the interaction in MeV:
``diag(v(r_n))`` for a local term and ``a sqrt(w_n w_m) v(r_n, r_m)`` for a
nonlocal one (Gauss quadrature of ``<φ_n|V|φ_m>``).

**Open channels** use ``B_c = 0``; the right-hand side is then ``φ(a) ũ'(a)``
and ``ũ(a) = a R ũ'(a)`` with the dimensionless, symmetric R-matrix

.. math::
    R = \frac{1}{a^2}\, b^T \tilde A^{-1} b .

**Closed channels** use ``B_c = z W'(z) / W(z)`` at ``z = 2 κ_c a``, the
logarithmic derivative of their decaying Whittaker tail, so their right-hand
side vanishes identically and they drop out of the matching while still
coupling in the interior. Eliminating them from the B=0 R-matrix instead gives
``R_oo + R_oc B (1 - R_cc B)^{-1} R_co`` (Woodbury), which is the same thing.

**Matching.** Outside ``a`` an open channel with flux-normalized incoming
amplitudes ``w`` is ``ũ_c = k_c^{-1/2} (i/2)[w_c H^-_c - (S w)_c H^+_c]`` with
``H^± = G ± iF`` at ``ρ = k_c r``. Inserting it in ``ũ(a) = a R ũ'(a)`` and
multiplying by ``K = diag(sqrt k)`` gives

.. math::
    S = Z_+^{-1} Z_-, \qquad
    Z_\pm = \mathrm{diag}(H^\pm) - a\,K R K\,\mathrm{diag}(H^{\pm\prime}),

with ``'`` the derivative with respect to ``ρ``. This S is flux-normalized
(unitary and symmetric for real potentials) for any per-channel ``k``, ``mu``
and Sommerfeld parameter; ``H^-`` is used explicitly rather than ``conj(H^+)``.

**Wavefunctions.** The interior coefficients for incoming weights ``w`` are
``c = Ã^{-1} (b/√a) ũ'(a)`` with
``ũ'_c(a) = sqrt(k_c) (i/2)[w_c H^-'_c - (S w)_c H^+'_c]``, so
``ũ_c(r_n) = c_{c,n} / sqrt(a w_n)``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sc


class Mesh(NamedTuple):
    """Lagrange-Legendre mesh on ``[0, a]``: nodes, weights, kinetic matrix
    ``F`` (with Bloch term, at ``a = 1``) and boundary values ``b``."""

    x: np.ndarray
    w: np.ndarray
    a: float
    F: np.ndarray
    b: np.ndarray

    @property
    def r(self) -> np.ndarray:
        """Mesh points in fm."""
        return self.a * self.x


def mesh(n: int, a: float) -> Mesh:
    """Lagrange-Legendre mesh with ``n`` points on ``[0, a]`` (``a`` in fm)."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    i = np.arange(1, n + 1)
    sign = (-1.0) ** np.add.outer(i, i)
    xn, xm = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (
            sign
            * (
                n**2
                + n
                + 1.0
                + (xn + xm - 2 * xn * xm) / (xn - xm) ** 2
                - 1 / (1 - xn)
                - 1 / (1 - xm)
            )
            / np.sqrt(xn * xm * (1 - xn) * (1 - xm))
        )
    F[np.diag_indices(n)] = ((4 * n**2 + 4 * n + 3) * x * (1 - x) - 6 * x + 1) / (
        3 * x**2 * (1 - x) ** 2
    )
    b = (-1.0) ** (n - i) / np.sqrt(x * (1 - x))
    return Mesh(x, w, float(a), F, b)


def interaction(mesh: Mesh, local=None, nonlocal_=None):
    """Lagrange matrix ``(..., N, N)`` of a local ``v(r_n)`` and/or nonlocal
    ``v(r_n, r_m)`` interaction, in the units of ``v``."""
    N = mesh.x.size
    V = jnp.zeros((N, N))
    if local is not None:
        V = V + local[..., :, None] * jnp.eye(N)
    if nonlocal_ is not None:
        V = V + mesh.a * jnp.sqrt(jnp.outer(mesh.w, mesh.w)) * nonlocal_
    return V


@jax.jit
def solve(mesh: Mesh, l, k2, M, V, H, Bc, is_open, weights=None):
    r"""Solve one block of ``nch`` channels at one energy.

    Args:
        mesh: The Lagrange-Legendre mesh.
        l: Orbital angular momentum of each channel, shape ``(nch,)``.
        k2: Squared wavenumber in fm^-2, negative for closed channels, ``(nch,)``.
        M: Symmetric mass factor ``2 sqrt(mu_c mu_c') / (ħc)^2`` in
            MeV^-1 fm^-2, ``(nch, nch)``.
        V: Lagrange matrix of the interaction in MeV, ``(nch, nch, N, N)``.
        H: ``(H^+, H^-, H^+', H^-')`` at ``ρ = k_c a``, shape ``(4, nch)``;
            ignored for channels that are not open.
        Bc: Bloch constant of each channel (0 for open ones), ``(nch,)``.
        is_open: Mask of open channels, ``(nch,)``.
        weights: Optional flux-normalized incoming amplitudes ``(nch, nw)``;
            if given, the interior coefficients are returned too.

    Returns:
        The flux-normalized S-matrix, zero outside the open-open block, and
        if ``weights`` is given the coefficients ``(nw, nch, N)`` so that
        ``ũ_c(r_n) = c[:, c, n] / sqrt(a w_n)``.
    """
    nch, N, a = l.shape[0], mesh.x.size, mesh.a
    r = a * mesh.x
    eye = jnp.eye(nch)
    diagonal = l[:, None] * (l[:, None] + 1) / r**2 - k2[:, None]
    A = (
        M[:, :, None, None] * V
        + eye[:, :, None, None] * (mesh.F / a**2)
        - (eye * Bc)[:, :, None, None] * jnp.outer(mesh.b, mesh.b) / a**2
        + eye[:, :, None, None] * (diagonal[:, None, :, None] * jnp.eye(N))
    )
    A = A.transpose(0, 2, 1, 3).reshape(nch * N, nch * N)
    B = (eye[:, None, :] * (mesh.b / jnp.sqrt(a))[None, :, None]).reshape(nch * N, nch)
    X = jnp.linalg.solve(A, B)
    R = B.T @ X / a

    # closed channels: unit H, zero H', so they sit in Z as identity rows
    Hp, Hm, Hpp, Hmp = jnp.where(is_open, H, jnp.array([1, 1, 0, 0])[:, None])
    K = jnp.sqrt(jnp.sqrt(jnp.where(is_open, k2, 1.0)))
    KRK = a * K[:, None] * R * K[None, :]
    S = jnp.linalg.solve(jnp.diag(Hp) - KRK * Hpp, jnp.diag(Hm) - KRK * Hmp)
    S = S * (is_open[:, None] & is_open[None, :])
    if weights is None:
        return S
    du = K[:, None] * 0.5j * (Hmp[:, None] * weights - Hpp[:, None] * (S @ weights))
    return S, (X @ du).T.reshape(-1, nch, N)


def integrate(mesh: Mesh, f):
    """Gauss quadrature of ``∫_0^a f(r) dr`` from ``f(r_n)``."""
    return mesh.a * jnp.sum(mesh.w * f, axis=-1)


def integrate2(mesh: Mesh, f):
    """Gauss quadrature of ``∫∫_0^a f(r, r') dr dr'`` from ``f(r_n, r_m)``."""
    return mesh.a**2 * jnp.einsum("n,m,...nm->...", mesh.w, mesh.w, f)


def fourier_bessel(mesh: Mesh, l: int, f, k):
    r"""``∫_0^a j_l(k r) r^2 f(r) dr`` at each ``k`` (host-side, numpy)."""
    r = mesh.r
    return mesh.a * (sc.spherical_jn(l, np.outer(k, r)) * r**2 * mesh.w) @ f


def fourier_bessel2(mesh: Mesh, l: int, f, k):
    r"""``(2/π) ∫∫ j_l(k r) r^2 f(r, r') r'^2 j_l(k' r') dr dr'`` on ``k × k'``."""
    r = mesh.r
    g = sc.spherical_jn(l, np.outer(k, r)) * r**2 * mesh.w
    return (2 / np.pi) * mesh.a**2 * g @ f @ g.T


def laguerre(n: int, h: float):
    r"""Lagrange-Laguerre mesh for bound states: nodes ``h x_n``, Gauss-Laguerre
    weights, kinetic matrix (``-d^2/dr^2``, exact integrals, Baye Eqs.
    3.75-3.77) and overlap (Eq. 3.71) of the regularized basis."""
    x, w = np.polynomial.laguerre.laggauss(n)
    i = np.arange(n)
    sign = (-1.0) ** np.subtract.outer(i, i)
    xn, xm = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        F = sign * (xn + xm) / np.sqrt(xn * xm) / (xn - xm) ** 2
    F[np.diag_indices(n)] = -(x**2 - 2 * (2 * n + 1) * x - 4) / (12 * x**2)
    F = (F - sign / (4 * np.sqrt(xn * xm))) / h**2
    overlap = np.eye(n) + sign / np.sqrt(np.outer(x, x))
    return h * x, w, F, overlap
