r"""Global nonlocal dispersive neutron optical potential of Morillon,
Blanchon, Romain & Arellano (MBRA).

Implements the potential of `B. Morillon, G. Blanchon, P. Romain and
H. F. Arellano, arXiv:2403.05843 (2024)`_ — the first global *nonlocal and
dispersive* optical model for neutron elastic scattering off spherical
nuclei, valid for 16 ≤ A ≤ 209 and incident energies from 1 keV to 250 MeV.
(The paper calls the model "NLD"; this module is named after the authors
because "NLD" conventionally means nuclear level density.)

The model is an extension of the Perey–Buck potential: every term except
the imaginary volume shares a Gaussian nonlocality form factor of range
``beta`` (Eq. 1-2 of the paper), and the energy dependence of the real
terms is generated entirely by dispersion relations acting on the
energy-dependent imaginary depths (Eq. 9). In partial waves the nonlocal
terms enter through the kernel (Eq. 7)

.. math::

   \nu_l(r, r') = \frac{4 r r'}{\sqrt{\pi}\beta^3}\,
       U\!\left(\frac{r + r'}{2}\right)
       e^{-(r^2 + r'^2)/\beta^2}\, i_l\!\left(\frac{2 r r'}{\beta^2}\right),

with :math:`i_l` the modified spherical Bessel function — an
*l-dependent* nonlocal kernel, built here as a ``(lmax+1, N, N)`` array
for the ``jitr.xs`` workspaces (pass ``l_dependent=True``).

Term structure (Eqs. 10-12):

- nonlocal real volume ``V_V^NL f(r̃, R, a)``,
- nonlocal surface ``(V_S^NL + ΔV_S^NL(E) + i W_S^NL(E)) (-4a f')``,
- nonlocal spin-orbit ``(V_so^NL + ΔV_so^NL(E) + i W_so^NL(E))
  (λ_π²/r̃) f' ⟨l·σ⟩`` (both real and imaginary parts nonlocal),
- **local** imaginary volume ``(ΔV_V^L(E) + i W_V^L(E)) f(r, R, a)``.

Dispersive-correction conventions (each validated against the
corresponding panel of Fig. 3 of the paper):

- ΔV_S^NL: once-subtracted principal-value dispersion integral of the
  asymmetric W_S^NL, computed numerically (matches panel (a) within
  ~1 MeV everywhere, including the ±6.5 MeV spike pair at E_F).
- ΔV_V^L: numerical once-subtracted dispersion of the Brown-Rho part
  plus the ECIS-06 closed-form partner of the Mahaux-Sartor √E tail
  (matches panel (b) within ~1.5 MeV for 0 < E < 100 MeV).
- ΔV_so^NL: closed form, symmetric about E_F (matches panel (c) exactly).

All corrections vanish at the Fermi energy. The Fermi energy is
``E_F = -[S_n(Z, N) + S_n(Z, N+1)]/2``, available as ``Reaction.Ef`` in
jitr. All energies are in the CM frame (MeV).

.. _B. Morillon, G. Blanchon, P. Romain and H. F. Arellano,
   arXiv:2403.05843 (2024): https://arxiv.org/abs/2403.05843
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from scipy.special import ive

from .._types import ArrayOrScalar, ComplexArray, FloatArray
from ..utils.constants import WAVENUMBER_PION
from .dispersion import build_quadrature
from .potential_forms import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)

#: (ħ/m_π c)² in fm², the conventional spin-orbit scale.
LAMBDA_PI2 = 1.0 / WAVENUMBER_PION**2

PARAM_NAMES: tuple[str, ...] = (
    # nonlocal real depths, X = X_0 + X_A * A   (Table I)
    "vv_0",
    "vv_A",
    "vs_0",
    "vs_A",
    "vso_0",
    "vso_A",
    # nonlocal surface imaginary depth (Eq. 13)
    "as_plus_0",
    "as_plus_A",
    "as_minus",
    "bs",
    "cs",
    # local volume imaginary depth (Eqs. 14-16)
    "av_plus_0",
    "av_plus_A",
    "av_minus_0",
    "av_minus_A",
    "bv",
    "ev_plus_0",
    "ev_plus_A",
    "ev_minus",
    "alpha_0",
    "alpha_A",
    # nonlocal spin-orbit imaginary depth (Eq. 17)
    "aso",
    "bso",
    "cso",
    "dso",
    # geometry (Table II): r0 linear above A=70, cubic below
    "r0_0",
    "r0_A",
    "r0_c0",
    "r0_c1",
    "r0_c2",
    "r0_c3",
    "a_0",
    "a_A",
    "beta",
)

#: Published global parameters (Tables I & II of arXiv:2403.05843).
#:
#: Note on ``av_minus``: Table I prints the sub-Fermi volume depth as
#: "−8.400A". Read literally (−8.400·A) it gives W_V^L ≈ −110 MeV and
#: |ΔV_V^L| ≈ 80 MeV for ²⁰⁸Pb — grossly inconsistent with the ±10 MeV
#: scale of the paper's own Fig. 3(b) — so the printed value cannot be a
#: mass-proportional depth. The default here, a constant −48.40 MeV
#: (split as ``av_minus_0 + av_minus_A * A`` so other readings remain
#: representable), uniquely reproduces the published W_V^L curves: the
#: −2.8…−3.0 MeV dip near E ≈ −60 MeV with the near-zero spread between
#: ⁴⁰Ca, ⁸⁹Y and ²⁰⁸Pb seen in Fig. 3(b).
DEFAULT_PARAMS: tuple[float, ...] = (
    # real depths
    -69.71,
    -1.140e-2,
    -8.600,
    -8.000e-3,
    -9.787,
    -1.140e-2,
    # surface imaginary
    -19.62,
    -1.500e-2,
    -16.00,
    11.11,
    9.200e-3,
    # volume imaginary
    -32.40,
    -2.000e-2,
    -48.40,
    0.0,
    135.0,
    40.00,
    -9.000e-2,
    25.50,
    3.000e-1,
    2.000e-3,
    # spin-orbit imaginary
    4.893,
    2.447,
    50.00,
    3.900,
    # geometry
    1.1446,
    2.4200e-4,
    9.4860e-1,
    8.8000e-3,
    -1.3200e-4,
    7.1000e-7,
    6.1600e-1,
    -1.8200e-4,
    0.915,
)


def get_param_names() -> list[str]:
    """Return the MBRA parameter names in ``calculate_params`` order."""
    return list(PARAM_NAMES)


def get_default_params() -> tuple[float, ...]:
    """Return the published global parameter vector (Tables I & II)."""
    return DEFAULT_PARAMS


class Coefficients(NamedTuple):
    """Mass-resolved depths and geometry for one target nucleus."""

    VV: float
    VS: float
    VSO: float
    AS_plus: float
    AS_minus: float
    BS: float
    CS: float
    AV_plus: float
    AV_minus: float
    BV: float
    EV_plus: float
    EV_minus: float
    ALPHA: float
    ASO: float
    BSO: float
    CSO: float
    DSO: float
    R: float
    a: float
    beta: float


def reduced_radius(
    A: int,
    r0_0: float,
    r0_A: float,
    r0_c0: float,
    r0_c1: float,
    r0_c2: float,
    r0_c3: float,
) -> float:
    """Return the reduced radius r0(A) in fm (Table II, piecewise at A=70)."""
    if A > 70:
        return r0_0 + r0_A * A
    return r0_c0 + r0_c1 * A + r0_c2 * A**2 + r0_c3 * A**3


def resolve_coefficients(A: int, *params: float) -> Coefficients:
    """Resolve the global parameter vector to per-nucleus coefficients.

    Args:
        A: Target mass number.
        *params: Global parameters in :func:`get_param_names` order;
            defaults to the published values when omitted.

    Returns:
        Mass-resolved depths and geometry.
    """
    if len(params) == 0:
        params = DEFAULT_PARAMS
    if len(params) != len(PARAM_NAMES):
        raise ValueError(
            f"MBRA expects {len(PARAM_NAMES)} parameters in get_param_names() "
            f"order, got {len(params)}."
        )
    p = dict(zip(PARAM_NAMES, params, strict=True))
    r0 = reduced_radius(
        A, p["r0_0"], p["r0_A"], p["r0_c0"], p["r0_c1"], p["r0_c2"], p["r0_c3"]
    )
    return Coefficients(
        VV=p["vv_0"] + p["vv_A"] * A,
        VS=p["vs_0"] + p["vs_A"] * A,
        VSO=p["vso_0"] + p["vso_A"] * A,
        AS_plus=p["as_plus_0"] + p["as_plus_A"] * A,
        AS_minus=p["as_minus"],
        BS=p["bs"],
        CS=p["cs"],
        AV_plus=p["av_plus_0"] + p["av_plus_A"] * A,
        AV_minus=p["av_minus_0"] + p["av_minus_A"] * A,
        BV=p["bv"],
        EV_plus=p["ev_plus_0"] + p["ev_plus_A"] * A,
        EV_minus=p["ev_minus"],
        ALPHA=p["alpha_0"] + p["alpha_A"] * A,
        ASO=p["aso"],
        BSO=p["bso"],
        CSO=p["cso"],
        DSO=p["dso"],
        R=r0 * A ** (1.0 / 3.0),
        a=p["a_0"] + p["a_A"] * A,
        beta=p["beta"],
    )


# -- imaginary depths (Eqs. 13-17) -------------------------------------------


def Ws_depth(
    E: ArrayOrScalar,
    Ef: float,
    AS_plus: float,
    AS_minus: float,
    BS: float,
    CS: float,
) -> ArrayOrScalar:
    r"""Nonlocal surface imaginary depth W_S^NL(E) (Eq. 13), in MeV."""
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = E_arr - Ef
    amp = np.where(x > 0.0, AS_plus, AS_minus)
    out = amp * x**2 * np.exp(-CS * np.abs(x)) / (x**2 + BS**2)
    return out.item() if np.ndim(E) == 0 else out


def Wv_depth(
    E: ArrayOrScalar,
    Ef: float,
    AV_plus: float,
    AV_minus: float,
    BV: float,
    EV_plus: float,
    EV_minus: float,
    ALPHA: float,
) -> ArrayOrScalar:
    r"""Local volume imaginary depth W_V^L(E) (Eqs. 14-16), in MeV.

    Brown-Rho near the Fermi energy, with the Mahaux-Sartor
    :math:`\alpha\sqrt{E}` high-energy behaviour above
    ``E_F + E_V^+`` and a suppression of the depth below ``E_F − E_V^−``.
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = E_arr - Ef
    out = np.where(x > 0.0, AV_plus, AV_minus) * x**2 / (x**2 + BV**2)

    high = E_arr > Ef + EV_plus
    if np.any(high):
        Eh = E_arr[high]
        y = Ef + EV_plus
        out[high] += ALPHA * (
            np.sqrt(Eh) + y ** (3.0 / 2.0) / (2.0 * Eh) - 1.5 * np.sqrt(y)
        )

    low = E_arr < Ef - EV_minus
    if np.any(low):
        xl = x[low] + EV_minus
        out[low] *= 1.0 - xl**2 / (xl**2 + EV_minus**2)

    return out.item() if np.ndim(E) == 0 else out


def Wso_depth(
    E: ArrayOrScalar,
    Ef: float,
    ASO: float,
    BSO: float,
    CSO: float,
    DSO: float,
) -> ArrayOrScalar:
    r"""Nonlocal spin-orbit imaginary depth W_so^NL(E) (Eq. 17), in MeV."""
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x2 = (E_arr - Ef) ** 2
    out = ASO * x2 / (x2 + CSO**2) - BSO * x2 / (x2 + DSO**2)
    return out.item() if np.ndim(E) == 0 else out


# -- dispersive corrections ----------------------------------------------------


def delta_Vso_depth(
    E: ArrayOrScalar,
    Ef: float,
    ASO: float,
    BSO: float,
    CSO: float,
    DSO: float,
) -> ArrayOrScalar:
    r"""Closed-form dispersive correction ΔV_so^NL(E) for Eq. 17, in MeV.

    The dispersion integral of a symmetric Brown-Rho form
    :math:`A x^2/(x^2 + C^2)` is :math:`A C x/(x^2 + C^2)`; following the
    paper's reference for this term (VanderKam, Weisel & Tornow,
    J. Phys. G 26, 1787 (2000)) the correction is taken symmetric about
    the Fermi energy, i.e. evaluated at :math:`|E - E_F|`. This is the
    only choice that reproduces Fig. 3(c) of arXiv:2403.05843 (even ΔV_so
    with +2.25 MeV maxima on *both* sides of E_F and +0.9 MeV wings).
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = np.abs(E_arr - Ef)
    out = ASO * CSO * x / (x**2 + CSO**2) - BSO * DSO * x / (x**2 + DSO**2)
    return out.item() if np.ndim(E) == 0 else out


@lru_cache(maxsize=8)
def _dispersion_quadrature(
    Ef: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Quadrature for the subtracted PV dispersion integral around ``Ef``."""
    segments = (
        (Ef - 3.0e4, Ef - 3.0e3, 64),
        (Ef - 3.0e3, Ef - 3.0e2, 64),
        (Ef - 3.0e2, Ef + 3.0e2, 256),
        (Ef + 3.0e2, Ef + 3.0e3, 64),
        (Ef + 3.0e3, Ef + 3.0e4, 64),
    )
    x_quad, w_quad, _ = build_quadrature(segments)
    return x_quad, w_quad, segments[0][0], segments[-1][1]


def dispersion_correction(
    W: Callable[[FloatArray], ArrayOrScalar],
    E: ArrayOrScalar,
    Ef: float,
) -> ArrayOrScalar:
    r"""Once-subtracted PV dispersion integral of an imaginary depth, in MeV.

    .. math::

       \Delta V(E) = \frac{E - E_F}{\pi}\,\mathcal{P}\!\int
           \frac{W(E')}{(E' - E)(E' - E_F)}\, dE',

    which satisfies ΔV(E_F) = 0 and converges for depths growing slower
    than linearly (the Mahaux-Sartor :math:`\sqrt{E}` tail included).

    Args:
        W: Vectorized imaginary depth ``W(E')``; must vanish at ``E_F``
            at least quadratically (true for Eqs. 13-17).
        E: Evaluation energies in MeV (CM).
        Ef: Fermi energy in MeV.

    Returns:
        ΔV(E), same shape as ``E``.
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x_quad, w_quad, lo, hi = _dispersion_quadrature(Ef)
    if np.any(E_arr <= lo) or np.any(E_arr >= hi):
        raise ValueError(
            f"dispersion_correction: E must lie inside the quadrature "
            f"interval ({lo:.1f}, {hi:.1f}) MeV"
        )

    g_quad = np.asarray(W(x_quad), dtype=float) / (x_quad - Ef)
    x = E_arr - Ef
    W_at_E = np.asarray(W(E_arr), dtype=float)
    g_at_E = np.divide(W_at_E, x, out=np.zeros_like(W_at_E), where=np.abs(x) > 1e-12)

    denom = x_quad[None, :] - E_arr[:, None]
    diff = g_quad[None, :] - g_at_E[:, None]
    ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=np.abs(denom) > 1e-9)
    pv = ratio @ w_quad + g_at_E * np.log((hi - E_arr) / (E_arr - lo))
    out = x * pv / np.pi
    return out.item() if np.ndim(E) == 0 else out


def delta_Vs_depth(
    E: ArrayOrScalar,
    Ef: float,
    AS_plus: float,
    AS_minus: float,
    BS: float,
    CS: float,
) -> ArrayOrScalar:
    r"""Dispersive correction ΔV_S^NL(E) to the surface depth, in MeV."""
    return dispersion_correction(
        lambda Ep: Ws_depth(Ep, Ef, AS_plus, AS_minus, BS, CS), E, Ef
    )


def _dlpe(el: float, E: ArrayOrScalar, Ef: float) -> FloatArray:
    r"""Dispersive partner of the Mahaux-Sartor sqrt(E) tail (per unit α).

    Closed form for the once-subtracted dispersion integral of the
    asymptotic term of Eq. 15,
    :math:`\sqrt{E} + e_l^{3/2}/(2E) - \tfrac{3}{2}\sqrt{e_l}` for
    ``E > e_l = E_F + E_V^+`` (zero below), transcribed from the ECIS-06
    routine ``dlpe`` (J. Raynal), which implements the same W_V^L form.
    Vanishes at ``E = E_F``.
    """
    ex = np.atleast_1d(np.asarray(E, dtype=float))
    ff = np.sqrt(abs(Ef))
    fl = np.sqrt(abs(el))
    fx = np.sqrt(np.abs(ex))
    base = (
        2.0 * ff * np.arctan2(ff, fl)
        + 0.5 * el * fl / Ef * np.log(1.0 - Ef / el)
        - 1.5 * fl * np.log(abs(el - Ef))
    )
    out = np.full_like(ex, base)

    pos = ex > 0.0
    neg = ~pos
    xn = ex[neg]
    small_n = np.abs(xn) <= el * 1e-5
    xn_safe = np.where(small_n, 1.0, xn)
    t_neg = np.where(
        small_n,
        0.5 * fl * (1.0 + xn / el / 2.0 + (xn / el) ** 2 / 3.0),
        -0.5 * el * fl / xn_safe * np.log(np.abs(1.0 - xn / el)),
    )
    out[neg] += (
        t_neg
        - 2.0 * fx[neg] * np.arctan2(fx[neg], fl)
        + 1.5 * fl * np.log(np.abs(el - xn))
    )

    xp = ex[pos]
    fxp = fx[pos]
    t_pos = (fxp + 1.5 * fl - 0.5 * el * fl / xp) * np.log(fl + fxp) + (
        el * fl / xp
    ) * np.log(fl)
    far = np.abs(xp - el) > 1e-3
    t_pos -= np.where(
        far,
        (fxp - 1.5 * fl + 0.5 * el * fl / xp)
        * np.log(np.where(far, np.abs(fl - fxp), 1.0)),
        0.0,
    )
    out[pos] += t_pos

    out = np.where(np.abs(ex - Ef) < 1e-12, 0.0, out)
    return out / np.pi


def delta_Vv_depth(
    E: ArrayOrScalar,
    Ef: float,
    AV_plus: float,
    AV_minus: float,
    BV: float,
    EV_plus: float,
    EV_minus: float,
    ALPHA: float,
) -> ArrayOrScalar:
    r"""Dispersive correction ΔV_V^L(E) to the volume depth, in MeV.

    The Brown-Rho part (with the sub-Fermi suppression of Eq. 16) is
    dispersed numerically with the once-subtracted PV integral; the
    Mahaux-Sartor :math:`\alpha\sqrt{E}` tail of Eq. 15 contributes its
    ECIS-06 closed-form partner :func:`_dlpe`.
    """
    br = dispersion_correction(
        lambda Ep: Wv_depth(Ep, Ef, AV_plus, AV_minus, BV, EV_plus, EV_minus, 0.0),
        E,
        Ef,
    )
    tail = ALPHA * _dlpe(Ef + EV_plus, E, Ef)
    out = np.asarray(br, dtype=float) + tail
    return out.item() if np.ndim(E) == 0 else out


# -- nonlocal kernels ----------------------------------------------------------


def perey_buck_kernel(rgrid: FloatArray, ls: FloatArray, beta: float) -> FloatArray:
    r"""Reduced Perey-Buck partial-wave kernel g_l(r, r') (Eq. 7 with U = 1).

    .. math::

       g_l(r, r') = \frac{4 r r'}{\sqrt{\pi}\beta^3}
           e^{-(r^2 + r'^2)/\beta^2} i_l\!\left(\frac{2rr'}{\beta^2}\right)

    evaluated stably via the exponentially scaled Bessel function
    ``ive``: :math:`e^{-(r^2+r'^2)/\beta^2} i_l(z) = e^{-(r-r')^2/\beta^2}
    \sqrt{\pi/2z}\,\mathrm{ive}(l+\tfrac12, z)`, ``z = 2rr'/β²``.

    Args:
        rgrid: Radial grid in fm, strictly positive.
        ls: Angular momenta, shape ``(N_b,)``.
        beta: Nonlocality range in fm.

    Returns:
        Kernel array of shape ``(N_b, N, N)``.
    """
    r = np.asarray(rgrid, dtype=float)
    ls = np.atleast_1d(np.asarray(ls))
    z = 2.0 * r[:, None] * r[None, :] / beta**2
    gauss = np.exp(-((r[:, None] - r[None, :]) ** 2) / beta**2)
    pref = 4.0 * r[:, None] * r[None, :] / (np.sqrt(np.pi) * beta**3)
    scaled_il = np.sqrt(np.pi / (2.0 * z))[None, ...] * ive(
        ls[:, None, None] + 0.5, z[None, ...]
    )
    return pref[None, ...] * gauss[None, ...] * scaled_il


def _midpoint_grid(rgrid: FloatArray) -> FloatArray:
    """Return the (N, N) grid of Perey-Buck midpoints (r + r')/2."""
    r = np.asarray(rgrid, dtype=float)
    return 0.5 * (r[:, None] + r[None, :])


# -- workspace-facing form factors ---------------------------------------------


def central_nonlocal(
    rgrid: FloatArray,
    ls: FloatArray,
    VV: float,
    VS: complex,
    R: float,
    a: float,
    beta: float,
) -> ComplexArray:
    r"""Nonlocal central kernel: volume + (dispersive, absorptive) surface.

    Args:
        rgrid: Radial grid in fm.
        ls: Angular momenta, shape ``(N_b,)``.
        VV: Real volume depth ``V_V^NL`` in MeV.
        VS: Complex surface depth ``V_S^NL + ΔV_S^NL(E) + i W_S^NL(E)``.
        R: Radius in fm.
        a: Diffuseness in fm.
        beta: Nonlocality range in fm.

    Returns:
        ``(N_b, N, N)`` complex kernel for
        ``IntegralWorkspace.central(..., l_dependent=True)``.
    """
    mid = _midpoint_grid(rgrid)
    U = np.asarray(
        VV * woods_saxon_safe(mid, R, a)
        + VS * (-4.0 * a) * woods_saxon_prime_safe(mid, R, a),
        dtype=complex,
    )
    return np.asarray(
        U[None, ...] * perey_buck_kernel(rgrid, ls, beta), dtype=np.complex128
    )


def spin_orbit_nonlocal(
    rgrid: FloatArray,
    ls: FloatArray,
    VSO: complex,
    R: float,
    a: float,
    beta: float,
) -> ComplexArray:
    r"""Unscaled nonlocal spin-orbit kernel (the workspace applies ⟨l·σ⟩).

    The paper's spin-orbit term is ``-2 U_so (λ_π²/r̃)(df/dr̃) l·s`` with
    ``2⟨l·s⟩ = ⟨l·σ⟩ = {l, -(l+1)}``, so the unscaled form factor passed to
    ``IntegralWorkspace.spin_orbit`` is ``-U_so λ_π² (1/r̃)(df/dr̃)``.

    Args:
        rgrid: Radial grid in fm.
        ls: Angular momenta, shape ``(N_b,)``.
        VSO: Complex spin-orbit depth
            ``V_so^NL + ΔV_so^NL(E) + i W_so^NL(E)``.
        R: Radius in fm.
        a: Diffuseness in fm.
        beta: Nonlocality range in fm.

    Returns:
        ``(N_b, N, N)`` complex kernel for
        ``IntegralWorkspace.spin_orbit(..., l_dependent=True)``.
    """
    mid = _midpoint_grid(rgrid)
    U = np.asarray(-VSO * LAMBDA_PI2 * thomas_safe(mid, R, a), dtype=complex)
    return np.asarray(
        U[None, ...] * perey_buck_kernel(rgrid, ls, beta), dtype=np.complex128
    )


def central_local(
    rgrid: FloatArray,
    UV: complex,
    R: float,
    a: float,
) -> ComplexArray:
    r"""Local volume term ``(ΔV_V^L(E) + i W_V^L(E)) f(r, R, a)`` (Eq. 11).

    Returns:
        ``(N,)`` complex array for ``IntegralWorkspace.central``.
    """
    return UV * np.asarray(
        woods_saxon_safe(np.asarray(rgrid, dtype=float), R, a), dtype=complex
    )


def calculate_params(
    projectile: tuple[int, int],
    target: tuple[int, int],
    Ecm: float,
    Ef: float,
    *params: float,
) -> tuple[
    tuple[float, complex, float, float, float],
    tuple[complex, float, float],
    tuple[complex, float, float, float],
]:
    """Assemble the MBRA term parameters at one CM energy.

    Args:
        projectile: ``(A, Z)`` of the projectile — must be a neutron
            ``(1, 0)``; the model is neutron-only.
        target: ``(A, Z)`` of the target.
        Ecm: Center-of-mass energy in MeV.
        Ef: Neutron Fermi energy in MeV (``Reaction.Ef``).
        *params: Global parameters in :func:`get_param_names` order;
            the published values are used when omitted.

    Returns:
        ``(nonlocal_central_params, local_central_params, spin_orbit_params)``
        ready for :func:`central_nonlocal`, :func:`central_local` and
        :func:`spin_orbit_nonlocal` respectively.
    """
    if tuple(projectile) != (1, 0):
        raise ValueError(
            f"The MBRA potential is neutron-only; got projectile {projectile}"
        )
    A = target[0]
    c = resolve_coefficients(A, *params)
    E = float(Ecm)

    VS = complex(
        c.VS + delta_Vs_depth(E, Ef, c.AS_plus, c.AS_minus, c.BS, c.CS),
        Ws_depth(E, Ef, c.AS_plus, c.AS_minus, c.BS, c.CS),
    )
    UV = complex(
        delta_Vv_depth(
            E, Ef, c.AV_plus, c.AV_minus, c.BV, c.EV_plus, c.EV_minus, c.ALPHA
        ),
        Wv_depth(E, Ef, c.AV_plus, c.AV_minus, c.BV, c.EV_plus, c.EV_minus, c.ALPHA),
    )
    VSO = complex(
        c.VSO + delta_Vso_depth(E, Ef, c.ASO, c.BSO, c.CSO, c.DSO),
        Wso_depth(E, Ef, c.ASO, c.BSO, c.CSO, c.DSO),
    )

    nonlocal_central_params = (c.VV, VS, c.R, c.a, c.beta)
    local_central_params = (UV, c.R, c.a)
    spin_orbit_params = (VSO, c.R, c.a, c.beta)
    return nonlocal_central_params, local_central_params, spin_orbit_params
