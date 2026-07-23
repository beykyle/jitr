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
:func:`assemble_terms` builds all three workspace-ready term arrays for a
whole energy grid in one call, sharing a single kernel build.

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
``E_F = -[S_n(Z, N) + S_n(Z, N+1)]/2`` — the standard average over the
last-occupied and first-unoccupied levels, available as ``Reaction.Ef`` in
jitr. Note the paper prints a *difference* of separation energies here;
that is a typo for the sum-average, which is what reproduces the Fig. 3
dip positions (E_F ≈ −12.0/−9.2/−5.7 MeV for ⁴⁰Ca/⁸⁹Y/²⁰⁸Pb).

The depth and dispersion functions take the paper's energy variable ``E``
directly; :func:`calculate_params` takes the laboratory-frame neutron
energy ``Elab``, consistent with the other global potentials in this
package (kduq, wlh, chuq). The paper does not state the frame of ``E``
explicitly (it is on the author-questions list in
``examples/notebooks/mbra_av_minus.ipynb``).

.. _B. Morillon, G. Blanchon, P. Romain and H. F. Arellano,
   arXiv:2403.05843 (2024): https://arxiv.org/abs/2403.05843
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._types import ArrayOrScalar, ComplexArray, FloatArray
from ..utils.constants import WAVENUMBER_PION
from .dispersion import (
    sqrt_tail_dispersive_partner,
    subtracted_dispersion_correction,
)
from .potential_forms import (
    perey_buck_kernel,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)

#: (ħ/m_π c)² in fm² — exactly 2.0, the conventional spin-orbit scale used
#: repo-wide (``WAVENUMBER_PION = sqrt(1/2)`` fm⁻¹, Thompson & Nunes).
#: The physical charged-pion value is ≈2.00 fm²; the historic
#: isospin-averaged value is ≈2.04 fm². The paper writes (ħ/m_π c)² only
#: symbolically, so which value the authors' code used is on the
#: author-questions list (a ~2% spin-orbit scale if they used 2.04).
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

_AV_MINUS_0_IDX = PARAM_NAMES.index("av_minus_0")

#: The literal reading of Table I's "−8.400A" sub-Fermi volume depth
#: (``av_minus_0 = 0``, ``av_minus_A = −8.400``). It reproduces neither the
#: scale nor the mass-independence of the paper's own Fig. 3(b) (see the
#: ``DEFAULT_PARAMS`` note and ``examples/notebooks/mbra_av_minus.ipynb``);
#: kept for reference and comparison.
TABLE_I_LITERAL_PARAMS: tuple[float, ...] = (
    DEFAULT_PARAMS[:_AV_MINUS_0_IDX]
    + (0.0, -8.400)
    + DEFAULT_PARAMS[_AV_MINUS_0_IDX + 2 :]
)


def get_param_names() -> list[str]:
    """Return the MBRA parameter names in ``calculate_params`` order."""
    return list(PARAM_NAMES)


def get_default_params() -> tuple[float, ...]:
    """Return the recommended global parameter vector.

    Tables I & II of the paper, except ``av_minus``: the printed "−8.400A"
    is replaced by the constant −48.40 MeV that reproduces the paper's own
    Fig. 3(b) (see the note on :data:`DEFAULT_PARAMS`;
    :data:`TABLE_I_LITERAL_PARAMS` holds the literal reading).
    """
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

    Caveat: the causal once-subtracted dispersion (Eq. 9) of the symmetric
    W_so is *odd* in ``E - E_F``, so this even form flips its sign for
    ``E < E_F`` (e.g. −1.52 MeV causal vs +1.52 MeV here at ``E − E_F ≈
    −24`` MeV for ²⁰⁸Pb). Scattering energies ``E > 0 > E_F`` are
    unaffected; DOM-style sub-Fermi use (bound states, occupations) should
    be aware the published form violates the dispersion relation there.
    """
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    x = np.abs(E_arr - Ef)
    out = ASO * CSO * x / (x**2 + CSO**2) - BSO * DSO * x / (x**2 + DSO**2)
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
    return subtracted_dispersion_correction(
        lambda Ep: Ws_depth(Ep, Ef, AS_plus, AS_minus, BS, CS), E, Ef
    )


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
    ECIS-06 closed-form partner
    :func:`jitr.optical_potentials.dispersion.sqrt_tail_dispersive_partner`.

    Note: outside ``0 < E < 100`` MeV this ΔV_V^L deviates from the paper's
    Fig. 3(b) by up to a few MeV (module−figure for ⁸⁹Y: +1.9 MeV at
    200 MeV, +2.5 MeV at 245 MeV with a sign flip; −1.6 to −3.2 MeV for
    ``E ≤ −100`` MeV, where the figure instead matches a Brown-Rho-only
    dispersion). The integral here is numerically exact for Eqs. 14-16, so
    the paper evidently used a different tail prescription — pending author
    clarification (see ``examples/notebooks/mbra_av_minus.ipynb``).
    """
    br = subtracted_dispersion_correction(
        lambda Ep: Wv_depth(Ep, Ef, AV_plus, AV_minus, BV, EV_plus, EV_minus, 0.0),
        E,
        Ef,
    )
    tail = ALPHA * sqrt_tail_dispersive_partner(Ef + EV_plus, E, Ef)
    out = np.asarray(br, dtype=float) + tail
    return out.item() if np.ndim(E) == 0 else out


# -- nonlocal kernels ----------------------------------------------------------

# The Eq. 7 partial-wave kernel (with U = 1) is the shared
# potential_forms.perey_buck_kernel, re-exported here as mbra.perey_buck_kernel.


def _midpoint_grid(rgrid: FloatArray) -> FloatArray:
    """Return the (N, N) grid of Perey-Buck midpoints (r + r')/2."""
    r = np.asarray(rgrid, dtype=float)
    return 0.5 * (r[:, None] + r[None, :])


# -- workspace-facing form factors ---------------------------------------------


def central_nonlocal(
    rgrid: FloatArray,
    ls: FloatArray,
    VV: float,
    VS: complex | ComplexArray,
    R: float,
    a: float,
    beta: float,
    *,
    kernel: FloatArray | None = None,
) -> ComplexArray:
    r"""Nonlocal central kernel: volume + (dispersive, absorptive) surface.

    Args:
        rgrid: Radial grid in fm.
        ls: Angular momenta, shape ``(N_b,)``.
        VV: Real volume depth ``V_V^NL`` in MeV (energy-independent).
        VS: Complex surface depth ``V_S^NL + ΔV_S^NL(E) + i W_S^NL(E)``:
            a scalar, or an ``(N_E,)`` array over an energy grid.
        R: Radius in fm.
        a: Diffuseness in fm.
        beta: Nonlocality range in fm (unused for the kernel when
            ``kernel`` is provided).
        kernel: Optional precomputed ``perey_buck_kernel(rgrid, ls, beta)``
            of shape ``(N_b, N, N)``, so several terms can share one build.

    Returns:
        Complex kernel for ``IntegralWorkspace.central``: ``(N_b, N, N)``
        for scalar ``VS`` (pass ``l_dependent=True``), or
        ``(N_b, N_E, N, N)`` for array ``VS`` (pass ``l_dependent=True,
        energy_dependent=True``).
    """
    mid = _midpoint_grid(rgrid)
    if kernel is None:
        kernel = perey_buck_kernel(rgrid, ls, beta)
    VS_arr = np.asarray(VS)
    if VS_arr.ndim == 0:
        U = np.asarray(
            VV * woods_saxon_safe(mid, R, a)
            + VS * (-4.0 * a) * woods_saxon_prime_safe(mid, R, a),
            dtype=complex,
        )
        return np.asarray(U[None, ...] * kernel, dtype=np.complex128)
    f = woods_saxon_safe(mid, R, a)
    d = -4.0 * a * woods_saxon_prime_safe(mid, R, a)
    U = VV * f[None, ...] + VS_arr[:, None, None] * d[None, ...]  # (N_E, N, N)
    return np.asarray(kernel[:, None, ...] * U[None, ...], dtype=np.complex128)


def spin_orbit_nonlocal(
    rgrid: FloatArray,
    ls: FloatArray,
    VSO: complex | ComplexArray,
    R: float,
    a: float,
    beta: float,
    *,
    kernel: FloatArray | None = None,
) -> ComplexArray:
    r"""Unscaled nonlocal spin-orbit kernel (the workspace applies ⟨l·σ⟩).

    The paper's spin-orbit term is ``-2 U_so (λ_π²/r̃)(df/dr̃) l·s`` with
    ``2⟨l·s⟩ = ⟨l·σ⟩ = {l, -(l+1)}``, so the unscaled form factor passed to
    ``IntegralWorkspace.spin_orbit`` is ``-U_so λ_π² (1/r̃)(df/dr̃)``.

    Args:
        rgrid: Radial grid in fm.
        ls: Angular momenta, shape ``(N_b,)``.
        VSO: Complex spin-orbit depth
            ``V_so^NL + ΔV_so^NL(E) + i W_so^NL(E)``: a scalar, or an
            ``(N_E,)`` array over an energy grid.
        R: Radius in fm.
        a: Diffuseness in fm.
        beta: Nonlocality range in fm (unused for the kernel when
            ``kernel`` is provided).
        kernel: Optional precomputed ``perey_buck_kernel(rgrid, ls, beta)``
            of shape ``(N_b, N, N)``, so several terms can share one build.

    Returns:
        Complex kernel for ``IntegralWorkspace.spin_orbit``: ``(N_b, N, N)``
        for scalar ``VSO`` (pass ``l_dependent=True``), or
        ``(N_b, N_E, N, N)`` for array ``VSO`` (pass ``l_dependent=True,
        energy_dependent=True``).
    """
    mid = _midpoint_grid(rgrid)
    if kernel is None:
        kernel = perey_buck_kernel(rgrid, ls, beta)
    VSO_arr = np.asarray(VSO)
    if VSO_arr.ndim == 0:
        U = np.asarray(-VSO * LAMBDA_PI2 * thomas_safe(mid, R, a), dtype=complex)
        return np.asarray(U[None, ...] * kernel, dtype=np.complex128)
    t = thomas_safe(mid, R, a)
    U = (-LAMBDA_PI2) * VSO_arr[:, None, None] * t[None, ...]  # (N_E, N, N)
    return np.asarray(kernel[:, None, ...] * U[None, ...], dtype=np.complex128)


def central_local(
    rgrid: FloatArray,
    UV: complex | ComplexArray,
    R: float,
    a: float,
) -> ComplexArray:
    r"""Local volume term ``(ΔV_V^L(E) + i W_V^L(E)) f(r, R, a)`` (Eq. 11).

    Args:
        rgrid: Radial grid in fm.
        UV: Complex volume depth: a scalar, or an ``(N_E,)`` array over an
            energy grid.
        R: Radius in fm.
        a: Diffuseness in fm.

    Returns:
        Complex array for ``IntegralWorkspace.central``: ``(N,)`` for scalar
        ``UV``, or ``(N_E, N)`` for array ``UV`` (pass
        ``energy_dependent=True``).
    """
    UV_arr = np.asarray(UV)
    if UV_arr.ndim == 0:
        return UV * np.asarray(
            woods_saxon_safe(np.asarray(rgrid, dtype=float), R, a), dtype=complex
        )
    f = np.asarray(
        woods_saxon_safe(np.asarray(rgrid, dtype=float), R, a), dtype=complex
    )
    return np.asarray(UV_arr[:, None] * f[None, :], dtype=np.complex128)


def calculate_params(
    projectile: tuple[int, int],
    target: tuple[int, int],
    Elab: ArrayOrScalar,
    Ef: float,
    *params: float,
) -> tuple[
    tuple[float, complex | ComplexArray, float, float, float],
    tuple[complex | ComplexArray, float, float],
    tuple[complex | ComplexArray, float, float, float],
]:
    """Assemble the MBRA term parameters at one or many lab energies.

    Args:
        projectile: ``(A, Z)`` of the projectile — must be a neutron
            ``(1, 0)``; the model is neutron-only.
        target: ``(A, Z)`` of the target.
        Elab: Laboratory-frame incident neutron energy in MeV (the paper's
            energy variable in Eqs. 13-17), matching the kduq/wlh/chuq
            convention. A scalar, or an ``(N_E,)`` energy grid — the grid
            amortizes each dispersion quadrature over all energies at once.
        Ef: Neutron Fermi energy in MeV (``Reaction.Ef``).
        *params: Global parameters in :func:`get_param_names` order;
            :func:`get_default_params` is used when omitted.

    Returns:
        ``(nonlocal_central_params, local_central_params, spin_orbit_params)``
        ready for :func:`central_nonlocal`, :func:`central_local` and
        :func:`spin_orbit_nonlocal` respectively. The energy-dependent
        depths ``VS``, ``UV`` and ``VSO`` are Python complex scalars for
        scalar ``Elab`` and ``(N_E,)`` complex arrays for an energy grid;
        ``VV``, ``R``, ``a`` and ``beta`` are always scalars.
    """
    if tuple(projectile) != (1, 0):
        raise ValueError(
            f"The MBRA potential is neutron-only; got projectile {projectile}"
        )
    A = target[0]
    c = resolve_coefficients(A, *params)
    scalar = np.ndim(Elab) == 0
    E = float(Elab) if scalar else np.asarray(Elab, dtype=float)

    vs_re = c.VS + delta_Vs_depth(E, Ef, c.AS_plus, c.AS_minus, c.BS, c.CS)
    vs_im = Ws_depth(E, Ef, c.AS_plus, c.AS_minus, c.BS, c.CS)
    uv_re = delta_Vv_depth(
        E, Ef, c.AV_plus, c.AV_minus, c.BV, c.EV_plus, c.EV_minus, c.ALPHA
    )
    uv_im = Wv_depth(E, Ef, c.AV_plus, c.AV_minus, c.BV, c.EV_plus, c.EV_minus, c.ALPHA)
    vso_re = c.VSO + delta_Vso_depth(E, Ef, c.ASO, c.BSO, c.CSO, c.DSO)
    vso_im = Wso_depth(E, Ef, c.ASO, c.BSO, c.CSO, c.DSO)

    if scalar:
        VS = complex(vs_re, vs_im)
        UV = complex(uv_re, uv_im)
        VSO = complex(vso_re, vso_im)
    else:
        VS = vs_re + 1j * vs_im
        UV = uv_re + 1j * uv_im
        VSO = vso_re + 1j * vso_im

    nonlocal_central_params = (c.VV, VS, c.R, c.a, c.beta)
    local_central_params = (UV, c.R, c.a)
    spin_orbit_params = (VSO, c.R, c.a, c.beta)
    return nonlocal_central_params, local_central_params, spin_orbit_params


def assemble_terms(
    rgrid: FloatArray,
    ls: FloatArray,
    projectile: tuple[int, int],
    target: tuple[int, int],
    Elab: ArrayOrScalar,
    Ef: float,
    *params: float,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Assemble the three MBRA terms on a grid, stacked over an energy grid.

    Builds the Perey-Buck kernel exactly once and shares it between the two
    nonlocal terms, and evaluates all energy-dependent depths in one
    vectorized pass.

    Args:
        rgrid: Radial grid in fm, shape ``(N,)``
            (``IntegralWorkspace.radial_grid()``).
        ls: Angular momenta, shape ``(N_b,)``.
        projectile: ``(A, Z)`` of the projectile — must be a neutron
            ``(1, 0)``.
        target: ``(A, Z)`` of the target.
        Elab: Laboratory-frame incident neutron energies in MeV; scalars
            are treated as a length-1 grid.
        Ef: Neutron Fermi energy in MeV (``Reaction.Ef``).
        *params: Global parameters in :func:`get_param_names` order;
            :func:`get_default_params` is used when omitted.

    Returns:
        ``(K_nl, W_loc, K_so)`` of shapes ``(N_b, N_E, N, N)``,
        ``(N_E, N)`` and ``(N_b, N_E, N, N)``, ready for::

            ws.central(K_nl, l_dependent=True, energy_dependent=True)
            + ws.central(W_loc, energy_dependent=True)
            + ws.spin_orbit(K_so, l_dependent=True, energy_dependent=True)
    """
    Elab = np.atleast_1d(np.asarray(Elab, dtype=float))
    (VV, VS, R, a, beta), (UV, _, _), (VSO, _, _, _) = calculate_params(
        projectile, target, Elab, Ef, *params
    )
    kernel = perey_buck_kernel(rgrid, ls, beta)
    K_nl = central_nonlocal(rgrid, ls, VV, VS, R, a, beta, kernel=kernel)
    W_loc = central_local(rgrid, UV, R, a)
    K_so = spin_orbit_nonlocal(rgrid, ls, VSO, R, a, beta, kernel=kernel)
    return K_nl, W_loc, K_so
