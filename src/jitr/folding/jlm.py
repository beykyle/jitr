"""JLM and JLMB local-density optical-potential helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ..utils import poly

FloatArray: TypeAlias = NDArray[np.float64]
ScalarOrArray: TypeAlias = float | FloatArray
PolynomialValue: TypeAlias = float | FloatArray
Projectile: TypeAlias = tuple[int, int]
Target: TypeAlias = tuple[int, int]
JLMRealMode: TypeAlias = Literal["linearized_delta_c", "shifted"]

# ---------------------------------------------------------------------------
# Table I — V_0(ρ, E) = Σ a_ij ρ^i E^(j-1)            [Eq. 25]
# Isoscalar real component (MeV)
# ---------------------------------------------------------------------------
A_V0 = np.array(
    [
        # j=1        j=2         j=3
        [-974.0, 11.26, -0.0425],  # i=1
        [7097.0, -125.7, 0.5853],  # i=2
        [-19530.0, 418.0, -2.054],  # i=3
    ]
)

# ---------------------------------------------------------------------------
# Table II — Re N(ρ, E) = Σ b_ij ρ^i E^(j-1)          [Eq. 28]
# Real isovector kernel; V_1 = (m̃/m) · Re N           [Eq. 11]
# ---------------------------------------------------------------------------
B_N_RE = np.array(
    [
        [360.1, -5.224, 0.02051],  # i=1
        [-2691.0, 51.30, -0.2470],  # i=2
        [7733.0, -171.7, 0.8846],  # i=3
    ]
)

# ---------------------------------------------------------------------------
# Table III — m̃(ρ, E)/m = 1 − Σ c_ij ρ^i E^(j-1)     [Eq. 29]
# k-mass ratio. Multiplies V_1 (Bauge 1998 Eq. 6); in JLMB it also multiplies
# the whole imaginary potential, W_0 + αW_1 (end of Bauge 1998 Sec. II.A).
# ---------------------------------------------------------------------------
C_KMASS = np.array(
    [
        [4.557, -5.291e-3, 6.108e-6],  # i=1
        [-2.051, -0.4906, 1.812e-3],  # i=2
        [-65.09, 3.095, -0.0119],  # i=3
    ]
)

# ---------------------------------------------------------------------------
# Table IV — W̄_0(ρ, E) = [1 + D/(E−ε_F)²]⁻¹ Σ d_ij ρ^i E^(j-1)    [Eq. 30]
# Imaginary isoscalar kernel, D = 600 MeV². The fit is W_0 itself, which
# already contains the E-mass factor: W_0 = (m/m̄) Im M_0 (Bauge 1998 Eq. 7).
# ---------------------------------------------------------------------------
D_W0 = np.array(
    [
        # j=1          j=2         j=3        j=4
        [-1483.0, 37.18, -0.3549, 1.119e-3],  # i=1
        [29880.0, -931.8, 9.591, -0.03160],  # i=2
        [-212800.0, 7209.0, -77.52, 0.2611],  # i=3
        [512500.0, -17960.0, 198.0, -0.6753],  # i=4
    ]
)
D_DAMPING = 600.0  # MeV²

# ---------------------------------------------------------------------------
# Table V — Im N_1(ρ, E) = [1 + F/(E−ε_F)]⁻¹ Σ f_ij ρ^i E^(j-1)   [Eq. 31]
# Imaginary isovector kernel, F = 1 MeV. The fit is Im N; W_1 = (m/m̄) Im N
# (Bauge 1998 Eq. 8), with the E-mass m̄/m = (m*/m)/(m̃/m) (Eqs. 19-20).
# ---------------------------------------------------------------------------
F_N_IM = np.array(
    [
        [546.1, -11.20, 0.1065, -3.541e-4],  # i=1
        [-8471.0, 230.0, -2.439, 8.544e-3],  # i=2
        [51720.0, -1520.0, 17.17, -0.06211],  # i=3
        [-114000.0, 3543.0, -41.69, 0.1537],  # i=4
    ]
)
F_DAMPING = 1.0  # MeV
# MeV; far below typical projectile energies, but avoids zero division.
DAMPING_EPS = 1e-12

# ---------------------------------------------------------------------------
# Eq. 27 — Fermi energy ε_F(ρ) = ρ·(−510.8 + 3222·ρ − 6250·ρ²)  [MeV]
# ---------------------------------------------------------------------------
EPS_F_coeffs = np.array([-510.8, 3222.0, -6250.0])  # multiplied by ρ¹, ρ², ρ³
EPS_F_LOW_OFFSET = -22.0
EPS_F_LOW_coeffs = np.array([-298.52, 3760.23, -12435.82])
EPS_F_BLEND_ENERGY = 9.0
EPS_F_BLEND_WIDTH = 2.0

# ---------------------------------------------------------------------------
# Bauge, Delaroche and Girod, PRC 58, 1118 (1998): refitted imaginary kernels
# (Tables I and II) and the smooth Fermi energy of Eqs. (21)-(22), with the
# values of the TALYS implementation (momjlmecis.f), which differ from the
# printed paper in three places:
#   - damping D = 126.25 MeV², not the 625 MeV² of Table I (TALYS: "typo in
#     PRC 58 1118 (1998)");
#   - the ρ³ coefficient of the low-energy Fermi energy, Eq. (17), is 12435.82,
#     not 12345;
#   - the blend of Eq. (22) is centred at E_0 = 9 MeV, not 10 MeV.
# The paper also states τ = +1 for protons in Eq. (4). TALYS and jitr use +1
# for neutrons: Re N > 0, so this is the sign that binds protons more strongly
# in N > Z nuclei, as the Lane model requires.
# ---------------------------------------------------------------------------
TALYS_D_W0 = np.array(
    [
        [-659.86, 10.768, -0.078863, 1.8755e-4],
        [11437.0, -290.76, 2.4430, -6.2028e-3],
        [-74505.0, 2206.8, -19.926, 5.1754e-2],
        [176090.0, -5457.9, 51.127, -0.13386],
    ]
)
TALYS_D_DAMPING = 126.25

TALYS_F_N_IM = np.array(
    [
        [459.59, -6.4399, 4.0403e-2, -9.0086e-5],
        [-7692.9, 146.39, -1.0244, 2.3367e-3],
        [55250.0, -1112.1, 7.9667, -1.8008e-2],
        [-143730.0, 3038.2, -22.202, 5.0258e-2],
    ]
)


@dataclass(frozen=True)
class JLMParameterization:
    """Bundled coefficient and low-energy choices for JLM/JLMB helpers.

    Attributes:
        name: Human-readable identifier.
        coeffs_A: Real isoscalar coefficient table.
        coeffs_B: Real isovector coefficient table.
        coeffs_C: Momentum-dependent mass coefficient table.
        coeffs_D: Imaginary isoscalar coefficient table.
        coeffs_F: Imaginary isovector coefficient table.
        coeffs_E_F: High-energy Fermi-energy coefficients.
        low_energy_E_F_offset: Constant term for the low-energy Fermi branch.
        low_energy_E_F_coeffs: Density coefficients for the low-energy branch.
        E_F_blend_energy: Logistic blend center in MeV.
        E_F_blend_width: Logistic blend width in MeV.
        D_damping: Imaginary isoscalar damping parameter in MeV².
        F_damping: Imaginary isovector damping parameter in MeV.
        local_jlm_real_mode: Real-part Coulomb treatment for
            :func:`potential_JLM`.
    """

    name: str
    coeffs_A: FloatArray
    coeffs_B: FloatArray
    coeffs_C: FloatArray
    coeffs_D: FloatArray
    coeffs_F: FloatArray
    coeffs_E_F: FloatArray
    low_energy_E_F_offset: float | None = None
    low_energy_E_F_coeffs: FloatArray | None = None
    E_F_blend_energy: float = EPS_F_BLEND_ENERGY
    E_F_blend_width: float = EPS_F_BLEND_WIDTH
    D_damping: float = D_DAMPING
    F_damping: float = F_DAMPING
    local_jlm_real_mode: JLMRealMode = "linearized_delta_c"


JLM1977 = JLMParameterization(
    name="jlm1977",
    coeffs_A=A_V0,
    coeffs_B=B_N_RE,
    coeffs_C=C_KMASS,
    coeffs_D=D_W0,
    coeffs_F=F_N_IM,
    coeffs_E_F=EPS_F_coeffs,
    D_damping=D_DAMPING,
    F_damping=F_DAMPING,
    local_jlm_real_mode="linearized_delta_c",
)

BAUGE1998 = JLMParameterization(
    name="bauge1998",
    coeffs_A=A_V0,
    coeffs_B=B_N_RE,
    coeffs_C=C_KMASS,
    coeffs_D=TALYS_D_W0,
    coeffs_F=TALYS_F_N_IM,
    coeffs_E_F=EPS_F_coeffs,
    low_energy_E_F_offset=EPS_F_LOW_OFFSET,
    low_energy_E_F_coeffs=EPS_F_LOW_coeffs,
    E_F_blend_energy=EPS_F_BLEND_ENERGY,
    E_F_blend_width=EPS_F_BLEND_WIDTH,
    D_damping=TALYS_D_DAMPING,
    F_damping=F_DAMPING,
    local_jlm_real_mode="shifted",
)

# Previous names.
ORIGINAL_PARAMETERIZATION = JLM1977
TALYS_PARAMETERIZATION = BAUGE1998

_PARAMETERIZATION_ALIASES = {
    "jlm1977": JLM1977,
    "original": JLM1977,
    "bauge1998": BAUGE1998,
    "jlmb1998": BAUGE1998,
    "talys": BAUGE1998,
}


def resolve_parameterization(
    parameterization: str | JLMParameterization | None,
) -> JLMParameterization | None:
    """Resolve a named or explicit JLM parameterization bundle.

    Names:

    - ``"jlm1977"`` (alias ``"original"``): Jeukenne, Lejeune and Mahaux,
      Phys. Rev. C 16, 80 (1977). Note that below 10 MeV this bundle keeps
      the high-energy Fermi energy and ``D = 600 MeV²``, where the original
      work switched to Lejeune's low-energy fit (``D = 100 MeV²``).
    - ``"bauge1998"`` (aliases ``"jlmb1998"``, ``"talys"``): the refitted
      imaginary kernels and smooth Fermi energy of Bauge, Delaroche and
      Girod, Phys. Rev. C 58, 1118 (1998), as implemented in TALYS. This is
      the JLMB potential.

    ``None`` means no bundle: the explicit ``coeffs_*`` arguments are used.
    """

    if parameterization is None:
        return None
    if isinstance(parameterization, JLMParameterization):
        return parameterization
    normalized = parameterization.strip().lower()
    if normalized in _PARAMETERIZATION_ALIASES:
        return _PARAMETERIZATION_ALIASES[normalized]
    raise ValueError(
        f"parameterization must be one of {sorted(_PARAMETERIZATION_ALIASES)}, "
        "a JLMParameterization instance, or None."
    )


def fermi_energy_MeV(
    rho_fm3: ScalarOrArray,
    coeffs: FloatArray = EPS_F_coeffs,
    projectile_energy_MeV: ScalarOrArray | None = None,
    low_energy_offset: float | None = None,
    low_energy_coeffs: FloatArray | None = None,
    blend_energy: float = EPS_F_BLEND_ENERGY,
    blend_width: float = EPS_F_BLEND_WIDTH,
) -> PolynomialValue:
    """Return the local Fermi energy in MeV.

    Args:
        rho_fm3: Matter density in fm⁻³.
        coeffs: Polynomial coefficients for the high-energy JLM Fermi-energy fit.
        projectile_energy_MeV: Incident energy used for the low-energy branch
            blend. If omitted, only the high-energy branch is evaluated.
        low_energy_offset: Constant term for the low-energy branch.
        low_energy_coeffs: Density coefficients for the low-energy branch.
        blend_energy: Logistic blend center in MeV.
        blend_width: Logistic blend width in MeV.

    Returns:
        Fermi energy evaluated at ``rho_fm3``.
    """

    high_energy_branch = poly.poly1d(rho_fm3, coeffs, start_i=1)
    if (
        projectile_energy_MeV is None
        or low_energy_offset is None
        or low_energy_coeffs is None
    ):
        return high_energy_branch

    low_energy_branch = low_energy_offset + poly.poly1d(
        rho_fm3, low_energy_coeffs, start_i=1
    )
    low_energy_weight = 1.0 / (
        1.0 + np.exp((np.asarray(projectile_energy_MeV) - blend_energy) / blend_width)
    )
    return low_energy_branch * low_energy_weight + high_energy_branch * (
        1.0 - low_energy_weight
    )


def V0(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    coeffs: FloatArray = A_V0,
) -> PolynomialValue:
    """Return the real isoscalar JLM self-energy component.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        coeffs: Polynomial coefficients for the fit.

    Returns:
        Real isoscalar self-energy values in MeV.
    """

    return poly.poly2d(rho_fm3, E_MeV, coeffs, start_i=1, start_j=0)


def W0(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    E_F: ScalarOrArray,
    coeffs: FloatArray = D_W0,
    damping: float = D_DAMPING,
    damping_eps: float = DAMPING_EPS,
) -> PolynomialValue:
    """Return the imaginary isoscalar JLM self-energy component.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        E_F: Local Fermi energy in MeV.
        coeffs: Polynomial coefficients for the fit.
        damping: Imaginary-part damping parameter in MeV².
        damping_eps: Minimum energy denominator scale in MeV.

    Returns:
        Imaginary isoscalar self-energy values in MeV.
    """

    # Eq. 30 uses (E - E_F)^2 in the damping denominator.
    safe_energy_diff_sq = np.maximum((E_MeV - E_F) ** 2, damping_eps**2)
    damping_term = damping / safe_energy_diff_sq
    return poly.poly2d(rho_fm3, E_MeV, coeffs, start_i=1, start_j=0) / (
        1 + damping_term
    )


def m_tilde_over_m(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    coeffs: FloatArray = C_KMASS,
) -> PolynomialValue:
    """Return the JLM momentum-dependent mass ratio.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        coeffs: Polynomial coefficients for the fit.

    Returns:
        Values of ``m̃ / m``.
    """

    return 1.0 - poly.poly2d(rho_fm3, E_MeV, coeffs, start_i=1, start_j=0)


def eff_mass(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    coeffs: FloatArray = A_V0,
) -> PolynomialValue:
    """Return the JLM effective-mass ratio.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        coeffs: Polynomial coefficients for the real isoscalar fit.

    Returns:
        Values of ``m* / m``.
    """

    c, si, sj = poly.poly2d_deriv(coeffs, start_i=1, start_j=0, wrt="y")
    return 1.0 - poly.poly2d(rho_fm3, E_MeV, c, start_i=si, start_j=sj)


def E_mass(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    coeffs_A: FloatArray = A_V0,
    coeffs_C: FloatArray = C_KMASS,
) -> PolynomialValue:
    """Return the JLM energy-dependent mass ratio.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        coeffs_A: Coefficients for the real isoscalar fit.
        coeffs_C: Coefficients for the momentum-dependent mass fit.

    Returns:
        Values of ``m̄ / m`` obtained from ``m* / m = (m̃ / m) (m̄ / m)``.
    """

    return eff_mass(rho_fm3, E_MeV, coeffs=coeffs_A) / m_tilde_over_m(
        rho_fm3, E_MeV, coeffs=coeffs_C
    )


def V1(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    E_F: ScalarOrArray,
    coeffs_B: FloatArray = B_N_RE,
    coeffs_C: FloatArray = C_KMASS,
) -> PolynomialValue:
    """Return the real isovector JLM self-energy component.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        E_F: Local Fermi energy in MeV. Retained for API symmetry with
            :func:`W1`; the analytic form does not depend on it.
        coeffs_B: Polynomial coefficients for the real isovector fit.
        coeffs_C: Coefficients for the momentum-dependent mass fit.

    Returns:
        Real isovector self-energy values in MeV.
    """

    re_n = poly.poly2d(rho_fm3, E_MeV, coeffs_B, start_i=1, start_j=0)
    return m_tilde_over_m(rho_fm3, E_MeV, coeffs=coeffs_C) * re_n


def W1(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    E_F: ScalarOrArray,
    coeffs_F: FloatArray = F_N_IM,
    coeffs_A: FloatArray = A_V0,
    coeffs_C: FloatArray = C_KMASS,
    damping: float = F_DAMPING,
    damping_eps: float = DAMPING_EPS,
) -> PolynomialValue:
    """Return the imaginary isovector JLM self-energy component.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        E_F: Local Fermi energy in MeV.
        coeffs_F: Polynomial coefficients for the imaginary isovector fit.
        coeffs_A: Coefficients for the real isoscalar fit.
        coeffs_C: Coefficients for the momentum-dependent mass fit.
        damping: Imaginary-part damping parameter in MeV.
        damping_eps: Minimum absolute energy denominator scale in MeV.

    Returns:
        Imaginary isovector self-energy values in MeV.
    """

    energy_diff = np.asarray(E_MeV - E_F)
    # Eq. 31 uses a signed (E - E_F) denominator, so preserve sign near zero.
    safe_energy_diff = np.where(
        np.abs(energy_diff) < damping_eps,
        np.copysign(damping_eps, energy_diff),
        energy_diff,
    )
    damping_term = damping / safe_energy_diff
    im_n = poly.poly2d(rho_fm3, E_MeV, coeffs_F, start_i=1, start_j=0) / (
        1 + damping_term
    )
    return im_n / E_mass(rho_fm3, E_MeV, coeffs_A=coeffs_A, coeffs_C=coeffs_C)


def Delta_C(
    rho_fm3: ScalarOrArray,
    E_MeV: ScalarOrArray,
    V_C_MeV: ScalarOrArray,
    coeffs_A: FloatArray = A_V0,
    linear: bool = True,
) -> PolynomialValue:
    """Return the real Coulomb correction for protons.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        V_C_MeV: Coulomb potential in MeV.
        coeffs_A: Polynomial coefficients for the real isoscalar fit.
        linear: Whether to use the preferred linearized form from JLM 1977.

    Returns:
        Coulomb correction values in MeV.
    """

    if linear:
        return (eff_mass(rho_fm3, E_MeV, coeffs=coeffs_A) - 1.0) * V_C_MeV
    return V0(rho_fm3, E_MeV - V_C_MeV, coeffs=coeffs_A) - V0(
        rho_fm3, E_MeV, coeffs=coeffs_A
    )


def potential_JLM(
    rgrid: FloatArray,
    rho_grid: FloatArray,
    projectile: Projectile,
    target: Target,
    E: float,
    V_C: FloatArray | None = None,
    parameterization: str | JLMParameterization | None = None,
    coeffs_A: FloatArray = A_V0,
    coeffs_B: FloatArray = B_N_RE,
    coeffs_C: FloatArray = C_KMASS,
    coeffs_D: FloatArray = D_W0,
    D_damping: float = D_DAMPING,
    coeffs_F: FloatArray = F_N_IM,
    F_damping: float = F_DAMPING,
    coeffs_E_F: FloatArray = EPS_F_coeffs,
    r_out: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the local-density JLM optical potential.

    Args:
        rgrid: Radial grid associated with ``rho_grid``
        rho_grid: Matter-density values sampled on ``rgrid``.
        projectile: Projectile identifier, ``(1, 0)`` for neutrons or
            ``(1, 1)`` for protons.
        target: Target ``(A, Z)`` tuple.
        E: Projectile energy in MeV.
        V_C: Coulomb potential sampled on ``rgrid``. Required for proton
            projectiles and ignored for neutrons.
        parameterization: Named kernel bundle (see
            :func:`resolve_parameterization`), e.g. ``'jlm1977'`` or
            ``'bauge1998'``. If ``None``, the explicit ``coeffs_*`` and
            damping arguments are used.
        coeffs_A: Real isoscalar coefficient table.
        coeffs_B: Real isovector coefficient table.
        coeffs_C: Momentum-dependent mass coefficient table.
        coeffs_D: Imaginary isoscalar coefficient table.
        D_damping: Imaginary isoscalar damping parameter in MeV².
        coeffs_F: Imaginary isovector coefficient table.
        F_damping: Imaginary isovector damping parameter in MeV.
        coeffs_E_F: Fermi-energy coefficient vector.
        r_out: Optional output grid in fm

    Returns:
        Tuple of real and imaginary optical-potential arrays in MeV.

    Raises:
        ValueError: If ``projectile`` is not neutron or proton, or if ``V_C`` is
            missing or malformed for proton calls.
    """

    r_array = np.asarray(rgrid, dtype=float)
    rho_array = np.asarray(rho_grid, dtype=float)
    if r_array.shape != rho_array.shape:
        raise ValueError("rgrid and rho_grid must have the same shape.")
    A, Z = target
    N = A - Z
    alpha = (N - Z) / A
    params = resolve_parameterization(parameterization)
    if params is not None:
        coeffs_A = params.coeffs_A
        coeffs_B = params.coeffs_B
        coeffs_C = params.coeffs_C
        coeffs_D = params.coeffs_D
        coeffs_F = params.coeffs_F
        coeffs_E_F = params.coeffs_E_F
        D_damping = params.D_damping
        F_damping = params.F_damping
        local_jlm_real_mode = params.local_jlm_real_mode
        low_energy_offset = params.low_energy_E_F_offset
        low_energy_coeffs = params.low_energy_E_F_coeffs
        blend_energy = params.E_F_blend_energy
        blend_width = params.E_F_blend_width
    else:
        local_jlm_real_mode = "linearized_delta_c"
        low_energy_offset = None
        low_energy_coeffs = None
        blend_energy = EPS_F_BLEND_ENERGY
        blend_width = EPS_F_BLEND_WIDTH

    E_F = fermi_energy_MeV(
        rho_fm3=rho_array,
        coeffs=coeffs_E_F,
        projectile_energy_MeV=E,
        low_energy_offset=low_energy_offset,
        low_energy_coeffs=low_energy_coeffs,
        blend_energy=blend_energy,
        blend_width=blend_width,
    )

    if projectile == (1, 1):
        if V_C is None:
            raise ValueError("V_C is required for proton projectiles.")
        V_c_grid = np.asarray(V_C, dtype=float)
        if V_c_grid.shape != rho_array.shape:
            raise ValueError("V_C must have the same shape as rho_grid.")
        if local_jlm_real_mode == "linearized_delta_c":
            DelC = Delta_C(
                rho_fm3=rho_array,
                E_MeV=E,
                V_C_MeV=V_c_grid,
                coeffs_A=coeffs_A,
                linear=True,
            )
            V0_energy = np.full_like(rho_array, float(E))
        else:
            DelC = np.zeros_like(rho_array)
            V0_energy = E - V_c_grid
        E_eff = E - V_c_grid
        sign = -1.0
    elif projectile == (1, 0):
        if V_C is not None:
            raise ValueError("V_C is only supported for proton projectiles.")
        DelC = np.zeros_like(rho_array)
        V0_energy = np.full_like(rho_array, float(E))
        E_eff = np.full_like(rho_array, float(E))
        sign = +1.0
    else:
        raise ValueError(
            f"Projectile must be neutron (1,0) or proton (1,1), received {projectile}"
        )

    V0_grid = V0(rho_fm3=rho_array, E_MeV=V0_energy, coeffs=coeffs_A)
    W0_grid = W0(
        rho_fm3=rho_array,
        E_MeV=E_eff,
        E_F=E_F,
        coeffs=coeffs_D,
        damping=D_damping,
    )
    V1_grid = V1(
        rho_fm3=rho_array,
        E_MeV=E_eff,
        coeffs_B=coeffs_B,
        coeffs_C=coeffs_C,
        E_F=E_F,
    )
    W1_grid = W1(
        rho_fm3=rho_array,
        E_MeV=E_eff,
        E_F=E_F,
        coeffs_A=coeffs_A,
        coeffs_C=coeffs_C,
        coeffs_F=coeffs_F,
        damping=F_damping,
    )
    v0_out = np.asarray(V0_grid + DelC + sign * alpha * V1_grid, dtype=float)
    w0_out = np.asarray(W0_grid + sign * alpha * W1_grid, dtype=float)
    if r_out is not None:
        v0_out = np.interp(r_out, rgrid, v0_out)
        w0_out = np.interp(r_out, rgrid, w0_out)
    return v0_out, w0_out


def lambda_v0(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB real-isoscalar normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Real isoscalar normalization values.
    """

    log_E = np.log(1000.0 * E_MeV)  # paper has ln(1000 E), E in MeV
    return 0.951 + 0.0008 * log_E + 0.00018 * log_E**2


def lambda_w0(E_MeV: ScalarOrArray, mode: int = 0) -> PolynomialValue:
    """Return the JLMB imaginary-isoscalar normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.
        mode: TALYS ``jlmmode`` selector (0–3). Mode 3 doubles the result
            relative to mode 2 to improve low-energy (E < 1 MeV) accuracy.
            Modes 0, 1, 2 do not affect λW; see :func:`lambda_w1` for those.

    Returns:
        Imaginary isoscalar normalization values.
    """

    E = E_MeV
    f1 = 1.24 - 1.0 / (1.0 + np.exp((E - 4.5) / 2.9))
    f2 = 1.0 + 0.06 * np.exp(-(((E - 14.0) / 3.7) ** 2))
    f3 = 1.0 - 0.09 * np.exp(-(((E - 80.0) / 78.0) ** 2))
    f4 = 1.0 + np.maximum(E - 80.0, 0.0) / 400.0  # Θ(E-80)·(E-80)/400
    result = f1 * f2 * f3 * f4
    if mode == 3:
        result = result * 2.0
    return result


def lambda_v1(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB real-isovector normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Real isovector normalization values.
    """

    return 1.5 - 0.65 / (1.0 + np.exp((E_MeV - 1.3) / 3.0))


def lambda_w1(E_MeV: ScalarOrArray, mode: int = 0) -> PolynomialValue:
    """Return the JLMB imaginary-isovector normalization factor.

    The ``mode`` parameter selects the TALYS ``jlmmode`` prescription for the
    ``alam`` amplitude in the sigmoid correction to the 1.1 base value.
    Coefficients taken from ``talys/source/mom.f90`` (authoritative; the TALYS
    manual has typographical errors for modes 1 and 2).

    Args:
        E_MeV: Projectile energy in MeV.
        mode: TALYS ``jlmmode`` selector:

            - ``0`` (default): ``alam = 0.44`` (Eq. 11.47 standard)
            - ``1``: ``alam = 1.10 * exp(-0.4 * E**0.25)``
            - ``2``: ``alam = 1.375 * exp(-0.2 * sqrt(E))``
            - ``3``: same ``alam`` as mode 2 (λW doubled via :func:`lambda_w0`)

    Returns:
        Imaginary isovector normalization values.
    """

    E = np.asarray(E_MeV, dtype=float)
    alam: PolynomialValue
    if mode == 0:
        alam = 0.44
    elif mode == 1:
        alam = 1.10 * np.exp(-0.4 * E**0.25)
    elif mode in (2, 3):
        alam = 1.375 * np.exp(-0.2 * np.sqrt(E))
    else:
        raise ValueError(f"mode must be 0–3, got {mode}")
    # paper: [1 + (e^((E-40)/50.9))^4]^-1  →  1 / (1 + e^(4(E-40)/50.9))
    f1 = 1.1 + alam / (1.0 + np.exp(4.0 * (E - 40.0) / 50.9))
    f2 = 1.0 - 0.065 * np.exp(-(((E - 40.0) / 13.0) ** 2))
    f3 = 1.0 - 0.083 * np.exp(-(((E - 200.0) / 80.0) ** 2))
    return f1 * f2 * f3


def lambda_vso(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB real spin-orbit normalization depth.

    From ``talys/source/mom.f90``: ``lvso = 40 + exp(-E*0.013)*130``.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Real spin-orbit depth in MeV.
    """
    return 40.0 + 130.0 * np.exp(-0.013 * np.asarray(E_MeV, dtype=float))


def lambda_wso(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB imaginary spin-orbit normalization depth.

    From ``talys/source/mom.f90``: ``lwso = -0.2*(E - 20)``.  Zero at E = 20 MeV.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Imaginary spin-orbit depth in MeV.
    """
    return -0.2 * (np.asarray(E_MeV, dtype=float) - 20.0)


def spin_orbit_jlmb(
    r_q: FloatArray,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    projectile: Projectile,
    r_out: FloatArray | None = None,
) -> FloatArray:
    """Return the JLMB spin-orbit form factor F(r) = (1/r) dρ_SO/dr.

    Uses the Scheerbaum prescription (Nucl. Phys. A257, 77, 1976; Bauge et
    al. 1998, Eq. 30): the effective spin-orbit density weights the nucleons
    unlike the projectile twice as much as the like nucleons:

    - neutron projectile: ``ρ_SO = (2ρ_p + ρ_n) / 3``
    - proton  projectile: ``ρ_SO = (2ρ_n + ρ_p) / 3``

    The derivative dρ_SO/dr is evaluated via :class:`~scipy.interpolate.CubicSpline`
    fitted to the density values at ``r_q``.

    The form is ready for :class:`jitr.xs.coupled.Potential`, whose
    spin-orbit term multiplies ``l·σ = 2 l·s``:

    .. code-block:: python

        F = spin_orbit_jlmb(...)
        Potential(..., spin_orbit=(lambda_vso(E) + 1j * lambda_wso(E)) * F)

    F is negative at the nuclear surface, the same sign as the Thomas term
    ``V_so λ_π² (1/r) df/dr`` of Koning-Delaroche type potentials, so the
    analyzing powers have the same sign. ECIS (as driven by TALYS) instead
    stores ``-½ F``: its potentials carry the opposite sign and its spin-orbit
    term multiplies ``l·s``. With that convention this function reproduces the
    TALYS JLMB elastic cross sections for n + ¹²⁰Sn at 10 MeV.

    Args:
        r_q: Radial grid in fm on which ``rho_n_q`` and ``rho_p_q`` are
            sampled (e.g. ``ILDAFolder.r_q``).
        rho_n_q: Neutron density on ``r_q``, in fm⁻³.
        rho_p_q: Proton density on ``r_q``, in fm⁻³.
        projectile: ``(1, 0)`` for neutrons, ``(1, 1)`` for protons.
        r_out: Output radial grid in fm.  Defaults to ``r_q``.

    Returns:
        Form factor F(r) sampled on ``r_out``, in fm⁻⁴.
    """
    rho_n_arr = np.asarray(rho_n_q, dtype=float)
    rho_p_arr = np.asarray(rho_p_q, dtype=float)
    r_arr = np.asarray(r_q, dtype=float)

    if projectile == (1, 0):
        rho_so = (2.0 * rho_p_arr + rho_n_arr) / 3.0
    elif projectile == (1, 1):
        rho_so = (2.0 * rho_n_arr + rho_p_arr) / 3.0
    else:
        raise ValueError(f"Projectile must be (1,0) [n] or (1,1) [p], got {projectile}")

    drho_dr = CubicSpline(r_arr, rho_so)(r_arr, 1)

    # Thomas form (1/r) dρ_SO/dr; use the L'Hôpital limit dρ/dr at r→0.
    form = np.where(r_arr > 1e-10, drho_dr / r_arr, drho_dr)

    r_eval = r_arr if r_out is None else np.asarray(r_out, dtype=float)
    return np.interp(r_eval, r_arr, form)


def isovector_spin_orbit_jlmb(
    r_q: FloatArray,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    r_out: FloatArray | None = None,
) -> FloatArray:
    r"""Return the isovector part of the Scheerbaum spin-orbit form factor,
    ``F_1 = (F_n - F_p)/2 = (1/r) d/dr [(ρ_p - ρ_n)/6]``, with ``F_n`` and
    ``F_p`` from :func:`spin_orbit_jlmb`.

    The neutron and proton forms are ``F_0 ± F_1``, so the (p,n) spin-orbit
    coupling to the isobaric analog state, in the convention of
    :mod:`jitr.xs.quasielastic_pn`, is

    .. math::
        U_1^{so} = -\frac{2\,(λ_{vso}(E) + i λ_{wso}(E))\, F_1}{\sqrt{N - Z}},

    with ``F_1`` from the target densities.

    Args:
        r_q, rho_n_q, rho_p_q, r_out: As in :func:`spin_orbit_jlmb`.

    Returns:
        Form factor ``F_1`` sampled on ``r_out``, in fm⁻⁴.
    """
    neutron = spin_orbit_jlmb(r_q, rho_n_q, rho_p_q, (1, 0), r_out=r_out)
    proton = spin_orbit_jlmb(r_q, rho_n_q, rho_p_q, (1, 1), r_out=r_out)
    return 0.5 * (neutron - proton)


def _jlmb_nuclear_matter(
    folder,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    projectile: Projectile,
    E: float,
    V_C: FloatArray | None,
    parameterization: str | JLMParameterization | None,
    coeffs_A: FloatArray,
    coeffs_B: FloatArray,
    coeffs_C: FloatArray,
    coeffs_D: FloatArray,
    coeffs_F: FloatArray,
    coeffs_E_F: FloatArray,
    D_damping: float,
    F_damping: float,
) -> tuple[Any, ...]:
    """The JLMB nuclear-matter terms on ``folder.r_q``, before normalization
    and folding: ``(sign, alpha, V0, W0, V1, W1, m_tilde_over_m)``, with
    ``sign`` +1 for neutrons and -1 for protons. Arguments as in
    :func:`potential_JLMB`."""

    params = resolve_parameterization(parameterization)
    if params is not None:
        coeffs_A = params.coeffs_A
        coeffs_B = params.coeffs_B
        coeffs_C = params.coeffs_C
        coeffs_D = params.coeffs_D
        coeffs_F = params.coeffs_F
        coeffs_E_F = params.coeffs_E_F
        D_damping = params.D_damping
        F_damping = params.F_damping
        low_energy_offset = params.low_energy_E_F_offset
        low_energy_coeffs = params.low_energy_E_F_coeffs
        blend_energy = params.E_F_blend_energy
        blend_width = params.E_F_blend_width
    else:
        low_energy_offset = None
        low_energy_coeffs = None
        blend_energy = EPS_F_BLEND_ENERGY
        blend_width = EPS_F_BLEND_WIDTH

    rho_n_array = np.asarray(rho_n_q, dtype=float)
    rho_p_array = np.asarray(rho_p_q, dtype=float)
    rho_q = rho_n_array + rho_p_array
    alpha_q = np.where(rho_q > 1e-12, (rho_n_q - rho_p_q) / rho_q, 0.0)

    if projectile == (1, 1):
        if V_C is None:
            raise ValueError("V_C is required for proton projectiles.")
        V_C_q = np.asarray(V_C, dtype=float)
        if V_C_q.shape != rho_q.shape:
            raise ValueError("V_C must have the same shape as rho_n_q and rho_p_q.")
        E_eff_q = E - V_C_q
        sign = -1.0
    elif projectile == (1, 0):
        if V_C is not None:
            raise ValueError("V_C is only supported for proton projectiles.")
        E_eff_q = np.full_like(folder.r_q, float(E))
        sign = +1.0
    else:
        raise ValueError(f"Projectile must be (1,0) [n] or (1,1) [p], got {projectile}")

    E_F_q = fermi_energy_MeV(
        rho_q,
        coeffs=coeffs_E_F,
        projectile_energy_MeV=E,
        low_energy_offset=low_energy_offset,
        low_energy_coeffs=low_energy_coeffs,
        blend_energy=blend_energy,
        blend_width=blend_width,
    )
    V0_q = V0(rho_q, E_eff_q, coeffs=coeffs_A)
    W0_q = W0(rho_q, E_eff_q, E_F=E_F_q, coeffs=coeffs_D, damping=D_damping)
    V1_q = V1(rho_q, E_eff_q, E_F=E_F_q, coeffs_B=coeffs_B, coeffs_C=coeffs_C)
    W1_q = W1(
        rho_q,
        E_eff_q,
        E_F=E_F_q,
        coeffs_F=coeffs_F,
        coeffs_A=coeffs_A,
        coeffs_C=coeffs_C,
        damping=F_damping,
    )
    # k-mass correction to the imaginary potential per Bauge et al. PRC 58, 1118 (1998).
    # The imaginary NM terms are scaled by m̃/m before folding, which produces
    # the surface-peaked imaginary potential characteristic of the JLMB prescription.
    m_tilde_q = m_tilde_over_m(rho_q, E_eff_q, coeffs=coeffs_C)
    return sign, alpha_q, V0_q, W0_q, V1_q, W1_q, m_tilde_q


def potential_JLMB(
    folder,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    projectile: Projectile,
    target: Target,
    E: float,
    V_C: FloatArray | None = None,
    parameterization: str | JLMParameterization | None = "bauge1998",
    coeffs_A: FloatArray = A_V0,
    coeffs_B: FloatArray = B_N_RE,
    coeffs_C: FloatArray = C_KMASS,
    coeffs_D: FloatArray = D_W0,
    coeffs_F: FloatArray = F_N_IM,
    coeffs_E_F: FloatArray = EPS_F_coeffs,
    D_damping: float = D_DAMPING,
    F_damping: float = F_DAMPING,
    lambda_V: float = 1,
    lambda_W: float = 1,
    lambda_V1: float = 1,
    lambda_W1: float = 1,
    t_r: float = 1.25,
    t_i: float = 1.35,
    r_out: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the finite-range JLMB optical potential.

    Args:
        folder: :class:`~jitr.folding.folding.ILDAFolder` providing the folding grid.
        rho_n_q: Neutron density sampled on ``folder.r_q``.
        rho_p_q: Proton density sampled on ``folder.r_q``.
        projectile: Projectile identifier, ``(1, 0)`` for neutrons or
            ``(1, 1)`` for protons.
        target: Target ``(A, Z)`` tuple.
        E: Projectile energy in MeV.
        V_C: Coulomb potential sampled on ``folder.r_q``. Required for proton
            projectiles and ignored for neutrons.
        parameterization: Named kernel bundle (see
            :func:`resolve_parameterization`). The default ``'bauge1998'`` is
            JLMB; ``'jlm1977'`` gives the original JLM kernels with the JLMB
            k-mass and folding. Pass ``None`` to use the explicit
            ``coeffs_*`` and damping arguments, which are ignored otherwise.
        coeffs_A: Real isoscalar coefficient table.
        coeffs_B: Real isovector coefficient table.
        coeffs_C: Momentum-dependent mass coefficient table.
        coeffs_D: Imaginary isoscalar coefficient table.
        coeffs_F: Imaginary isovector coefficient table.
        coeffs_E_F: Fermi-energy coefficient vector.
        D_damping: Imaginary isoscalar damping parameter in MeV².
        F_damping: Imaginary isovector damping parameter in MeV.
        lambda_V: Real isoscalar normalization.
        lambda_W: Imaginary isoscalar normalization.
        lambda_V1: Real isovector normalization.
        lambda_W1: Imaginary isovector normalization.
        t_r: Real-part Gaussian folding width in fm.
        t_i: Imaginary-part Gaussian folding width in fm.
        r_out: Optional output grid in fm.

    Returns:
        Tuple of folded real and imaginary optical-potential arrays in MeV.

    Raises:
        ValueError: If ``projectile`` is not neutron or proton, or if ``V_C`` is
            missing or malformed for proton calls.
    """

    sign, alpha_q, V0_q, W0_q, V1_q, W1_q, m_tilde_q = _jlmb_nuclear_matter(
        folder,
        rho_n_q,
        rho_p_q,
        projectile,
        E,
        V_C,
        parameterization,
        coeffs_A,
        coeffs_B,
        coeffs_C,
        coeffs_D,
        coeffs_F,
        coeffs_E_F,
        D_damping,
        F_damping,
    )
    V_NM_q = lambda_V * (V0_q + sign * lambda_V1 * alpha_q * V1_q)
    W_NM_q = lambda_W * m_tilde_q * (W0_q + sign * lambda_W1 * alpha_q * W1_q)

    V_r = folder.gaussian_fold(V_NM_q, t=t_r, r_out=r_out)
    W_r = folder.gaussian_fold(W_NM_q, t=t_i, r_out=r_out)
    return np.asarray(V_r, dtype=float), np.asarray(W_r, dtype=float)


def isovector_JLMB(
    folder,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    target: Target,
    E: float,
    V_C: FloatArray | None = None,
    parameterization: str | JLMParameterization | None = "bauge1998",
    coeffs_A: FloatArray = A_V0,
    coeffs_B: FloatArray = B_N_RE,
    coeffs_C: FloatArray = C_KMASS,
    coeffs_D: FloatArray = D_W0,
    coeffs_F: FloatArray = F_N_IM,
    coeffs_E_F: FloatArray = EPS_F_coeffs,
    D_damping: float = D_DAMPING,
    F_damping: float = F_DAMPING,
    lambda_V: float = 1,
    lambda_W: float = 1,
    lambda_V1: float = 1,
    lambda_W1: float = 1,
    t_r: float = 1.25,
    t_i: float = 1.35,
    r_out: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Evaluate the folded isovector part of the JLMB potential.

    In JLMB the isovector part of the nuclear-matter potential is the term
    in the local asymmetry ``α = (ρ_n - ρ_p)/ρ``, entering the neutron
    potential with ``+`` and the proton potential with ``-``. This function
    folds it alone:

    .. math::
        F(r) = g \ast \left[λ_V λ_{V_1}\, α V_1
               + i\, λ_W λ_{W_1}\, (\tilde m/m)\, α W_1\right],

    so that at a common effective energy ``U_n = U_0 + F`` and
    ``U_p = U_0 - F``. This is the Lane potential of the target: in the
    convention of :mod:`jitr.xs.quasielastic_pn`, the (p,n) coupling to the
    isobaric analog state is

    .. math::
        U_1 = -\frac{2 F}{\sqrt{N - Z}},

    with ``F`` from the target densities. Differencing the full neutron and
    proton potentials instead also picks up the Coulomb shift of the proton
    energy in the isoscalar terms and the density change between target and
    residual. See :func:`isovector_spin_orbit_jlmb` for the spin-orbit part.

    Args:
        folder: :class:`~jitr.folding.folding.ILDAFolder` providing the folding grid.
        rho_n_q: Neutron density sampled on ``folder.r_q``.
        rho_p_q: Proton density sampled on ``folder.r_q``.
        target: Target ``(A, Z)`` tuple.
        E: Projectile energy in MeV.
        V_C: Optional Coulomb potential sampled on ``folder.r_q``. If given,
            the nuclear-matter terms are evaluated at the proton effective
            energy ``E - V_C``; otherwise at ``E``.
        parameterization, coeffs_A, coeffs_B, coeffs_C, coeffs_D, coeffs_F,
            coeffs_E_F, D_damping, F_damping, lambda_V, lambda_W, lambda_V1,
            lambda_W1, t_r, t_i, r_out: As in :func:`potential_JLMB`.

    Returns:
        Tuple of folded real and imaginary isovector arrays ``F`` in MeV.
    """
    sign, alpha_q, V0_q, W0_q, V1_q, W1_q, m_tilde_q = _jlmb_nuclear_matter(
        folder,
        rho_n_q,
        rho_p_q,
        (1, 1) if V_C is not None else (1, 0),
        E,
        V_C,
        parameterization,
        coeffs_A,
        coeffs_B,
        coeffs_C,
        coeffs_D,
        coeffs_F,
        coeffs_E_F,
        D_damping,
        F_damping,
    )
    V_NM_q = lambda_V * lambda_V1 * alpha_q * V1_q
    W_NM_q = lambda_W * m_tilde_q * lambda_W1 * alpha_q * W1_q

    V_r = folder.gaussian_fold(V_NM_q, t=t_r, r_out=r_out)
    W_r = folder.gaussian_fold(W_NM_q, t=t_i, r_out=r_out)
    return np.asarray(V_r, dtype=float), np.asarray(W_r, dtype=float)


def pn_transition_bauge2001(
    folder,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    target: Target,
    E: float,
    asymmetry: Literal["local", "average"] = "local",
    energy_prescription: Literal["paper", "local_midpoint"] = "paper",
    V_C: FloatArray | None = None,
    k_mass: bool = True,
    parameterization: str | JLMParameterization | None = "bauge1998",
    coeffs_A: FloatArray = A_V0,
    coeffs_B: FloatArray = B_N_RE,
    coeffs_C: FloatArray = C_KMASS,
    coeffs_D: FloatArray = D_W0,
    coeffs_F: FloatArray = F_N_IM,
    coeffs_E_F: FloatArray = EPS_F_coeffs,
    D_damping: float = D_DAMPING,
    F_damping: float = F_DAMPING,
    lambda_V: float = 1,
    lambda_W: float = 1,
    lambda_V1: float = 1,
    lambda_W1: float = 1,
    t_r: float = 1.25,
    t_i: float = 1.35,
    r_out: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Evaluate the JLMB (p,n) IAS transition potential of Bauge, Delaroche
    and Girod, Phys. Rev. C 63, 024607 (2001), Eq. (7):

    .. math::
        U_{QE}(r) = 2 \sqrt{\frac{α(r)}{A}}
            \left[λ_V λ_{V_1} V_1(r) + i\, λ_W λ_{W_1} W_1(r)\right],

    where :math:`V_1(r)` and :math:`W_1(r)` are the folded nuclear-matter
    isovector strengths, with no asymmetry inside the fold. The asymmetry
    multiplies after folding. With ``asymmetry="local"`` it is
    ``α(r) = (ρ_n - ρ_p)/ρ`` at ``r``, replaced by ``(N - Z)/A`` wherever
    ``ρ_p ≥ ρ_n``, as in the paper; with ``"average"`` it is ``(N - Z)/A``
    everywhere.

    The energy of the nuclear-matter terms follows ``energy_prescription``:

    - ``"paper"``: the constant ``E``. The paper uses ``E = E_p + Q/2``
      (``Q`` the IAS Q-value), without the Coulomb shift of the proton
      energy.
    - ``"local_midpoint"``: ``E - V_C(r)/2``. With ``E = E_p + Q/2`` this is
      the average of the proton's local energy ``E_p - V_C(r)``, at which its
      own JLMB potential is evaluated, and the neutron's ``E_p + Q``.
      Requires ``V_C``.

    The normalizations ``lambda_*`` are evaluated by the caller, at ``E``.

    In Bauge, Delaroche and Girod, Phys. Rev. C 58, 1118 (1998), Eq. (8),
    the isovector imaginary strength is ``W_1 = (m/m̄) Im N`` and the k-mass
    factor ``m̃/m`` scales the imaginary part of the elastic potential
    afterwards. ``k_mass=True`` applies it to ``W_1`` here too, as
    :func:`potential_JLMB` does for elastic scattering; ``k_mass=False``
    leaves ``W_1`` as defined in Eq. (8).

    For densities with a constant asymmetry this equals ``2F/sqrt(N - Z)``
    with ``F`` from :func:`isovector_JLMB`. In general the two differ in
    where the local asymmetry enters: inside the fold there, after it here.
    In the convention of :mod:`jitr.xs.quasielastic_pn` the coupling is
    ``U1 = -U_QE``.

    Args:
        folder: :class:`~jitr.folding.folding.ILDAFolder` providing the folding grid.
        rho_n_q: Neutron density sampled on ``folder.r_q``.
        rho_p_q: Proton density sampled on ``folder.r_q``.
        target: Target ``(A, Z)`` tuple.
        E: Energy in MeV at which to evaluate the transition potential.
        asymmetry: ``"local"`` (the paper's choice) or ``"average"``.
        energy_prescription: ``"paper"`` or ``"local_midpoint"``.
        V_C: Proton Coulomb potential sampled on ``folder.r_q``, required for
            ``energy_prescription="local_midpoint"``.
        k_mass: Whether ``W_1`` carries the k-mass factor ``m̃/m``.
        parameterization, coeffs_A, coeffs_B, coeffs_C, coeffs_D, coeffs_F,
            coeffs_E_F, D_damping, F_damping, lambda_V, lambda_W, lambda_V1,
            lambda_W1, t_r, t_i, r_out: As in :func:`potential_JLMB`.

    Returns:
        Tuple of real and imaginary ``U_QE`` in MeV on ``r_out``
        (``folder.r_q`` if not given).
    """
    if asymmetry not in ("local", "average"):
        raise ValueError(f"asymmetry must be 'local' or 'average', got {asymmetry!r}")
    if energy_prescription == "paper":
        projectile, shift = (1, 0), None
    elif energy_prescription == "local_midpoint":
        if V_C is None:
            raise ValueError("energy_prescription='local_midpoint' requires V_C")
        projectile, shift = (1, 1), 0.5 * np.asarray(V_C, dtype=float)
    else:
        raise ValueError(
            "energy_prescription must be 'paper' or 'local_midpoint', "
            f"got {energy_prescription!r}"
        )
    _, _, _, _, V1_q, W1_q, m_tilde_q = _jlmb_nuclear_matter(
        folder,
        rho_n_q,
        rho_p_q,
        projectile,
        E,
        shift,
        parameterization,
        coeffs_A,
        coeffs_B,
        coeffs_C,
        coeffs_D,
        coeffs_F,
        coeffs_E_F,
        D_damping,
        F_damping,
    )
    r = folder.r_q if r_out is None else np.asarray(r_out, dtype=float)
    V1_r = folder.gaussian_fold(lambda_V * lambda_V1 * V1_q, t=t_r, r_out=r)
    W1_q = m_tilde_q * W1_q if k_mass else W1_q
    W1_r = folder.gaussian_fold(lambda_W * lambda_W1 * W1_q, t=t_i, r_out=r)

    A, Z = target
    alpha_average = (A - 2 * Z) / A
    alpha = np.full_like(r, alpha_average)
    if asymmetry == "local":
        rho_n = np.interp(r, folder.r_q, np.asarray(rho_n_q, dtype=float))
        rho_p = np.interp(r, folder.r_q, np.asarray(rho_p_q, dtype=float))
        rho = rho_n + rho_p
        local = (rho > 1e-12) & (rho_n > rho_p)
        alpha[local] = (rho_n[local] - rho_p[local]) / rho[local]

    scale = 2.0 * np.sqrt(alpha / A)
    return (
        np.asarray(scale * V1_r, dtype=float),
        np.asarray(scale * W1_r, dtype=float),
    )
