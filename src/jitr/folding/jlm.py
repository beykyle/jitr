"""JLM and JLMB local-density optical-potential helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

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
# k-mass ratio (multiplies V_1 and divides W_0, W_1)
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
# Imaginary isoscalar kernel, D = 600 MeV²
# Physical W_0 = (m/m̃) · W̄_0                                     [Eq. 7]
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
# Imaginary isovector kernel, F = 1 MeV
# Physical W_1 = (m/m̃) · Im N_1                                   [Eq. 12]
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
# TALYS revised 1998 imaginary kernels and damping
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


ORIGINAL_PARAMETERIZATION = JLMParameterization(
    name="original",
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

TALYS_PARAMETERIZATION = JLMParameterization(
    name="talys",
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


def resolve_parameterization(
    parameterization: str | JLMParameterization | None,
) -> JLMParameterization | None:
    """Resolve a named or explicit JLM parameterization bundle."""

    if parameterization is None:
        return None
    if isinstance(parameterization, JLMParameterization):
        return parameterization
    normalized = parameterization.strip().lower()
    if normalized in {"original", "jlm", "paper"}:
        return ORIGINAL_PARAMETERIZATION
    if normalized in {"talys", "revised", "jlmb-talys"}:
        return TALYS_PARAMETERIZATION
    raise ValueError(
        "parameterization must be one of 'original', 'talys', or a "
        "JLMParameterization instance."
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
        parameterization: Named kernel bundle. ``'original'`` preserves the
            existing paper-style implementation, while ``'talys'`` enables the
            revised TALYS-compatible low-energy handling.
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
    """Return the JLMB spin-orbit form factor F(r) = -½ (1/r) dρ_SO/dr.

    Uses the Scheerbaum prescription (Nucl. Phys. A257, 77, 1976): the
    effective spin-orbit density up-weights the minority species:

    - neutron projectile: ``ρ_SO = (2ρ_p + ρ_n) / 3``
    - proton  projectile: ``ρ_SO = (2ρ_n + ρ_p) / 3``

    The derivative dρ_SO/dr is evaluated via :class:`~scipy.interpolate.CubicSpline`
    fitted to the density values at ``r_q``.  The leading factor of −½ matches
    the TALYS/ECIS type-5 spin-orbit convention (Scheerbaum form stored with
    opposite sign and halved, confirmed numerically against ECIS output for
    n + ¹²⁰Sn at 10 MeV).

    The full complex spin-orbit potential is assembled by the caller as:

    .. code-block:: python

        V_SO(r) = (lambda_vso(E) + 1j * lambda_wso(E)) * spin_orbit_jlmb(...)

    and passed to ``workspace.xs(spin_orbit_potential=...)``.  jitr applies
    the L·S eigenvalue (l/2 for j=l+½, −(l+1)/2 for j=l−½) internally.

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

    result = -0.5 * form
    r_eval = r_arr if r_out is None else np.asarray(r_out, dtype=float)
    return np.interp(r_eval, r_arr, result)


def potential_JLMB(
    folder,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    projectile: Projectile,
    target: Target,
    E: float,
    V_C: FloatArray | None = None,
    parameterization: str | JLMParameterization | None = None,
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
        parameterization: Named kernel bundle. ``'original'`` preserves the
            existing behavior, while ``'talys'`` enables the revised
            TALYS-compatible coefficients and low-energy Fermi-energy blend.
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
    V_NM_q = lambda_V * (V0_q + sign * lambda_V1 * alpha_q * V1_q)
    W_NM_q = lambda_W * m_tilde_q * (W0_q + sign * lambda_W1 * alpha_q * W1_q)

    V_r = folder.gaussian_fold(V_NM_q, t=t_r, r_out=r_out)
    W_r = folder.gaussian_fold(W_NM_q, t=t_i, r_out=r_out)
    return np.asarray(V_r, dtype=float), np.asarray(W_r, dtype=float)
