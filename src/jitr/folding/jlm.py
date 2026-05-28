"""JLM and JLMB local-density optical-potential helpers."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..utils import poly
from ..utils.constants import ALPHA, HBARC

FloatArray: TypeAlias = NDArray[np.float64]
ScalarOrArray: TypeAlias = float | FloatArray
PolynomialValue: TypeAlias = float | FloatArray
Projectile: TypeAlias = tuple[int, int]
Target: TypeAlias = tuple[int, int]

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

# ---------------------------------------------------------------------------
# Eq. 27 — Fermi energy ε_F(ρ) = ρ·(−510.8 + 3222·ρ − 6250·ρ²)  [MeV]
# ---------------------------------------------------------------------------
EPS_F_coeffs = np.array([-510.8, 3222.0, -6250.0])  # multiplied by ρ¹, ρ², ρ³


def fermi_energy_MeV(
    rho_fm3: ScalarOrArray,
    coeffs: FloatArray = EPS_F_coeffs,
) -> PolynomialValue:
    """Return the local Fermi energy in MeV.

    Args:
        rho_fm3: Matter density in fm⁻³.
        coeffs: Polynomial coefficients for the JLM Fermi-energy fit.

    Returns:
        Fermi energy evaluated at ``rho_fm3``.
    """

    return poly.poly1d(rho_fm3, coeffs, start_i=1)


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
) -> PolynomialValue:
    """Return the imaginary isoscalar JLM self-energy component.

    Args:
        rho_fm3: Matter density in fm⁻³.
        E_MeV: Projectile energy in MeV.
        E_F: Local Fermi energy in MeV.
        coeffs: Polynomial coefficients for the fit.
        damping: Imaginary-part damping parameter in MeV².

    Returns:
        Imaginary isoscalar self-energy values in MeV.
    """

    energy_diff_sq = np.asarray((E_MeV - E_F) ** 2, dtype=float)
    damping_term = np.divide(
        damping,
        energy_diff_sq,
        out=np.full(energy_diff_sq.shape, np.inf, dtype=float),
        where=energy_diff_sq != 0.0,
    )
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

    Returns:
        Imaginary isovector self-energy values in MeV.
    """

    energy_diff = np.asarray(E_MeV - E_F, dtype=float)
    damping_term = np.divide(
        damping,
        energy_diff,
        out=np.full(energy_diff.shape, np.inf, dtype=float),
        where=energy_diff != 0.0,
    )
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
    coeffs_A: FloatArray = A_V0,
    coeffs_B: FloatArray = B_N_RE,
    coeffs_C: FloatArray = C_KMASS,
    coeffs_D: FloatArray = D_W0,
    D_damping: float = D_DAMPING,
    coeffs_F: FloatArray = F_N_IM,
    F_damping: float = F_DAMPING,
    coeffs_E_F: FloatArray = EPS_F_coeffs,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the local-density JLM optical potential.

    Args:
        rgrid: Radial grid associated with ``rho_grid``. Retained for API
            compatibility; the local-density evaluation only uses ``rho_grid``.
        rho_grid: Matter-density values sampled on ``rgrid``.
        projectile: Projectile identifier, ``(1, 0)`` for neutrons or
            ``(1, 1)`` for protons.
        target: Target ``(A, Z)`` tuple.
        E: Projectile energy in MeV.
        coeffs_A: Real isoscalar coefficient table.
        coeffs_B: Real isovector coefficient table.
        coeffs_C: Momentum-dependent mass coefficient table.
        coeffs_D: Imaginary isoscalar coefficient table.
        D_damping: Imaginary isoscalar damping parameter in MeV².
        coeffs_F: Imaginary isovector coefficient table.
        F_damping: Imaginary isovector damping parameter in MeV.
        coeffs_E_F: Fermi-energy coefficient vector.

    Returns:
        Tuple of real and imaginary optical-potential arrays in MeV.

    Raises:
        ValueError: If ``projectile`` is not neutron or proton.
    """

    _ = rgrid
    rho_array = np.asarray(rho_grid, dtype=float)
    A, Z = target
    N = A - Z
    alpha = (N - Z) / A
    E_F = fermi_energy_MeV(rho_fm3=rho_array, coeffs=coeffs_E_F)
    V0_grid = V0(rho_fm3=rho_array, E_MeV=E, coeffs=coeffs_A)
    if projectile == (1, 1):
        RC = 1.2 * A ** (1 / 3)
        VC = 6.0 * Z * ALPHA * HBARC / (5 * RC)
        DelC = Delta_C(
            rho_fm3=rho_array, E_MeV=E, V_C_MeV=VC, coeffs_A=coeffs_A, linear=True
        )
        E_eff = E - VC
        sign = -1.0
    elif projectile == (1, 0):
        DelC = 0.0
        E_eff = E
        sign = +1.0
    else:
        raise ValueError(
            f"Projectile must be neutron (1,0) or proton (1,1), received {projectile}"
        )

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
    return (
        np.asarray(V0_grid + DelC + sign * alpha * V1_grid, dtype=float),
        np.asarray(W0_grid + sign * alpha * W1_grid, dtype=float),
    )


def lambda_v0(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB real-isoscalar normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Real isoscalar normalization values.
    """

    log_E = np.log(1000.0 * E_MeV)  # paper has ln(1000 E), E in MeV
    return 0.951 + 0.0008 * log_E + 0.00018 * log_E**2


def lambda_w0(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB imaginary-isoscalar normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Imaginary isoscalar normalization values.
    """

    E = E_MeV
    f1 = 1.24 - 1.0 / (1.0 + np.exp((E - 4.5) / 2.9))
    f2 = 1.0 + 0.06 * np.exp(-(((E - 14.0) / 3.7) ** 2))
    f3 = 1.0 - 0.09 * np.exp(-(((E - 80.0) / 78.0) ** 2))
    f4 = 1.0 + np.maximum(E - 80.0, 0.0) / 400.0  # Θ(E-80)·(E-80)/400
    return f1 * f2 * f3 * f4


def lambda_v1(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB real-isovector normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Real isovector normalization values.
    """

    return 1.5 - 0.65 / (1.0 + np.exp((E_MeV - 1.3) / 3.0))


def lambda_w1(E_MeV: ScalarOrArray) -> PolynomialValue:
    """Return the JLMB imaginary-isovector normalization factor.

    Args:
        E_MeV: Projectile energy in MeV.

    Returns:
        Imaginary isovector normalization values.
    """

    E = E_MeV
    # paper: [1 + (e^((E-40)/50.9))^4]^-1  →  1 / (1 + e^(4(E-40)/50.9))
    f1 = 1.1 + 0.44 / (1.0 + np.exp(4.0 * (E - 40.0) / 50.9))
    f2 = 1.0 - 0.065 * np.exp(-(((E - 40.0) / 13.0) ** 2))
    f3 = 1.0 - 0.083 * np.exp(-(((E - 200.0) / 80.0) ** 2))
    return f1 * f2 * f3


def potential_JLMB(
    folder,
    rho_n_q: FloatArray,
    rho_p_q: FloatArray,
    projectile: Projectile,
    target: Target,
    E: float,
    coulomb_mode: str = "density",
    coulomb_R_C: str | float | None = None,
    include_exchange: bool = False,
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
        coulomb_mode: Coulomb model passed to :meth:`ILDAFolder.V_coulomb`.
        coulomb_R_C: Coulomb radius override for the uniform-sphere mode.
        include_exchange: Whether to include the Coulomb exchange correction.
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
        ValueError: If ``projectile`` is not neutron or proton.
    """

    A, Z = target
    rho_n_array = np.asarray(rho_n_q, dtype=float)
    rho_p_array = np.asarray(rho_p_q, dtype=float)
    rho_q = rho_n_array + rho_p_array
    alpha_q = np.where(rho_q > 1e-12, (rho_n_q - rho_p_q) / rho_q, 0.0)

    if projectile == (1, 1):
        V_C_q = folder.V_coulomb(
            rho_p_array,
            mode=coulomb_mode,
            R_C=coulomb_R_C,
            include_exchange=include_exchange,
        )
        E_eff_q = E - V_C_q
        sign = -1.0
    elif projectile == (1, 0):
        E_eff_q = np.full_like(folder.r_q, float(E))
        sign = +1.0
    else:
        raise ValueError(f"Projectile must be (1,0) [n] or (1,1) [p], got {projectile}")

    E_F_q = fermi_energy_MeV(rho_q, coeffs=coeffs_E_F)
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
    V_NM_q = lambda_V * (V0_q + sign * lambda_V1 * alpha_q * V1_q)
    W_NM_q = lambda_W * (W0_q + sign * lambda_W1 * alpha_q * W1_q)

    V_r = folder.gaussian_fold(V_NM_q, t=t_r, r_out=r_out)
    W_r = folder.gaussian_fold(W_NM_q, t=t_i, r_out=r_out)
    return np.asarray(V_r, dtype=float), np.asarray(W_r, dtype=float)
