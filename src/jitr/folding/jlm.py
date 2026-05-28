import numpy as np

from ..utils import poly
from ..utils.constants import ALPHA, HBARC

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


def fermi_energy_MeV(rho_fm3: float, coeffs: np.ndarray = EPS_F_coeffs) -> float:
    return poly.poly1d(rho_fm3, coeffs, start_i=1)


def V0(rho_fm3, E_MeV, coeffs=A_V0):
    return poly.poly2d(rho_fm3, E_MeV, coeffs, start_i=1, start_j=0)


def W0(rho_fm3, E_MeV, E_F, coeffs=D_W0, damping=D_DAMPING):
    return poly.poly2d(rho_fm3, E_MeV, coeffs, start_i=1, start_j=0) / (
        1 + damping / (E_MeV - E_F) ** 2
    )


def m_tilde_over_m(rho_fm3, E_MeV, coeffs=C_KMASS):
    """k-mass m̃/m = 1 − Σ c_ij ρ^i E^(j-1)    [Eq. 29 / Table III]"""
    return 1.0 - poly.poly2d(rho_fm3, E_MeV, coeffs, start_i=1, start_j=0)


def eff_mass(rho_fm3, E_MeV, coeffs=A_V0):
    """m*(ρ,E)/m = 1 − ∂V₀/∂E    [Eq. 14]"""
    c, si, sj = poly.poly2d_deriv(coeffs, start_i=1, start_j=0, wrt="y")
    return 1.0 - poly.poly2d(rho_fm3, E_MeV, c, start_i=si, start_j=sj)


def E_mass(rho_fm3, E_MeV, coeffs_A=A_V0, coeffs_C=C_KMASS):
    """m̄(ρ,E)/m via Eq. 15:   m*/m = (m̃/m)·(m̄/m)."""
    return eff_mass(rho_fm3, E_MeV, coeffs=coeffs_A) / m_tilde_over_m(
        rho_fm3, E_MeV, coeffs=coeffs_C
    )


def V1(rho_fm3, E_MeV, E_F, coeffs_B=B_N_RE, coeffs_C=C_KMASS):
    """V₁ = (m̃/m) · Re N    [Eq. 11]"""
    re_n = poly.poly2d(rho_fm3, E_MeV, coeffs_B, start_i=1, start_j=0)
    return m_tilde_over_m(rho_fm3, E_MeV, coeffs=coeffs_C) * re_n


def W1(
    rho_fm3,
    E_MeV,
    E_F,
    coeffs_F=F_N_IM,
    coeffs_A=A_V0,
    coeffs_C=C_KMASS,
    damping=F_DAMPING,
):
    """W₁ = (m/m̄) · Im N    [Eq. 12]"""
    im_n = poly.poly2d(rho_fm3, E_MeV, coeffs_F, start_i=1, start_j=0) / (
        1 + damping / (E_MeV - E_F)
    )
    return im_n / E_mass(rho_fm3, E_MeV, coeffs_A=coeffs_A, coeffs_C=coeffs_C)


def Delta_C(rho_fm3, E_MeV, V_C_MeV, coeffs_A=A_V0, linear=True):
    """Real Coulomb correction Δ_C(ρ, E) for protons    [JLM 1977 Eqs. 17–18]

    Defined exactly as
        Δ_C(ρ, E) = V₀(ρ, E − V_C) − V₀(ρ, E).                       (Eq. 17)

    Two evaluation modes:

    `linear=True`  (paper's preferred form, Eq. 18)
        Δ_C ≈ (m*(ρ,E)/m − 1) · V_C
        Uses m*/m at the *physical* energy E, so the polynomial is only
        ever evaluated inside its [10, 160] MeV fit range. Linear in V_C.

    `linear=False`  (direct substitution, Eq. 16/17)
        Δ_C = V₀(ρ, E − V_C) − V₀(ρ, E)
        Exact within the LDA, but the cubic in E in V₀ extrapolates
        wildly once E − V_C drops below ~10 MeV (low-E protons on heavy
        targets).
    """
    if linear:
        return (eff_mass(rho_fm3, E_MeV, coeffs=coeffs_A) - 1.0) * V_C_MeV
    return V0(rho_fm3, E_MeV - V_C_MeV, coeffs=coeffs_A) - V0(
        rho_fm3, E_MeV, coeffs=coeffs_A
    )


def potential_JLM(
    rgrid,
    rho_grid,
    projectile,
    target,
    E,
    coeffs_A=A_V0,
    coeffs_B=B_N_RE,
    coeffs_C=C_KMASS,
    coeffs_D=D_W0,
    D_damping=D_DAMPING,
    coeffs_F=F_N_IM,
    F_damping=F_DAMPING,
    coeffs_E_F=EPS_F_coeffs,
):
    A, Z = target
    N = A - Z
    alpha = (N - Z) / A
    E_F = fermi_energy_MeV(rho_fm3=rho_grid, coeffs=coeffs_E_F)
    V0_grid = V0(rho_fm3=rho_grid, E_MeV=E, coeffs=coeffs_A)
    if projectile == (1, 1):
        # TODO calculate VC from rho_p
        RC = 1.2 * A ** (1 / 3)
        VC = 6.0 * Z * ALPHA * HBARC / (5 * RC)
        DelC = Delta_C(
            rho_fm3=rho_grid, E_MeV=E, V_C_MeV=VC, coeffs_A=coeffs_A, linear=True
        )
        DeltaE = E - DelC
        W0_grid = W0(rho_fm3=rho_grid, E_MeV=DeltaE, E_F=E_F, coeffs=coeffs_D)
        V1_grid = V1(
            rho_fm3=rho_grid,
            E_MeV=DeltaE,
            coeffs_B=coeffs_B,
            coeffs_C=coeffs_C,
            E_F=E_F,
        )
        W1_grid = W1(
            rho_fm3=rho_grid,
            E_MeV=DeltaE,
            E_F=E_F,
            coeffs_A=coeffs_A,
            coeffs_C=coeffs_C,
            coeffs_F=coeffs_F,
            damping=F_damping,
        )
        return V0_grid + DelC + alpha * V1_grid, W0_grid + alpha * W1_grid

    elif projectile == (1, 0):
        DeltaE = E
        W0_grid = W0(rho_fm3=rho_grid, E_MeV=E, E_F=E_F, coeffs=coeffs_D)
        V1_grid = V1(
            rho_fm3=rho_grid, E_MeV=E, coeffs_B=coeffs_B, coeffs_C=coeffs_C, E_F=E_F
        )
        W1_grid = W1(
            rho_fm3=rho_grid,
            E_MeV=E,
            E_F=E_F,
            coeffs_A=coeffs_A,
            coeffs_C=coeffs_C,
            coeffs_F=coeffs_F,
            damping=F_damping,
        )
        return V0_grid + alpha * V1_grid, W0_grid + alpha * W1_grid
    else:
        raise ValueError(
            f"Projectile must be neutron (1,0) or proton (1,1), recieved {projectile}"
        )
    return


def lambda_v0(E_MeV):
    """λ_V(E) — real isoscalar.       Eq. 8: 0.95–0.99 over 1 keV–200 MeV."""
    log_E = np.log(1000.0 * E_MeV)  # paper has ln(1000 E), E in MeV
    return 0.951 + 0.0008 * log_E + 0.00018 * log_E**2


def lambda_w0(E_MeV):
    """λ_W(E) — imaginary isoscalar.   Eq. 9."""
    E = E_MeV
    f1 = 1.24 - 1.0 / (1.0 + np.exp((E - 4.5) / 2.9))
    f2 = 1.0 + 0.06 * np.exp(-(((E - 14.0) / 3.7) ** 2))
    f3 = 1.0 - 0.09 * np.exp(-(((E - 80.0) / 78.0) ** 2))
    f4 = 1.0 + np.maximum(E - 80.0, 0.0) / 400.0  # Θ(E-80)·(E-80)/400
    return f1 * f2 * f3 * f4


def lambda_v1(E_MeV):
    """λ_V1(E) — real isovector enhancement.    Eq. 10: 1.1 → 1.5."""
    return 1.5 - 0.65 / (1.0 + np.exp((E_MeV - 1.3) / 3.0))


def lambda_w1(E_MeV):
    """λ_W1(E) — imaginary isovector enhancement.    Eq. 11."""
    E = E_MeV
    # paper: [1 + (e^((E-40)/50.9))^4]^-1  →  1 / (1 + e^(4(E-40)/50.9))
    f1 = 1.1 + 0.44 / (1.0 + np.exp(4.0 * (E - 40.0) / 50.9))
    f2 = 1.0 - 0.065 * np.exp(-(((E - 40.0) / 13.0) ** 2))
    f3 = 1.0 - 0.083 * np.exp(-(((E - 200.0) / 80.0) ** 2))
    return f1 * f2 * f3


def potential_JLMB(
    folder,
    rho_n_q,
    rho_p_q,
    projectile,
    target,
    E,
    coulomb_mode="density",
    coulomb_R_C=None,
    include_exchange=False,
    coeffs_A=A_V0,
    coeffs_B=B_N_RE,
    coeffs_C=C_KMASS,
    coeffs_D=D_W0,
    coeffs_F=F_N_IM,
    coeffs_E_F=EPS_F_coeffs,
    D_damping=D_DAMPING,
    F_damping=F_DAMPING,
    lambda_V=1,
    lambda_W=1,
    lambda_V1=1,
    lambda_W1=1,
    t_r=1.25,
    t_i=1.35,
    r_out=None,
):
    A, Z = target
    rho_q = rho_n_q + rho_p_q
    alpha_q = np.where(rho_q > 1e-12, (rho_n_q - rho_p_q) / rho_q, 0.0)

    if projectile == (1, 1):
        V_C_q = folder.V_coulomb(
            rho_p_q,
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
    return V_r, W_r
