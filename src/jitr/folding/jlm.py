"""JLM and JLMB optical-potential models built on generic folding primitives."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..utils.constants import ALPHA, HBARC, MASS_N, MASS_P
from .folding import gaussian_fold
from .nuclear_matter_self_energy import RHO_SAT, NMSelfEnergy

M_NUCLEON: float = 0.5 * (MASS_N + MASS_P)
E2: float = ALPHA * HBARC

T_R_DEFAULT: float = 1.20
T_I_DEFAULT: float = 1.75


@dataclass(frozen=True)
class JLMV0Parameters:
    """Coefficients for the isoscalar real self-energy component."""

    energy_constant: float = 108.0
    energy_linear: float = 0.61
    energy_quadratic: float = 0.00163
    density_quadratic: float = 0.51


@dataclass(frozen=True)
class JLMW0Parameters:
    """Coefficients for the isoscalar imaginary self-energy component."""

    fermi_energy: float = -8.0
    energy_width: float = 25.0
    saturation_strength: float = 22.0
    density_quadratic: float = 0.20


@dataclass(frozen=True)
class JLMV1Parameters:
    """Coefficients for the isovector real self-energy component."""

    energy_constant: float = 14.0
    energy_linear: float = 0.04
    density_quadratic: float = 0.15


@dataclass(frozen=True)
class JLMW1Parameters:
    """Coefficients for the isovector imaginary self-energy component."""

    fermi_energy: float = -8.0
    energy_width: float = 30.0
    saturation_strength: float = 12.0


@dataclass(frozen=True)
class JLMSelfEnergyModelParameters:
    """Full parameter bundle for the analytical JLM self-energy model."""

    rho_sat: float = RHO_SAT
    V0: JLMV0Parameters = field(default_factory=JLMV0Parameters)
    W0: JLMW0Parameters = field(default_factory=JLMW0Parameters)
    V1: JLMV1Parameters = field(default_factory=JLMV1Parameters)
    W1: JLMW1Parameters = field(default_factory=JLMW1Parameters)


@dataclass(frozen=True)
class JLMBLambdaModelParameters:
    """Coefficient bundle for the JLMB renormalization-factor helper."""

    lambda_V0_offset: float = 1.0
    lambda_V0_slope: float = 0.0005
    lambda_V0_reference_energy: float = 30.0
    lambda_W0_offset: float = 0.6
    lambda_W0_amplitude: float = 0.5
    lambda_W0_center_energy: float = 5.0
    lambda_W0_width: float = 20.0
    lambda_V1_value: float = 1.0
    lambda_W1_value: float = 1.6


DEFAULT_JLM_SELF_ENERGY_PARAMETERS = JLMSelfEnergyModelParameters()
DEFAULT_JLMB_LAMBDA_MODEL_PARAMETERS = JLMBLambdaModelParameters()


class JLMSelfEnergy(NMSelfEnergy):
    """Analytical JLM-style nuclear-matter self-energy parameterization."""

    def __init__(
        self,
        projectile: str = "n",
        model_parameters: JLMSelfEnergyModelParameters | None = None,
    ) -> None:
        super().__init__(projectile=projectile)
        self.model_parameters = (
            DEFAULT_JLM_SELF_ENERGY_PARAMETERS
            if model_parameters is None
            else model_parameters
        )

    def V0(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        coefficients = self.model_parameters.V0
        x = np.asarray(rho, dtype=float) / self.model_parameters.rho_sat
        prefac = -(
            coefficients.energy_constant
            - coefficients.energy_linear * E
            + coefficients.energy_quadratic * E**2
        )
        return prefac * x * (1.0 - coefficients.density_quadratic * x)

    def W0(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        coefficients = self.model_parameters.W0
        x = np.asarray(rho, dtype=float) / self.model_parameters.rho_sat
        W_sat = -coefficients.saturation_strength * (
            (E - coefficients.fermi_energy) ** 2
            / ((E - coefficients.fermi_energy) ** 2 + coefficients.energy_width**2)
        )
        return W_sat * x * (1.0 - coefficients.density_quadratic * x)

    def V1(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        coefficients = self.model_parameters.V1
        x = np.asarray(rho, dtype=float) / self.model_parameters.rho_sat
        prefac = coefficients.energy_constant - coefficients.energy_linear * E
        return prefac * x * (1.0 - coefficients.density_quadratic * x)

    def W1(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        coefficients = self.model_parameters.W1
        x = np.asarray(rho, dtype=float) / self.model_parameters.rho_sat
        return (
            -coefficients.saturation_strength
            * (
                (E - coefficients.fermi_energy) ** 2
                / ((E - coefficients.fermi_energy) ** 2 + coefficients.energy_width**2)
            )
            * x
        )


def coulomb_potential_center(Z: int, A: int, r_C: float = 1.25) -> float:
    """Approximate the central Coulomb potential for a uniform sphere."""
    R_C = r_C * (A ** (1.0 / 3.0))
    return 1.5 * Z * E2 / R_C


@dataclass
class JLMParameters:
    """Tunable parameters of the JLM/JLMB folding construction."""

    t_R: float = T_R_DEFAULT
    t_I: float = T_I_DEFAULT
    lambda_V0: float = 1.0
    lambda_W0: float = 1.0
    lambda_V1: float = 1.0
    lambda_W1: float = 1.0
    apply_coulomb_shift: bool = True
    n_quad: int = 400
    r_max: float = 20.0


def jlmb_lambda_factors(
    E: float,
    model_parameters: JLMBLambdaModelParameters | None = None,
) -> tuple[float, float, float, float]:
    """Return the default JLMB renormalization factors at energy ``E``."""
    coefficients = (
        DEFAULT_JLMB_LAMBDA_MODEL_PARAMETERS
        if model_parameters is None
        else model_parameters
    )
    lambda_V0 = coefficients.lambda_V0_offset + coefficients.lambda_V0_slope * (
        E - coefficients.lambda_V0_reference_energy
    )
    lambda_W0 = float(
        coefficients.lambda_W0_offset
        + coefficients.lambda_W0_amplitude
        * np.tanh(
            (E - coefficients.lambda_W0_center_energy) / coefficients.lambda_W0_width
        )
    )
    lambda_V1 = coefficients.lambda_V1_value
    lambda_W1 = coefficients.lambda_W1_value
    return lambda_V0, lambda_W0, lambda_V1, lambda_W1


def make_jlmb_parameters(
    E: float,
    lambda_model_parameters: JLMBLambdaModelParameters | None = None,
    **overrides,
) -> JLMParameters:
    """Build :class:`JLMParameters` using JLMB renormalization factors."""
    lV0, lW0, lV1, lW1 = jlmb_lambda_factors(
        E, model_parameters=lambda_model_parameters
    )
    params = JLMParameters(
        t_R=T_R_DEFAULT,
        t_I=T_I_DEFAULT,
        lambda_V0=lV0,
        lambda_W0=lW0,
        lambda_V1=lV1,
        lambda_W1=lW1,
    )
    for name, value in overrides.items():
        if not hasattr(params, name):
            raise TypeError(f"Unknown JLM parameter override: {name}")
        setattr(params, name, value)
    return params


class JLMPotential:
    """Microscopic JLM/JLMB nucleon-nucleus optical potential."""

    def __init__(
        self,
        rho_n: Callable[[np.ndarray], np.ndarray],
        rho_p: Callable[[np.ndarray], np.ndarray],
        self_energy: Callable,
        parameters: JLMParameters | None = None,
    ) -> None:
        self.rho_n = rho_n
        self.rho_p = rho_p
        self.self_energy = self_energy
        self.params = parameters if parameters is not None else JLMParameters()

    def _local_rho_beta(self, r: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radii = np.atleast_1d(np.asarray(r, dtype=float))
        rho_n = np.asarray(self.rho_n(radii), dtype=float)
        rho_p = np.asarray(self.rho_p(radii), dtype=float)
        rho = rho_n + rho_p
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(rho > 1e-15, (rho_n - rho_p) / rho, 0.0)
        return rho, beta

    def _M_components(
        self, r: float | np.ndarray, E_eff: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rho, beta = self._local_rho_beta(r)
        V_full, W_full = self.self_energy(E_eff, rho, beta)
        V_iso, W_iso = self.self_energy(E_eff, rho, np.zeros_like(rho))
        V_iv = V_full - V_iso
        W_iv = W_full - W_iso
        return V_iso, W_iso, V_iv, W_iv

    def compute(
        self,
        R_grid: float | np.ndarray,
        E: float,
        projectile: str,
        Z: int | None = None,
        A: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the real and imaginary potential on ``R_grid``."""
        if projectile not in ("n", "p"):
            raise ValueError("projectile must be 'n' or 'p'.")

        params = self.params
        E_eff = float(E)
        if projectile == "p" and params.apply_coulomb_shift:
            if Z is None or A is None:
                warnings.warn(
                    "Coulomb shift requested but Z/A not provided; "
                    "using uncorrected lab energy.",
                    stacklevel=2,
                )
            else:
                E_eff = float(E) - coulomb_potential_center(int(Z), int(A))

        def V_pre(r: np.ndarray) -> np.ndarray:
            V_iso, _, V_iv, _ = self._M_components(r, E_eff)
            return params.lambda_V0 * V_iso + params.lambda_V1 * V_iv

        def W_pre(r: np.ndarray) -> np.ndarray:
            _, W_iso, _, W_iv = self._M_components(r, E_eff)
            return params.lambda_W0 * W_iso + params.lambda_W1 * W_iv

        V = gaussian_fold(
            R_grid, V_pre, params.t_R, r_max=params.r_max, n_quad=params.n_quad
        )
        W = gaussian_fold(
            R_grid, W_pre, params.t_I, r_max=params.r_max, n_quad=params.n_quad
        )
        return V, W

    def volume_integrals(
        self,
        E: float,
        projectile: str,
        A_target: int,
        Z: int | None = None,
        A: int | None = None,
        r_max: float = 15.0,
        n_R: int = 600,
    ) -> tuple[float, float]:
        """Return the conventional positive volume integrals per nucleon."""
        R = np.linspace(0.0, r_max, n_R)
        V, W = self.compute(R, E, projectile, Z=Z, A=A)
        J_V = -4.0 * np.pi * np.trapezoid(R**2 * V, R)
        J_W = -4.0 * np.pi * np.trapezoid(R**2 * W, R)
        return float(J_V / A_target), float(J_W / A_target)


__all__ = [
    "DEFAULT_JLMB_LAMBDA_MODEL_PARAMETERS",
    "DEFAULT_JLM_SELF_ENERGY_PARAMETERS",
    "E2",
    "HBARC",
    "JLMBLambdaModelParameters",
    "JLMParameters",
    "JLMPotential",
    "JLMSelfEnergy",
    "JLMSelfEnergyModelParameters",
    "JLMV0Parameters",
    "JLMV1Parameters",
    "JLMW0Parameters",
    "JLMW1Parameters",
    "M_NUCLEON",
    "T_I_DEFAULT",
    "T_R_DEFAULT",
    "coulomb_potential_center",
    "jlmb_lambda_factors",
    "make_jlmb_parameters",
]
