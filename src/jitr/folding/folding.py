"""ILDA folding utilies"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..utils.constants import ALPHA, HBARC


class ILDAFolder:
    """Gauss-Legendre quadrature on [0, r_max] with ILDA folding utilities.

    Owns the GL nodes and weights and provides three services that all live on
    (or fold to/from) that grid: density interpolation, Coulomb potential, and
    Gaussian convolution for ILDA. Methods default to returning values on r_q
    so outputs can be chained; pass r_out=... to land on an arbitrary grid.

    Attributes
    ----------
    r_q : (n_quad,) ndarray   GL nodes on [0, r_max], in fm
    w_q : (n_quad,) ndarray   GL weights on [0, r_max], in fm
    """

    e2 = ALPHA * HBARC  # MeV·fm

    def __init__(self, r_max: float = 20.0, n_quad: int = 200):
        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        if n_quad < 4:
            raise ValueError("n_quad must be at least 4.")
        self.r_max, self.n_quad = float(r_max), int(n_quad)
        nodes, weights = leggauss(self.n_quad)
        self.r_q = 0.5 * self.r_max * (nodes + 1.0)
        self.w_q = 0.5 * self.r_max * weights

    def interp_to_quad(self, r_grid, f_grid):
        """Linear interpolation f(r_grid) → f(r_q). Ensure r_grid covers [0, r_max]."""
        return np.interp(self.r_q, np.asarray(r_grid, float), np.asarray(f_grid, float))

    def integrate(self, f_q):
        """∫_0^{r_max} f(r) dr ≈ Σ_q w_q · f(r_q)."""
        return float(np.sum(self.w_q * np.asarray(f_q, float)))

    def Z_from_density(self, rho_q):
        """4π ∫ r² ρ(r) dr — equals Z for a proton density (sanity check)."""
        return 4.0 * np.pi * self.integrate(self.r_q**2 * rho_q)

    def rms_radius(self, rho_q):
        """sqrt(<r²>) of a spherical density on the quadrature grid (fm)."""
        Z = self.Z_from_density(rho_q)
        if Z <= 0:
            raise ValueError("rms_radius: density integrates to ≤ 0.")
        return np.sqrt(4.0 * np.pi * self.integrate(self.r_q**4 * rho_q) / Z)

    def V_coulomb(
        self, rho_p_q, mode="density", R_C=None, include_exchange=False, r_out=None
    ):
        """V_C(r) [MeV] from a proton density on r_q.

        Parameters
        ----------
        rho_p_q : (n_quad,) array — proton density on the quadrature grid.
        mode : {'density', 'uniform_sphere'}
            'density'        — full integration  V_C(r) = 4π e² ∫ r'² ρ/r_> dr'.
            'uniform_sphere' — analytic uniform-sphere form. Z derived from
                               rho_p_q for self-consistency.
        R_C : 'auto' | float | None
            For 'uniform_sphere' only.
              'auto'/None → R_C = √(5/3) · √<r²>_p (rms-equivalent sphere)
              float       → user-supplied value (fm).
        include_exchange : bool
            Add Slater LDA exchange  V_C^x = −e² (3/π)^(1/3) ρ_p^(1/3).
        r_out : array_like or None — target grid; None returns on r_q.
        """
        r_eval = self.r_q if r_out is None else np.asarray(r_out, float)

        if mode == "density":
            r_big = np.maximum(r_eval[..., None], self.r_q)
            V_C = (
                4.0
                * np.pi
                * self.e2
                * np.sum(self.w_q * self.r_q**2 * rho_p_q / r_big, axis=-1)
            )
        elif mode == "uniform_sphere":
            Z = self.Z_from_density(rho_p_q)
            if R_C is None or (isinstance(R_C, str) and R_C == "auto"):
                R_C = np.sqrt(5.0 / 3.0) * self.rms_radius(rho_p_q)
            else:
                R_C = float(R_C)
            Ze2 = Z * self.e2
            inside = (Ze2 / (2.0 * R_C)) * (3.0 - (r_eval / R_C) ** 2)
            outside = Ze2 / np.where(r_eval > 0, r_eval, 1.0)
            V_C = np.where(r_eval < R_C, inside, outside)
        else:
            raise ValueError(
                f"mode must be 'density' or 'uniform_sphere', got {mode!r}"
            )

        if include_exchange:
            V_x_q = (
                -self.e2
                * (3.0 / np.pi) ** (1.0 / 3.0)
                * np.cbrt(np.clip(rho_p_q, 0.0, None))
            )
            V_C = V_C + (V_x_q if r_out is None else np.interp(r_eval, self.r_q, V_x_q))
        return V_C

    def gaussian_fold(self, U_q, t, r_out=None):
        """Convolve a spherical U(r) with a 3D Gaussian of width t (MeV·fm³)."""
        if t <= 0:
            raise ValueError("t must be positive.")
        r_eval = self.r_q if r_out is None else np.asarray(r_out, float)
        sqrt_pi = np.sqrt(np.pi)

        kernel = np.exp(-(((r_eval[..., None] - self.r_q) / t) ** 2)) - np.exp(
            -(((r_eval[..., None] + self.r_q) / t) ** 2)
        )
        sum_int = np.sum(self.r_q * U_q * kernel * self.w_q, axis=-1)

        eps = 1e-10
        safe_R = np.where(r_eval > eps, r_eval, 1.0)
        U_general = sum_int / (sqrt_pi * t * safe_R)

        # R=0 limit: U(0) = (4/(√π t³)) ∫ r² U(r) exp(-r²/t²) dr
        U_zero = (4.0 / (sqrt_pi * t**3)) * np.sum(
            self.r_q**2 * U_q * np.exp(-((self.r_q / t) ** 2)) * self.w_q
        )
        return np.where(r_eval > eps, U_general, U_zero)
