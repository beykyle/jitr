"""ILDA quadrature and folding helpers."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from ..utils.constants import ALPHA, HBARC

FloatArray: TypeAlias = NDArray[np.float64]
GridInput: TypeAlias = float | FloatArray


class ILDAFolder:
    """Quadrature helper for ILDA Coulomb and Gaussian folding.

    The folder owns a Gauss-Legendre grid on ``[0, r_max]`` and exposes
    interpolation, integration, Coulomb, and Gaussian-folding helpers on that
    grid.

    Attributes:
        r_q: Quadrature nodes on ``[0, r_max]`` in fm.
        w_q: Quadrature weights on ``[0, r_max]`` in fm.
    """

    e2 = ALPHA * HBARC

    def __init__(self, r_max: float = 20.0, n_quad: int = 200) -> None:
        """Initialize the quadrature grid.

        Args:
            r_max: Upper integration bound in fm.
            n_quad: Number of Gauss-Legendre points.

        Raises:
            ValueError: If ``r_max`` is non-positive or ``n_quad`` is too small.
        """

        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        if n_quad < 4:
            raise ValueError("n_quad must be at least 4.")
        self.r_max = float(r_max)
        self.n_quad = int(n_quad)
        nodes, weights = leggauss(self.n_quad)
        self.r_q = np.asarray(0.5 * self.r_max * (nodes + 1.0), dtype=float)
        self.w_q = np.asarray(0.5 * self.r_max * weights, dtype=float)

    def interp_to_quad(self, r_grid: ArrayLike, f_grid: ArrayLike) -> FloatArray:
        """Interpolate tabulated data onto the quadrature grid.

        Args:
            r_grid: Source radial grid in fm.
            f_grid: Tabulated values sampled on ``r_grid``.

        Returns:
            Values interpolated onto ``self.r_q``.
        """

        return np.interp(self.r_q, np.asarray(r_grid, float), np.asarray(f_grid, float))

    def integrate(self, f_q: ArrayLike) -> float:
        """Integrate a quantity sampled on the quadrature grid.

        Args:
            f_q: Function values sampled on ``self.r_q``.

        Returns:
            Approximation to ``∫_0^{r_max} f(r) dr``.
        """

        return float(np.sum(self.w_q * np.asarray(f_q, float)))

    def Z_from_density(self, rho_q: ArrayLike) -> float:
        """Compute the particle number implied by a spherical density.

        Args:
            rho_q: Density values sampled on ``self.r_q``.

        Returns:
            Value of ``4π ∫ r² ρ(r) dr``.
        """

        return 4.0 * np.pi * self.integrate(self.r_q**2 * np.asarray(rho_q, float))

    def rms_radius(self, rho_q: ArrayLike) -> float:
        """Compute the RMS radius of a spherical density.

        Args:
            rho_q: Density values sampled on ``self.r_q``.

        Returns:
            Root-mean-square radius in fm.

        Raises:
            ValueError: If the density integrates to a non-positive norm.
        """

        z_value = self.Z_from_density(rho_q)
        if z_value <= 0:
            raise ValueError("rms_radius: density integrates to ≤ 0.")
        weighted_radius = self.r_q**4 * np.asarray(rho_q, float)
        return float(np.sqrt(4.0 * np.pi * self.integrate(weighted_radius) / z_value))

    def V_coulomb(
        self,
        rho_p_q: GridInput,
        mode: str = "density",
        R_C: str | float | None = None,
        include_exchange: bool = False,
        r_out: GridInput | None = None,
    ) -> FloatArray:
        """Compute the Coulomb potential for a proton density.

        Args:
            rho_p_q: Proton density sampled on ``self.r_q``.
            mode: Coulomb model, either ``"density"`` or ``"uniform_sphere"``.
            R_C: Coulomb radius for the uniform-sphere mode, or ``"auto"`` to
                infer it from the RMS radius.
            include_exchange: Whether to add the Slater exchange correction.
            r_out: Optional output grid in fm. Defaults to ``self.r_q``.

        Returns:
            Coulomb potential values in MeV sampled on ``r_out``.

        Raises:
            ValueError: If ``mode`` is not recognized.
        """

        rho_array = np.asarray(rho_p_q, dtype=float)
        r_eval = self.r_q if r_out is None else np.asarray(r_out, float)

        if mode == "density":
            r_big = np.maximum(r_eval[..., None], self.r_q)
            v_c = (
                4.0
                * np.pi
                * self.e2
                * np.sum(self.w_q * self.r_q**2 * rho_array / r_big, axis=-1)
            )
        elif mode == "uniform_sphere":
            z_value = self.Z_from_density(rho_array)
            if R_C is None or (isinstance(R_C, str) and R_C == "auto"):
                radius_c = float(np.sqrt(5.0 / 3.0) * self.rms_radius(rho_array))
            else:
                radius_c = float(R_C)
            ze2 = z_value * self.e2
            inside = (ze2 / (2.0 * radius_c)) * (3.0 - (r_eval / radius_c) ** 2)
            outside = ze2 / np.where(r_eval > 0, r_eval, 1.0)
            v_c = np.where(r_eval < radius_c, inside, outside)
        else:
            raise ValueError(
                f"mode must be 'density' or 'uniform_sphere', got {mode!r}"
            )

        if include_exchange:
            v_x_q = (
                -self.e2
                * (3.0 / np.pi) ** (1.0 / 3.0)
                * np.cbrt(np.clip(rho_array, 0.0, None))
            )
            v_c = v_c + (v_x_q if r_out is None else np.interp(r_eval, self.r_q, v_x_q))
        return np.asarray(v_c, dtype=float)

    def gaussian_fold(
        self,
        U_q: GridInput,
        t: float,
        r_out: GridInput | None = None,
    ) -> FloatArray:
        """Fold a radial quantity with a three-dimensional Gaussian.

        Args:
            U_q: Input values sampled on ``self.r_q``.
            t: Gaussian width in fm.
            r_out: Optional output grid in fm. Defaults to ``self.r_q``.

        Returns:
            Folded values sampled on ``r_out``.

        Raises:
            ValueError: If ``t`` is non-positive.
        """

        if t <= 0:
            raise ValueError("t must be positive.")
        u_array = np.asarray(U_q, dtype=float)
        r_eval = self.r_q if r_out is None else np.asarray(r_out, float)
        sqrt_pi = np.sqrt(np.pi)

        kernel = np.exp(-(((r_eval[..., None] - self.r_q) / t) ** 2)) - np.exp(
            -(((r_eval[..., None] + self.r_q) / t) ** 2)
        )
        sum_int = np.sum(self.r_q * u_array * kernel * self.w_q, axis=-1)

        eps = 1e-10
        safe_r = np.where(r_eval > eps, r_eval, 1.0)
        u_general = sum_int / (sqrt_pi * t * safe_r)

        # Use the analytic R -> 0 limit instead of the general expression.
        u_zero = (4.0 / (sqrt_pi * t**3)) * np.sum(
            self.r_q**2 * u_array * np.exp(-((self.r_q / t) ** 2)) * self.w_q
        )
        return np.asarray(np.where(r_eval > eps, u_general, u_zero), dtype=float)
