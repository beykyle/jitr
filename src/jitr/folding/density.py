"""General radial density utilities for folding calculations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class TwoParameterFermiDensity:
    """Callable two-parameter Fermi density."""

    R: float
    a: float
    rho0: float | None = None
    N: float | None = None

    def __post_init__(self) -> None:
        if self.rho0 is None and self.N is None:
            raise ValueError("Must provide either `rho0` or `N`.")

    def __call__(self, r: float | np.ndarray) -> np.ndarray:
        return two_parameter_fermi(r, R=self.R, a=self.a, rho0=self.rho0, N=self.N)


def two_parameter_fermi(
    r: float | np.ndarray,
    R: float,
    a: float,
    rho0: float | None = None,
    N: float | None = None,
) -> np.ndarray:
    """Return a two-parameter Fermi density."""
    r_array = np.asarray(r, dtype=float)
    shape = 1.0 / (1.0 + np.exp((r_array - R) / a))
    if rho0 is None:
        if N is None:
            raise ValueError("Must provide either `rho0` or `N`.")
        rmax = max(R + 20.0 * a, 25.0)
        rg = np.linspace(0.0, rmax, 8001)
        fg = 1.0 / (1.0 + np.exp((rg - R) / a))
        norm = 4.0 * np.pi * np.trapezoid(rg**2 * fg, rg)
        normalization = float(N / norm)
    else:
        normalization = float(rho0)
    return normalization * shape


def density_from_array(
    r_array: float | np.ndarray,
    rho_array: float | np.ndarray,
    clip_negative: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a tabulated radial density as a callable."""
    radii = np.asarray(r_array, dtype=float)
    densities = np.asarray(rho_array, dtype=float)
    if radii.shape != densities.shape:
        raise ValueError("r_array and rho_array must have the same shape.")
    if not np.all(np.diff(radii) > 0):
        raise ValueError("r_array must be strictly increasing.")
    if radii[0] > 1e-12:
        radii = np.concatenate(([0.0], radii))
        densities = np.concatenate(([densities[0]], densities))

    interp = PchipInterpolator(radii, densities, extrapolate=False)
    r_max = float(radii[-1])

    def rho(r: np.ndarray) -> np.ndarray:
        r_values = np.asarray(r, dtype=float)
        out = np.zeros_like(r_values)
        mask = (r_values >= 0.0) & (r_values <= r_max)
        if mask.any():
            out[mask] = interp(r_values[mask])
        if clip_negative:
            out = np.where(out < 0.0, 0.0, out)
        return out

    rho.r_max = r_max  # type: ignore[attr-defined]
    return rho


__all__ = ["TwoParameterFermiDensity", "density_from_array", "two_parameter_fermi"]
