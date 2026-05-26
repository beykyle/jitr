"""Generic nuclear-matter self-energy abstractions for folding models."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

RHO_SAT: float = 0.16


class NMSelfEnergy:
    """Lane-decomposed nuclear-matter self-energy model.

    Subclasses implement the four component functions ``V0``, ``W0``, ``V1``,
    and ``W1`` and inherit the common Lane-combination logic.
    """

    def __init__(self, projectile: str = "n") -> None:
        if projectile not in ("n", "p"):
            raise ValueError("projectile must be 'n' or 'p'.")
        self.projectile = projectile
        self.tau_z = +1 if projectile == "n" else -1

    def V0(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def W0(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def V1(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def W1(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self, E: float, rho: float | np.ndarray, beta: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rho_array, beta_array = np.broadcast_arrays(
            np.asarray(rho, dtype=float),
            np.asarray(beta, dtype=float),
        )
        V0 = self.V0(E, rho_array)
        W0 = self.W0(E, rho_array)
        V1 = self.V1(E, rho_array)
        W1 = self.W1(E, rho_array)
        V = V0 + self.tau_z * beta_array * V1
        W = W0 + self.tau_z * beta_array * W1
        return V, W


class TabulatedNMSelfEnergy(NMSelfEnergy):
    """Self-energy model backed by tabulated ``(E, rho)`` component grids."""

    def __init__(
        self,
        E_grid: float | np.ndarray,
        rho_grid: float | np.ndarray,
        V0_table: float | np.ndarray,
        W0_table: float | np.ndarray,
        V1_table: float | np.ndarray | None = None,
        W1_table: float | np.ndarray | None = None,
        projectile: str = "n",
        bounds_error: bool = False,
    ) -> None:
        super().__init__(projectile=projectile)

        self.E_grid = np.asarray(E_grid, dtype=float)
        self.rho_grid = np.asarray(rho_grid, dtype=float)
        if not np.all(np.diff(self.E_grid) > 0):
            raise ValueError("E_grid must be strictly increasing.")
        if not np.all(np.diff(self.rho_grid) > 0):
            raise ValueError("rho_grid must be strictly increasing.")

        expected_shape = (self.E_grid.size, self.rho_grid.size)
        self.V0_table = np.asarray(V0_table, dtype=float)
        self.W0_table = np.asarray(W0_table, dtype=float)
        if self.V0_table.shape != expected_shape:
            raise ValueError(
                f"V0_table must have shape (nE, nrho) = {expected_shape}, "
                f"got {self.V0_table.shape}."
            )
        if self.W0_table.shape != expected_shape:
            raise ValueError(
                f"W0_table must have shape (nE, nrho) = {expected_shape}, "
                f"got {self.W0_table.shape}."
            )

        if V1_table is None:
            V1_table = np.zeros(expected_shape)
        if W1_table is None:
            W1_table = np.zeros(expected_shape)
        self.V1_table = np.asarray(V1_table, dtype=float)
        self.W1_table = np.asarray(W1_table, dtype=float)
        if (
            self.V1_table.shape != expected_shape
            or self.W1_table.shape != expected_shape
        ):
            raise ValueError("V1_table / W1_table must match (nE, nrho).")

        self.bounds_error = bounds_error
        kw = dict(bounds_error=bounds_error, fill_value=None)
        self._V0 = RegularGridInterpolator(
            (self.E_grid, self.rho_grid), self.V0_table, **kw
        )
        self._W0 = RegularGridInterpolator(
            (self.E_grid, self.rho_grid), self.W0_table, **kw
        )
        self._V1 = RegularGridInterpolator(
            (self.E_grid, self.rho_grid), self.V1_table, **kw
        )
        self._W1 = RegularGridInterpolator(
            (self.E_grid, self.rho_grid), self.W1_table, **kw
        )

    @classmethod
    def from_analytical(
        cls,
        analytical: NMSelfEnergy,
        E_grid: float | np.ndarray | None = None,
        rho_grid: float | np.ndarray | None = None,
        projectile: str | None = None,
    ) -> TabulatedNMSelfEnergy:
        """Sample an analytical Lane-decomposed model onto a regular grid."""
        energies = (
            np.linspace(1.0, 200.0, 60)
            if E_grid is None
            else np.asarray(E_grid, dtype=float)
        )
        densities = (
            np.linspace(0.0, 0.20, 41)
            if rho_grid is None
            else np.asarray(rho_grid, dtype=float)
        )
        return cls(
            E_grid=energies,
            rho_grid=densities,
            V0_table=np.array([analytical.V0(E, densities) for E in energies]),
            W0_table=np.array([analytical.W0(E, densities) for E in energies]),
            V1_table=np.array([analytical.V1(E, densities) for E in energies]),
            W1_table=np.array([analytical.W1(E, densities) for E in energies]),
            projectile=projectile if projectile is not None else analytical.projectile,
        )

    @classmethod
    def from_text_file(
        cls,
        path: str,
        projectile: str = "n",
    ) -> TabulatedNMSelfEnergy:
        """Load a tabulated self-energy from the text layout written below."""
        with open(path, encoding="utf-8") as fh:
            lines = [line for line in fh if not line.lstrip().startswith("#")]
        tokens = " ".join(lines).split()
        idx = 0
        nE = int(tokens[idx])
        idx += 1
        nrho = int(tokens[idx])
        idx += 1
        E_grid = np.array(tokens[idx : idx + nE], dtype=float)
        idx += nE
        rho_grid = np.array(tokens[idx : idx + nrho], dtype=float)
        idx += nrho
        ntot = nE * nrho
        tables = []
        for _ in range(4):
            tables.append(
                np.array(tokens[idx : idx + ntot], dtype=float).reshape(nE, nrho)
            )
            idx += ntot
        V0_table, W0_table, V1_table, W1_table = tables
        return cls(
            E_grid=E_grid,
            rho_grid=rho_grid,
            V0_table=V0_table,
            W0_table=W0_table,
            V1_table=V1_table,
            W1_table=W1_table,
            projectile=projectile,
        )

    def save_text_file(self, path: str) -> None:
        """Persist the tabulated components to a text file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# Tabulated nuclear-matter self-energy\n")
            fh.write("# Format: nE nrho, then E_grid, rho_grid,\n")
            fh.write("#         then V0, W0, V1, W1 each as (nE, nrho).\n")
            fh.write(f"{self.E_grid.size} {self.rho_grid.size}\n")
            np.savetxt(fh, self.E_grid[None, :])
            np.savetxt(fh, self.rho_grid[None, :])
            for tab in (
                self.V0_table,
                self.W0_table,
                self.V1_table,
                self.W1_table,
            ):
                np.savetxt(fh, tab)

    def _eval(
        self,
        interp: RegularGridInterpolator,
        E: float,
        rho: float | np.ndarray,
    ) -> np.ndarray:
        rho_array = np.asarray(rho, dtype=float)
        E_array = np.full(rho_array.shape, float(E))
        if not self.bounds_error:
            E_array = np.clip(E_array, self.E_grid[0], self.E_grid[-1])
            rho_array = np.clip(rho_array, self.rho_grid[0], self.rho_grid[-1])
        points = np.stack([E_array.ravel(), rho_array.ravel()], axis=-1)
        return interp(points).reshape(rho_array.shape)

    def V0(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        return self._eval(self._V0, E, rho)

    def W0(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        return self._eval(self._W0, E, rho)

    def V1(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        return self._eval(self._V1, E, rho)

    def W1(self, E: float, rho: float | np.ndarray) -> np.ndarray:
        return self._eval(self._W1, E, rho)


__all__ = ["NMSelfEnergy", "RHO_SAT", "TabulatedNMSelfEnergy"]
