"""Base classes and helpers for single-channel optical potentials."""

from __future__ import annotations

import numpy as np

from .._types import ArrayOrScalar, PotentialArray
from ..reactions import reaction
from ..utils import kinematics
from ..utils.constants import WAVENUMBER_PION
from .potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)


class SingleChannelOpticalModel:
    """Base class for local single-channel optical potentials."""

    def __init__(self, params: list[str]) -> None:
        self.params = params
        self.n_params = len(params)

    def evaluate(
        self,
        rgrid: ArrayOrScalar,
        reaction: reaction.Reaction,
        kinematics: kinematics.ChannelKinematics,
        *params: float,
    ) -> tuple[PotentialArray, PotentialArray, PotentialArray | ArrayOrScalar]:
        """Evaluate central, spin-orbit, and Coulomb terms on ``rgrid``."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(
        self,
        rgrid: ArrayOrScalar,
        reaction: reaction.Reaction,
        kinematics: kinematics.ChannelKinematics,
        *params: float,
    ) -> tuple[PotentialArray, PotentialArray, PotentialArray | ArrayOrScalar]:
        return self.evaluate(rgrid, reaction, kinematics, *params)


def _as_potential_array(result: complex | np.ndarray) -> PotentialArray:
    """Return a potential result with repository-standard complex dtypes."""

    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=np.complex128)
    return complex(result)


def central(
    r: ArrayOrScalar,
    Vv: float,
    Rv: float,
    av: float,
    Wv: float,
    Rw: float,
    aw: float,
    Wd: float,
    Vd: float,
    Rd: float,
    ad: float,
) -> PotentialArray:
    """Evaluate the default Woods-Saxon central potential."""
    result = (
        -Vv * woods_saxon_safe(r, Rv, av)
        - 1j * Wv * woods_saxon_safe(r, Rw, aw)
        - (-4 * ad) * Vd * woods_saxon_prime_safe(r, Rd, ad)
        - 1j * (-4 * ad) * Wd * woods_saxon_prime_safe(r, Rd, ad)
    )
    return _as_potential_array(result)


def spin_orbit(
    r: ArrayOrScalar, Vso: float, Wso: float, Rso: float, aso: float
) -> PotentialArray:
    """Evaluate the default Thomas-form spin-orbit potential."""
    result = (Vso + 1j * Wso) / WAVENUMBER_PION**2 * thomas_safe(r, Rso, aso)
    return _as_potential_array(result)


class LocalOpticalPotential(SingleChannelOpticalModel):
    """Simple local optical potential with optional nucleus-nucleus radius scaling."""

    def __init__(self, scale_radii_by_At_and_Ap: bool = False) -> None:
        self.central_params = [
            "Vv",
            "rv",
            "av",
            "Wv",
            "rw",
            "aw",
            "Wd",
            "Vd",
            "rd",
            "ad",
        ]
        self.spin_orbit_params = ["Vso", "Wso", "rso", "aso"]
        self.coulomb_params = ["rC"]
        super().__init__(
            params=self.central_params + self.spin_orbit_params + self.coulomb_params
        )
        self.scale_radii_by_At_and_Ap = scale_radii_by_At_and_Ap

    def radius_factor(self, reaction_model: reaction.Reaction) -> float:
        """Return the radius scaling factor for the current reaction."""
        target_mass = reaction_model.target.A
        projectile_mass = reaction_model.projectile.A
        if self.scale_radii_by_At_and_Ap:
            return target_mass ** (1 / 3) + projectile_mass ** (1 / 3)
        return target_mass ** (1 / 3)

    def evaluate(  # type: ignore[override]
        self,
        rgrid: ArrayOrScalar,
        reaction_model: reaction.Reaction,
        kinematics_model: kinematics.ChannelKinematics,  # noqa: ARG002
        Vv: float,
        rv: float,
        av: float,
        Wv: float,
        rw: float,
        aw: float,
        Wd: float,
        Vd: float,
        rd: float,
        ad: float,
        Vso: float,
        Wso: float,
        rso: float,
        aso: float,
        rC: float,
    ) -> tuple[PotentialArray, PotentialArray, PotentialArray | ArrayOrScalar]:
        """Evaluate the local optical-potential terms on ``rgrid``."""
        radius_factor = self.radius_factor(reaction_model)
        Rv = rv * radius_factor
        Rw = rw * radius_factor
        Rd = rd * radius_factor
        Rso = rso * radius_factor
        RC = rC * radius_factor
        zz = reaction_model.target.Z * reaction_model.projectile.Z

        central_term = central(rgrid, Vv, Rv, av, Wv, Rw, aw, Wd, Vd, Rd, ad)
        spin_orbit_term = spin_orbit(rgrid, Vso, Wso, Rso, aso)
        coulomb_term = coulomb_charged_sphere(rgrid, zz, RC)
        return central_term, spin_orbit_term, coulomb_term
