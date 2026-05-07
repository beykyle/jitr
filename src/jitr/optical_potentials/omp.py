"""Base classes and helpers for single-channel optical potentials."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy.typing as npt

from ..reactions import reaction
from ..utils import kinematics
from ..utils.constants import WAVENUMBER_PION
from .potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)

PotentialCallable: TypeAlias = Callable[..., complex]
RadiusInput: TypeAlias = float | npt.NDArray


class SingleChannelOpticalModel:
    """Base class for local single-channel optical potentials.

    Subclasses map a high-level parameter vector to the central, spin-orbit, and
    Coulomb terms used by the elastic-scattering workspaces.
    """

    def __init__(
        self,
        params: list[str],
        interaction_central: PotentialCallable,
        interaction_spin_orbit: PotentialCallable,
        interaction_coulomb: PotentialCallable | None = None,
    ) -> None:
        """Store the parameter names and interaction callables."""
        self.params = params
        self.n_params = len(params)
        self.interaction_central = interaction_central
        self.interaction_spin_orbit = interaction_spin_orbit
        self.interaction_coulomb = interaction_coulomb

    def params_by_term(
        self,
        reaction: reaction.Reaction,
        kinematics: kinematics.ChannelKinematics,
        *params: float,
    ) -> tuple[tuple, tuple, tuple]:
        """Split model parameters into central, spin-orbit, and Coulomb terms.

        Args:
            reaction: Reaction defining the projectile and target.
            kinematics: Entrance-channel kinematics.
            *params: Model parameters in the order declared by ``self.params``.

        Returns:
            ``(central_params, spin_orbit_params, coulomb_params)``.
        """
        raise NotImplementedError(
            "Subclasses must return the parameters for the central, spin-orbit, "
            "and Coulomb terms for the requested reaction and kinematics."
        )


def central(
    r: float,
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
) -> complex:
    """Evaluate the default Woods-Saxon central potential."""
    return (
        -Vv * woods_saxon_safe(r, Rv, av)
        - 1j * Wv * woods_saxon_safe(r, Rw, aw)
        - (-4 * ad) * Vd * woods_saxon_prime_safe(r, Rd, ad)
        - 1j * (-4 * ad) * Wd * woods_saxon_prime_safe(r, Rd, ad)
    )


def spin_orbit(r: float, Vso: float, Wso: float, Rso: float, aso: float) -> complex:
    """Evaluate the default Thomas-form spin-orbit potential."""
    return (Vso + 1j * Wso) / WAVENUMBER_PION**2 * thomas_safe(r, Rso, aso)


class LocalOpticalPotential(SingleChannelOpticalModel):
    """Simple local optical potential with optional nucleus-nucleus radius scaling."""

    def __init__(self, scale_radii_by_At_and_Ap: bool = False) -> None:
        """Configure the default local optical-potential parameterization.

        Args:
            scale_radii_by_At_and_Ap: If ``True``, scale radius parameters by
                ``A_target^(1/3) + A_projectile^(1/3)``. Otherwise use the target
                mass only.
        """
        super().__init__(
            params=[
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
                "Vso",
                "Wso",
                "rso",
                "aso",
                "rc",
            ],
            interaction_central=central,
            interaction_spin_orbit=spin_orbit,
            interaction_coulomb=coulomb_charged_sphere,
        )
        self.scale_radii_by_At_and_Ap = scale_radii_by_At_and_Ap

    def params_by_term(
        self,
        reaction: reaction.Reaction,
        kinematics: kinematics.ChannelKinematics,
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
        rc: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Scale radius parameters for the requested reaction."""
        A, Z = reaction.target.A, reaction.target.Z
        Ap, Zp = reaction.projectile.A, reaction.projectile.Z
        if self.scale_radii_by_At_and_Ap:
            radius_factor = A ** (1 / 3) + Ap ** (1 / 3)
        else:
            radius_factor = A ** (1 / 3)

        Rv = rv * radius_factor
        Rw = rw * radius_factor
        Rd = rd * radius_factor
        Rso = rso * radius_factor
        RC = rc * radius_factor

        central_params = (Vv, Rv, av, Wv, Rw, aw, Wd, Vd, Rd, ad)
        spin_orbit_params = (Vso, Wso, Rso, aso)
        coulomb_params = (Z * Zp, RC)
        return central_params, spin_orbit_params, coulomb_params
