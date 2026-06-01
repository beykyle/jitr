from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jitr.optical_potentials.omp import LocalOpticalPotential
from jitr.reactions import ElasticReaction, Nucleus, Particle
from jitr.rmatrix import Solver
from jitr.utils.constants import AMU
from jitr.utils.kinematics import classical_kinematics, classical_kinematics_cm
from jitr.xs.elastic import DifferentialWorkspace

from ._readers import ReferenceCase


@dataclass(frozen=True)
class BuiltCase:
    """Concrete workspace and inputs for one regression case."""

    workspace: Any
    xs_kwargs: dict[str, np.ndarray | None]


def build_case(ref: ReferenceCase) -> BuiltCase:
    """Build the workspace and input arrays for a committed reference case."""
    if ref.observable_type != "elastic":
        raise NotImplementedError(
            f"{ref.case_id} uses unsupported observable_type {ref.observable_type!r}"
        )
    return _build_elastic_case(ref)


def _build_elastic_case(ref: ReferenceCase) -> BuiltCase:
    metadata = ref.metadata
    reaction_data = metadata["reaction"]
    mass_kwargs = metadata.get("mass_kwargs", {})
    mass_model = metadata.get("mass_model", "tabulated")
    reaction = ElasticReaction(
        _build_reaction_particle(reaction_data["target"], mass_model, mass_kwargs),
        _build_reaction_particle(reaction_data["projectile"], mass_model, mass_kwargs),
        mass_kwargs=mass_kwargs,
    )

    kinematics = metadata["kinematics"]
    energy = float(kinematics["energy_MeV"])
    frame = kinematics["frame"]
    relativistic = bool(kinematics.get("relativistic", True))
    if frame == "lab":
        if relativistic:
            channel_kinematics = reaction.kinematics(energy)
        else:
            channel_kinematics = classical_kinematics(
                reaction.target.m0,
                reaction.projectile.m0,
                energy,
                reaction.target.Z * reaction.projectile.Z,
            )
    elif frame == "cm":
        if relativistic:
            channel_kinematics = reaction.kinematics_cm(energy)
        else:
            channel_kinematics = classical_kinematics_cm(
                reaction.target.m0,
                reaction.projectile.m0,
                energy,
                reaction.target.Z * reaction.projectile.Z,
            )
    else:
        raise ValueError(f"{ref.case_id} uses unsupported frame {frame!r}")

    matching = metadata["matching"]
    solver = Solver(int(matching["nbasis"]))
    workspace = DifferentialWorkspace.build_from_system(
        reaction=reaction,
        kinematics=channel_kinematics,
        channel_radius_fm=float(matching["channel_radius_fm"]),
        solver=solver,
        lmax=int(matching["lmax"]),
        angles=ref.theta_cm_rad,
    )

    potential = metadata["optical_potential"]
    if potential["kind"] != "woods_saxon_local":
        raise NotImplementedError(
            f"{ref.case_id} uses unsupported optical_potential.kind "
            f"{potential['kind']!r}"
        )

    model = LocalOpticalPotential(
        scale_radii_by_At_and_Ap=bool(potential["scale_radii_by_At_and_Ap"])
    )
    radial_grid = workspace.radial_grid()
    central_data = potential["central"]
    spin_orbit_data = potential["spin_orbit"]
    charge_product = reaction.target.Z * reaction.projectile.Z
    coulomb_data = potential.get("coulomb")
    if charge_product != 0 and coulomb_data is None:
        raise ValueError(f"{ref.case_id} is missing optical_potential.coulomb metadata")
    coulomb_radius = float(coulomb_data["rC"]) if coulomb_data is not None else 1.0
    central, spin_orbit, coulomb = model.evaluate(
        radial_grid,
        reaction,
        channel_kinematics,
        float(central_data["Vv"]),
        float(central_data["rv"]),
        float(central_data["av"]),
        float(central_data["Wv"]),
        float(central_data["rw"]),
        float(central_data["aw"]),
        float(central_data["Wd"]),
        float(central_data["Vd"]),
        float(central_data["rd"]),
        float(central_data["ad"]),
        float(spin_orbit_data["Vso"]),
        float(spin_orbit_data["Wso"]),
        float(spin_orbit_data["rso"]),
        float(spin_orbit_data["aso"]),
        coulomb_radius,
    )
    return BuiltCase(
        workspace=workspace,
        xs_kwargs={
            "central_potential": central,
            "spin_orbit_potential": spin_orbit,
            "coulomb_potential": (
                np.asarray(coulomb, dtype=np.complex128)
                if charge_product != 0
                else None
            ),
        },
    )


def _build_reaction_particle(
    particle_data: dict[str, Any],
    mass_model: str,
    mass_kwargs: dict[str, str],
) -> Particle | tuple[int, int]:
    if mass_model == "tabulated":
        return (particle_data["A"], particle_data["Z"])
    if mass_model == "integer_amu":
        particle = Nucleus(
            int(particle_data["A"]),
            int(particle_data["Z"]),
            mass_kwargs=mass_kwargs,
        )
        particle.m0 = float(particle_data["A"]) * AMU
        return particle
    raise ValueError(f"Unsupported regression mass_model {mass_model!r}")
