from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jitr.folding.folding import ILDAFolder
from jitr.folding.jlm import (
    lambda_v0,
    lambda_v1,
    lambda_vso,
    lambda_w0,
    lambda_w1,
    lambda_wso,
    potential_JLMB,
    spin_orbit_jlmb,
)
from jitr.optical_potentials.omp import LocalOpticalPotential
from jitr.reactions import ElasticReaction, Nucleus, Particle
from jitr.rmatrix import Solver
from jitr.utils.constants import AMU
from jitr.utils.density import density_table
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


def _build_jlm_elastic_case(
    ref: ReferenceCase,
    reaction: ElasticReaction,
    channel_kinematics,
    workspace,
    potential: dict,
) -> BuiltCase:
    radial_grid = workspace.radial_grid()
    variant = potential["variant"]
    if variant != "jlmb":
        raise ValueError(f"{ref.case_id}: unknown JLM variant {variant!r}")

    param = potential.get("parameterization", "talys")
    density_model = potential.get("density_model", "d1m")
    jlmmode = int(potential.get("jlmmode", 0))
    fw = potential["folding_widths_fm"]
    t_r = float(fw["real"])
    t_i = float(fw["imag"])

    energy = float(ref.metadata["kinematics"]["energy_MeV"])
    lv = float(lambda_v0(energy))
    lw = float(lambda_w0(energy, mode=jlmmode))
    lv1 = float(lambda_v1(energy))
    lw1 = float(lambda_w1(energy, mode=jlmmode))

    folder = ILDAFolder(r_max=15.0, n_quad=200)
    dt = density_table(reaction.target.A, reaction.target.Z, model=density_model)
    rho_n_q = folder.interp_to_quad(dt.radial_grid, dt.neutron_density_grid)
    rho_p_q = folder.interp_to_quad(dt.radial_grid, dt.proton_density_grid)

    charge_product = reaction.target.Z * reaction.projectile.Z
    if charge_product != 0:
        V_C_q = folder.V_coulomb(rho_p_q)
        V_C_out = folder.V_coulomb(rho_p_q, r_out=radial_grid)
    else:
        V_C_q = V_C_out = None

    central_re, central_im = potential_JLMB(
        folder,
        rho_n_q,
        rho_p_q,
        (reaction.projectile.A, reaction.projectile.Z),
        (reaction.target.A, reaction.target.Z),
        energy,
        V_C=V_C_q,
        parameterization=param,
        lambda_V=lv,
        lambda_W=lw,
        lambda_V1=lv1,
        lambda_W1=lw1,
        t_r=t_r,
        t_i=t_i,
        r_out=radial_grid,
    )

    central = np.asarray(central_re + 1j * central_im, dtype=np.complex128)
    coulomb = np.asarray(V_C_out, dtype=np.complex128) if charge_product != 0 else None

    # Spin-orbit: Scheerbaum Thomas form of the density (no Gaussian folding).
    vso = float(lambda_vso(energy))
    wso = float(lambda_wso(energy))
    SO_form = spin_orbit_jlmb(
        folder.r_q,
        rho_n_q,
        rho_p_q,
        (reaction.projectile.A, reaction.projectile.Z),
        r_out=radial_grid,
    )
    spin_orbit = np.asarray((vso + 1j * wso) * SO_form, dtype=np.complex128)

    return BuiltCase(
        workspace=workspace,
        xs_kwargs={
            "central_potential": central,
            "spin_orbit_potential": spin_orbit,
            "coulomb_potential": coulomb,
        },
    )


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
    kind = potential["kind"]
    if kind == "woods_saxon_local":
        return _build_ws_elastic_case(
            ref, reaction, channel_kinematics, workspace, potential
        )
    elif kind == "jlm":
        return _build_jlm_elastic_case(
            ref, reaction, channel_kinematics, workspace, potential
        )
    else:
        raise NotImplementedError(
            f"{ref.case_id} uses unsupported optical_potential.kind {kind!r}"
        )


def _build_ws_elastic_case(
    ref: ReferenceCase,
    reaction: ElasticReaction,
    channel_kinematics,
    workspace,
    potential: dict,
) -> BuiltCase:
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
