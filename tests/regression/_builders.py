from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
from jitr.reactions import ElasticReaction, Nucleus, Particle, Reaction
from jitr.rmatrix import Solver
from jitr.utils.constants import AMU
from jitr.utils.density import density_table
from jitr.utils.kinematics import classical_kinematics, classical_kinematics_cm
from jitr.xs.elastic import DifferentialWorkspace
from jitr.xs.quasielastic_pn import Workspace as QuasielasticPnWorkspace

from ._readers import ReferenceCase


@dataclass(frozen=True)
class BuiltCase:
    """Concrete workspace and inputs for one regression case."""

    workspace: Any
    xs_kwargs: dict[str, np.ndarray | None]
    # elastic workspaces return an ElasticXS; the (p,n) workspace returns dsdo itself
    extract_dsdo: Callable[[Any], np.ndarray] = field(
        default=lambda result: result.dsdo
    )

    def dsdo(self) -> np.ndarray:
        """Return the differential cross section in mb/sr for this case."""
        return self.extract_dsdo(self.workspace.xs(**self.xs_kwargs))


def build_case(ref: ReferenceCase) -> BuiltCase:
    """Build the workspace and input arrays for a committed reference case."""
    if ref.observable_type == "elastic":
        return _build_elastic_case(ref)
    if ref.observable_type == "quasielastic_pn":
        return _build_quasielastic_pn_case(ref)
    raise NotImplementedError(
        f"{ref.case_id} uses unsupported observable_type {ref.observable_type!r}"
    )


def _evaluate_local_potential(
    reaction_model,
    channel_kinematics,
    radial_grid: np.ndarray,
    block: dict[str, Any],
    coulomb_radius: float,
    scale_radii_by_At_and_Ap: bool,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Evaluate one KD-style local potential block on the quadrature grid.

    ``block`` carries the 13 KD02 parameters as named in the deck
    (``V rv av W rw aw Wd rvd avd Vso Wso rvso avso``); the surface real depth
    ``Vd`` is zero for these decks.
    """
    model = LocalOpticalPotential(
        scale_radii_by_At_and_Ap=scale_radii_by_At_and_Ap,
    )
    return model.evaluate(
        radial_grid,
        reaction_model,
        channel_kinematics,
        float(block["V"]),
        float(block["rv"]),
        float(block["av"]),
        float(block["W"]),
        float(block["rw"]),
        float(block["aw"]),
        float(block["Wd"]),
        0.0,
        float(block["rvd"]),
        float(block["avd"]),
        float(block["Vso"]),
        float(block["Wso"]),
        float(block["rvso"]),
        float(block["avso"]),
        coulomb_radius,
    )


def _build_quasielastic_pn_case(ref: ReferenceCase) -> BuiltCase:
    """Build a DWBA (p,n) case against a Frescox charge-exchange deck.

    Both channels use the deck's own lab energies and integer-amu masses, so
    jitR and Frescox see identical kinematics. ``U1_central`` is left to jitR's
    default isovector difference, which equals the form factor the deck reads;
    ``U1_spin_orbit`` is zero because a Frescox ``KIND=1`` form factor is
    central only.
    """
    metadata = ref.metadata
    reaction_data = metadata["reaction"]
    mass_kwargs = metadata.get("mass_kwargs", {})
    mass_model = metadata.get("mass_model", "tabulated")
    particles = {
        name: _build_reaction_particle(reaction_data[name], mass_model, mass_kwargs)
        for name in ("target", "projectile", "product", "residual")
    }
    reaction = Reaction(
        particles["target"],
        particles["projectile"],
        particles["product"],
        particles["residual"],
        mass_kwargs=mass_kwargs,
    )
    exit_reaction = Reaction(
        particles["residual"],
        particles["product"],
        process="El",
        mass_kwargs=mass_kwargs,
    )

    kinematics = metadata["kinematics"]
    frame = kinematics["frame"]
    if frame != "lab" or bool(kinematics.get("relativistic", True)):
        raise NotImplementedError(
            f"{ref.case_id}: (p,n) cases expect non-relativistic lab kinematics"
        )
    kinematics_entrance = classical_kinematics(
        reaction.target.m0,
        reaction.projectile.m0,
        float(kinematics["energy_MeV"]),
        reaction.target.Z * reaction.projectile.Z,
    )
    kinematics_exit = classical_kinematics(
        exit_reaction.target.m0,
        exit_reaction.projectile.m0,
        float(kinematics["exit_energy_MeV"]),
        exit_reaction.target.Z * exit_reaction.projectile.Z,
    )

    matching = metadata["matching"]
    workspace = QuasielasticPnWorkspace(
        reaction=reaction,
        kinematics_entrance=kinematics_entrance,
        kinematics_exit=kinematics_exit,
        solver=Solver(int(matching["nbasis"])),
        angles=ref.theta_cm_rad,
        lmax=int(matching["lmax"]),
        channel_radius_fm=float(matching["channel_radius_fm"]),
        tmatrix_abs_tol=0.0,
    )

    potential = metadata["optical_potential"]
    kind = potential["kind"]
    if kind != "woods_saxon_local_pn":
        raise NotImplementedError(
            f"{ref.case_id} uses unsupported optical_potential.kind {kind!r}"
        )
    scale_radii = bool(potential["scale_radii_by_At_and_Ap"])
    radial_grid = workspace.radial_grid()
    coulomb_radius = float(potential["coulomb"]["rC"])
    proton_central, proton_spin_orbit, proton_coulomb = _evaluate_local_potential(
        reaction,
        kinematics_entrance,
        radial_grid,
        potential["proton"],
        coulomb_radius,
        scale_radii,
    )
    neutron_central, neutron_spin_orbit, _ = _evaluate_local_potential(
        exit_reaction,
        kinematics_exit,
        radial_grid,
        potential["neutron"],
        coulomb_radius,
        scale_radii,
    )

    transition = metadata["transition_potential"]
    if transition["central"] != "default_isovector_difference":
        raise NotImplementedError(
            f"{ref.case_id}: unsupported transition_potential.central "
            f"{transition['central']!r}"
        )
    if transition["spin_orbit"] != "zero":
        raise NotImplementedError(
            f"{ref.case_id}: unsupported transition_potential.spin_orbit "
            f"{transition['spin_orbit']!r}"
        )
    return BuiltCase(
        workspace=workspace,
        xs_kwargs={
            "U_p_coulomb": np.asarray(proton_coulomb, dtype=np.complex128),
            "U_p_central": np.asarray(proton_central, dtype=np.complex128),
            "U_p_spin_orbit": np.asarray(proton_spin_orbit, dtype=np.complex128),
            "U_n_central": np.asarray(neutron_central, dtype=np.complex128),
            "U_n_spin_orbit": np.asarray(neutron_spin_orbit, dtype=np.complex128),
            "U1_spin_orbit": np.zeros_like(radial_grid, dtype=np.complex128),
        },
        extract_dsdo=lambda result: np.asarray(result, dtype=np.float64),
    )


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
