from typing import Callable

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
    """
    A simple optical model base class that uses a central and spin-orbit
    interaction.

    For use with jitr.xs.elastic.DifferentialWorkspace, which requires
    as input functions U(r;params_{central}) and U_{so}(r;params_{so})
    for the central and spin-orbit interactions, respectively. These are
    provided as callables that take a radius and a tuple of parameters
    and return a complex potential.

    In general, the parameters for the central and spin-orbit
    interactions may depend on the reaction and kinematics, so the class
    includes a method central_and_spin_orbit_params that can be
    implemented by subclasses to calculate the appropriate parameters
    for U and U_{so}.

    This can be as complicated as a full global optical model in which a
    full set of model parameters determines (the `params` arg in
    `centralcentral_and_spin_orbit_params`) determines the A,Z,E,...
    dependence of params_{central} and params_{so}, or it can be
    something more simple, like a fixed set of parameters applicable to
    a specific reaction or set of reactions.
    """

    def __init__(
        self,
        params: list[str],
        interaction_central: Callable[[float, tuple], complex],
        interaction_spin_orbit: Callable[[float, tuple], complex],
        interaction_coulomb: Callable[[float, tuple], complex] = None,
    ):
        self.params = params
        self.n_params = len(params)
        self.interaction_central = interaction_central
        self.interaction_spin_orbit = interaction_spin_orbit
        self.interaction_coulomb = interaction_coulomb

    def params_by_term(
        self,
        reaction: reaction.Reaction,
        kinematics: kinematics.ChannelKinematics,
        *params,
    ) -> tuple:
        """
        Calculates the arguments to self.interaction_central,
        self.interaction_spin_orbit, and self.interaction_coulomb if
        applicable, for a given reaction and kinematics, based on the
        model parameters passed in `params`.

        This method should be implemented by subclasses to
        return the appropriate parameters for the interactions based on
        the specific model being used.

        Parameters:
        ----------
        reaction : jitr.reactions.reactions.Reaction
            The reaction for which the parameters are being calculated.
        kinematics : jitr.utils.kinematics.ChannelKinematics
            The kinematics of the reaction channel.
        params : tuple
            The parameters of the model.

        Returns:
        -------
        central_params : tuple
            The parameters to be passed to self.interaction_central.
        spin_orbit_params : tuple
            The parameters to be passed to self.interaction_spin_orbit.
        coulomb_params : tuple
            The parameters to be passed to self.interaction_coulomb, or
            an empty tuple if self.interaction_coulomb is None.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to return "
            "the parameters for the central and spin-orbit interactions "
            "based on the reaction and kinematics."
        )
        central_params = ()
        spin_orbit_params = ()
        coulomb_params = ()
        return central_params, spin_orbit_params, coulomb_params


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
):
    """
    Standard Woods-Saxon based central potential. Note that, by
    convention, the capital R parameters are radii in fm, and are not
    multiplied by A^(1/3). The diffuseness parameters are in fm, and the
    strengths are in MeV.

    Parameters:
    ----------
    r : float
        The radius at which to evaluate the potential.
    Vv : float
        The real volume strength of the potential.
    Rv : float
        The radius parameter for the real volume term.
    av : float
        The diffuseness parameter for the real volume term.
    Wv : float
        The imaginary volume strength of the potential.
    Rw : float
        The radius parameter for the imaginary volume term.
    aw : float
        The diffuseness parameter for the imaginary volume term.
    Wd : float
        The imaginary surface strength of the potential.
    Vd : float
        The real surface strength of the potential.
    Rd : float
        The radius parameter for the surface terms.
    ad : float
        The diffuseness parameter for the surface terms.
    """
    return (
        -Vv * woods_saxon_safe(r, Rv, av)
        - 1j * Wv * woods_saxon_safe(r, Rw, aw)
        - (-4 * ad) * Vd * woods_saxon_prime_safe(r, Rd, ad)
        - 1j * (-4 * ad) * Wd * woods_saxon_prime_safe(r, Rd, ad)
    )


def spin_orbit(r: float, Vso: float, Wso: float, Rso: float, aso: float) -> complex:
    """
    Standard Thomas form for the spin-orbit potential. Note that, by
    convention, the capital R parameters are radii in fm, and are not
    multiplied by A^(1/3). The diffuseness parameters are in fm, and the
    strengths are in MeV.

    Parameters:
    ----------
    r : float
        The radius at which to evaluate the potential.
    Vso : float
        The real strength of the spin-orbit potential.
    Wso : float
        The imaginary strength of the spin-orbit potential.
    Rso : float
        The radius parameter for the spin-orbit potential.
    aso : float
        The diffuseness parameter for the spin-orbit potential.
    """
    return (Vso + 1j * Wso) / WAVENUMBER_PION**2 * thomas_safe(r, Rso, aso)


class LocalOpticalPotential(SingleChannelOpticalModel):
    """
    A simple local optical potential which can describe elastic
    nucleon-nucleus or nucleus-nucleus scattering using a central and
    spin-orbit interaction, without any explicit energy or mass
    dependence in the potential parameters aside from scaling reduced
    radii by A^(1/3) of the target or by (A_target^(1/3) +
    A_projectile^(1/3)).
    """

    def __init__(self, scale_radii_by_At_and_Ap=False):
        """
        Parameters:
        ----------
        scale_radii_by_At_and_Ap : bool
            Whether to scale the radius parameters by A_target^(1/3) of
            the target or by (A_target^(1/3) + A_projectile^(1/3)).

            Default is False, which means to scale the radius parameters
            by A_target^(1/3), which is the common choice for
            nucleon-nucleus optical potentials. Setting this to True
            will scale the radius parameters by (A_target^(1/3) +
            A_projectile^(1/3)), which is a common choice for optical
            potentials that are intended to be used for nucleus-nucleus
            scattering.
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
    ) -> tuple:
        """
        A simple implementation of the central_and_spin_orbit_params
        method for a local optical potential. The only dependence on
        the reacting system is through the A^(1/3) scaling of the radius
        parameters and the Z*Zp dependence of the Coulomb potential. The
        strengths and diffuseness parameters are fixed and do not depend
        on the reaction or kinematics.

        Parameters:
        ----------
        reaction : jitr.reactions.reactions.Reaction
            The reaction for which the parameters are being calculated.
        kinematics : jitr.utils.kinematics.ChannelKinematics
            The kinematics of the reaction channel.
        Vv : float
            The real volume strength of the potential.
        rv : float
            The reduced radius parameter for the real volume term
        av : float
            The diffuseness parameter for the real volume term.
        Wv : float
            The imaginary volume strength of the potential.
        rw : float
            The reduced radius parameter for the imaginary volume term
        aw : float
            The diffuseness parameter for the imaginary volume term.
        Wd : float
            The imaginary surface strength of the potential.
        Vd : float
            The real surface strength of the potential.
        rd : float
            The reduced radius parameter for the surface terms
        ad : float
            The diffuseness parameter for the surface terms.
        Vso : float
            The real strength of the spin-orbit potential.
        Wso : float
            The imaginary strength of the spin-orbit potential.
        rso : float
            The reduced radius parameter for the spin-orbit potential
        aso : float
            The diffuseness parameter for the spin-orbit potential.
        rc : float
            The reduced radius parameter for the Coulomb potential


        Returns:
        --------
        central_params : tuple
            (Vv, Rv, av, Wv, Rw, aw, Wd, Vd, Rd, ad)
        spin_orbit_params : tuple
            (Vso, Wso, Rso, aso)
        coulomb_params : tuple
            (Z*Zp, RC)
        """
        A, Z = reaction.target.A, reaction.target.Z
        Ap, Zp = reaction.projectile.A, reaction.projectile.Z
        if self.scale_radii_by_At_and_Ap:
            A_factor = A ** (1 / 3) + Ap ** (1 / 3)
        else:
            A_factor = A ** (1 / 3)
        Rv = rv * A_factor
        Rw = rw * A_factor
        Rd = rd * A_factor
        Rso = rso * A_factor
        RC = rc * A_factor

        central_params = (Vv, Rv, av, Wv, Rw, aw, Wd, Vd, Rd, ad)
        coulomb_params = (Z * Zp, RC)
        spin_orbit_params = (Vso, Wso, Rso, aso)
        return central_params, spin_orbit_params, coulomb_params
