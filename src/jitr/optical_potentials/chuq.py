"""The CHUQ potential is a global phenomenological nucleon-nucleus
optical potential

See [Pruitt, et al., 2023]
(https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602), or
the original CH89 paper [Varner, et al., 1991]
(https://www.sciencedirect.com/science/article/pii/037015739190039O?via%3Dihub)
for details. Equation references are with respect to the former paper.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..data import data_dir
from ..reactions.reaction import Reaction
from ..utils.constants import ALPHA, HBARC
from ..utils.kinematics import ChannelKinematics
from .omp import SingleChannelOpticalModel
from .potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)

NUM_POSTERIOR_SAMPLES = 208
NUM_PARAMS = 22


def get_param_names() -> list[str]:
    """
    Get the names of the parameters for the given projectile, in the
    order they are returned by the get_samples function.
    """
    return list(Global().params.keys())


def get_samples(posterior: str = "federal") -> np.ndarray:
    """
    Get the posterior samples for the given projectile (neutron or
    proton) from the CHUQ Federal or Democratic posteriors.

    See [Pruitt, et al., 2023]
    (https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602) for
    details on the CHUQ posteriors.

    Parameters:
    ----------
    posterior : str
        Which CHUQ posterior to return samples from. Must be either
        "federal" or "democratic". Defaults to "federal".

    Returns:
    -------
    np.ndarray
        An array of shape (NUM_POSTERIOR_SAMPLES, num_params) containing
        the posterior samples for the given projectile, where num_params
        is the number of parameters in the Chapel-Hill potential,
        and the parameters are ordered according to the order set by
        the get_param_names function.
    """
    if posterior == "federal":
        directory = "CHUQFederal"
    elif posterior == "democratic":
        directory = "CHUQDemocratic"
    else:
        raise ValueError("posterior must be either 'federal' or 'democratic'")

    return np.array(
        [
            list(Global(data_dir / f"{directory}/{i}/parameters.json").params.values())
            for i in range(NUM_POSTERIOR_SAMPLES)
        ]
    )


def central(
    r: float | np.ndarray,
    V: float,
    W: float,
    Wd: float,
    Rv: float,
    av: float,
    Rd: float,
    ad: float,
) -> complex:
    r"""
    Form of the central term of the CHUQ potential, given by Eqs. A7-8
    of [Pruitt, et al., 2023]

    Parameters:
    ----------
    r : float or np.ndarray
        The radius at which to evaluate the potential.
    V : float
        The depth of the real central potential.
    W : float
        The depth of the imaginary volume potential.
    Wd : float
        The depth of the imaginary surface potential.
    Rv : float
        The radius of the real central potential.
    av : float
        The diffuseness of the real central potential.
    Rd : float
        The radius of the imaginary potential.
    ad : float
        The diffuseness of the imaginary potential.
    """
    volume = V * woods_saxon_safe(r, Rv, av)
    imag_volume = 1j * W * woods_saxon_safe(r, Rd, ad)
    surface = -(4j * ad * Wd) * woods_saxon_prime_safe(r, Rd, ad)
    return -volume - imag_volume - surface


def spin_orbit(r: float | np.ndarray, Vso: float, Rso: float, aso: float) -> complex:
    """
    Form of the spin-orbit term of the CHUQ potential, given by Eqs.
    A7-8 of [Pruitt, et al., 2023]

    Parameters:
    ----------
    r : float or np.ndarray
        The radius at which to evaluate the potential.
    Vso : float
        The depth of the spin-orbit potential.
    Rso : float
        The radius of the spin-orbit potential.
    aso : float
        The diffuseness of the spin-orbit potential.
    """
    return 2 * Vso * thomas_safe(r, Rso, aso)


def calculate_params(
    projectile: tuple,
    target: tuple,
    Elab: float,
    V0: float = 52.9,
    Ve: float = -0.299,
    Vt: float = 13.1,
    r0: float = 1.25,
    r0_0: float = -0.225,
    a0: float = 0.69,
    Wv0: float = 7.8,
    Wve0: float = 35.0,
    Wvew: float = 16.0,
    rw: float = 1.33,
    rw_0: float = -0.42,
    aw: float = 0.69,
    Ws0: float = 10.0,
    Wst: float = 18.0,
    Wse0: float = 36.0,
    Wsew: float = 37,
    Vso: float = 5.9,
    rso: float = 1.34,
    rso_0: float = -1.2,
    aso: float = 0.63,
    rc: float = 1.24,
    rc_0: float = 0.12,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """
    Calculate the arguments for the central, spin_orbit, and
    coulomb_charged_sphere functions corresponding to the CHUQ potential
    for a given projectile, target, lab energy, and the CHUQ parameters.

    Parameters:
    ----------
    projectile : tuple
        A tuple containing the mass number and charge of the projectile.
    target : tuple
        A tuple containing the mass number and charge of the target.
    Elab : float
        The laboratory energy of the projectile in MeV.
    V0, Ve, ..., rc_0: float
        Parameters for the Chapel-Hill optical model potential.
        See Table V and the Appendix
        of [Pruitt, et al., 2023]
        (https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602)
        for details.

        See also Table 3. of the original CH89 paper [Varner, et al., 1991]
        (https://www.sciencedirect.com/science/article/pii/037015739190039O?via%3Dihub).

        Note: there is a typo in Eq. A4 of the CHUQ paper, there should be a
        plus/minus sign in front of Wst, not a plus. See Table 3 of the
        original CH89 paper for the correct sign.

    Returns:
    -------
        central_params : tuple
            A tuple containing the parameters for the central part
            of the potential, in the form (V0, Wv, Ws, R0, a0, Rw,
            aw), where V0 is the depth of the real central
            potential, Wv is the depth of the imaginary volume
            potential, Ws is the depth of the imaginary surface
            potential, R0 is the radius of the real central
            potential, a0 is the diffuseness of the real central
            potential, Rw is the radius of the imaginary potential,
            and aw is the diffuseness of the imaginary potential.
        spin_orbit_params : tuple
            A tuple containing the parameters for the spin-orbit
            part of the potential, in the form (Vso, Rso, aso),
            where Vso is the depth of the spin-orbit potential,
            Rso is the radius of the spin-orbit potential, and aso
            is the diffuseness of the spin-orbit potential.
        coulomb_params : tuple
            A tuple containing the parameters for the Coulomb
            potential,
            in the form (Z*Zp, RC), where Z is the charge of the
            target, Zp is the charge of the projectile, and RC is
            the Coulomb radius.
    """
    A, Z = target
    Ap, Zp = projectile
    is_proton = Zp == 1
    N = A - Z
    assert Ap == 1 and Zp in (0, 1)

    # Asymmetry factor
    alpha = (N - Z) / A
    sign = 1 if is_proton else -1
    asym_factor = sign * alpha

    # Radii (Eq. A6)
    R0 = r0 * A ** (1 / 3) + r0_0
    Rw = rw * A ** (1 / 3) + rw_0
    Rso = rso * A ** (1 / 3) + rso_0
    RC = rc * A ** (1 / 3) + rc_0

    # Coulomb correction
    Ec = coulomb_correction(A, Z, RC) if is_proton else 0.0
    delta_E = Elab - Ec

    # Real central depths (Eq. A4)
    V0 = V0 + Ve * delta_E + asym_factor * Vt

    # Imaginary depths (Eq. A4)
    Wv = Wv0 / (1 + np.exp((Wve0 - delta_E) / Wvew))
    Ws = (Ws0 + asym_factor * Wst) / (1 + np.exp((delta_E - Wse0) / Wsew))

    central_params = (V0, Wv, Ws, R0, a0, Rw, aw)
    spin_orbit_params = (Vso, Rso, aso)
    coulomb_params = (Z * Zp, RC)

    return central_params, spin_orbit_params, coulomb_params


def coulomb_correction(A: int, Z: int, RC: float) -> float:
    r"""
    Coulomb correction for proton energy
    """
    return 6.0 * Z * ALPHA * HBARC / (5 * RC)


class Global:
    r"""Global optical potential in CHUQ form."""

    def __init__(self, param_fpath: Path = None):
        r"""
        Parameters:
            param_fpath : path to json file encoding parameter values.
                Defaults to data/WLH_mean.json
        """
        if param_fpath is None:
            param_fpath = Path(__file__).parent.resolve() / Path(
                "./../../data/CH89_default.json"
            )

        self.params = OrderedDict()

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)

            if "CH89RealCentral" in data:
                self.params["V0"] = data["CH89RealCentral"]["V_0"]
                self.params["Ve"] = data["CH89RealCentral"]["V_e"]
                self.params["Vt"] = data["CH89RealCentral"]["V_t"]
                self.params["r0"] = data["CH89RealCentral"]["r_o"]
                self.params["r0_0"] = data["CH89RealCentral"]["r_o_0"]
                self.params["a0"] = data["CH89RealCentral"]["a_0"]

                self.params["Wv0"] = data["CH89ImagCentral"]["W_v0"]
                self.params["Wve0"] = data["CH89ImagCentral"]["W_ve0"]
                self.params["Wvew"] = data["CH89ImagCentral"]["W_vew"]
                self.params["rw"] = data["CH89ImagCentral"]["r_w"]
                self.params["rw_0"] = data["CH89ImagCentral"]["r_w0"]
                self.params["aw"] = data["CH89ImagCentral"]["a_w"]
                self.params["Ws0"] = data["CH89ImagCentral"]["W_s0"]
                self.params["Wst"] = data["CH89ImagCentral"]["W_st"]
                self.params["Wse0"] = data["CH89ImagCentral"]["W_se0"]
                self.params["Wsew"] = data["CH89ImagCentral"]["W_sew"]

                self.params["Vso"] = data["CH89SpinOrbit"]["V_so"]
                self.params["rso"] = data["CH89SpinOrbit"]["r_so"]
                self.params["rso_0"] = data["CH89SpinOrbit"]["r_so_0"]
                self.params["aso"] = data["CH89SpinOrbit"]["a_so"]

                self.params["rc"] = data["CH89Coulomb"]["r_c"]
                self.params["rc_0"] = data["CH89Coulomb"]["r_c_0"]

            elif "CH89RealCentral_V_0" in data:
                self.params["V0"] = data["CH89RealCentral_V_0"]
                self.params["Ve"] = data["CH89RealCentral_V_e"]
                self.params["Vt"] = data["CH89RealCentral_V_t"]
                self.params["r0"] = data["CH89RealCentral_r_o"]
                self.params["r0_0"] = data["CH89RealCentral_r_o_0"]
                self.params["a0"] = data["CH89RealCentral_a_0"]
                self.params["Wv0"] = data["CH89ImagCentral_W_v0"]
                self.params["Wve0"] = data["CH89ImagCentral_W_ve0"]
                self.params["Wvew"] = data["CH89ImagCentral_W_vew"]
                self.params["rw"] = data["CH89ImagCentral_r_w"]
                self.params["rw_0"] = data["CH89ImagCentral_r_w0"]
                self.params["aw"] = data["CH89ImagCentral_a_w"]
                self.params["Ws0"] = data["CH89ImagCentral_W_s0"]
                self.params["Wst"] = data["CH89ImagCentral_W_st"]
                self.params["Wse0"] = data["CH89ImagCentral_W_se0"]
                self.params["Wsew"] = data["CH89ImagCentral_W_sew"]
                self.params["Vso"] = data["CH89SpinOrbit_V_so"]
                self.params["rso"] = data["CH89SpinOrbit_r_so"]
                self.params["rso_0"] = data["CH89SpinOrbit_r_so_0"]
                self.params["aso"] = data["CH89SpinOrbit_a_so"]
                self.params["rc"] = data["CH89Coulomb_r_c"]
                self.params["rc_0"] = data["CH89Coulomb_r_c_0"]

            else:
                raise ValueError("Unrecognized parameter file format for WLH!")

    def get_params(
        self, projectile: tuple[int, int], target: tuple[int, int], Elab: float
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Return CHUQ central, spin-orbit, and Coulomb parameters."""
        return calculate_params(projectile, target, Elab, *list(self.params.values()))


class CHUQ(SingleChannelOpticalModel):
    """
    Chapel-Hill Uncertainty Quantification (CHUQ) optical
    potential model.

    Note that CH89 is Lane consistent, so the same parameters can be
    used for both neutron and proton projectiles.
    """

    def __init__(self):
        super().__init__(params=get_param_names())

    def evaluate(
        self,
        rgrid: float | np.ndarray,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        *params: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Evaluate the CHUQ central, spin-orbit, and Coulomb terms."""
        central_params, spin_orbit_params, coulomb_params = calculate_params(
            tuple(reaction.projectile),
            tuple(reaction.target),
            kinematics.Elab,
            *params,
        )
        return (
            central(rgrid, *central_params),
            spin_orbit(rgrid, *spin_orbit_params),
            coulomb_charged_sphere(rgrid, *coulomb_params),
        )
