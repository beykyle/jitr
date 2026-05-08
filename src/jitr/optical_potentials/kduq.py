"""The Koning-Delaroche potential is a common optical potential for nuclear
scattering. It is provided here in simplified form specifically to address this
need.

See the [Koning-Delaroche
paper](https://www.sciencedirect.com/science/article/pii/S0375947402013210) for
details. Equation references are with respect to (w.r.t.) this paper.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .._types import ArrayOrScalar, PotentialArray
from ..data import data_dir
from ..reactions.reaction import Reaction
from ..utils.constants import WAVENUMBER_PION
from ..utils.kinematics import ChannelKinematics
from .omp import SingleChannelOpticalModel
from .potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)

NUM_POSTERIOR_SAMPLES = 416


def get_param_names(projectile: tuple[int, int]) -> list[str]:
    """
    Get the names of the parameters for the given projectile, in the
    order they are returned by the get_samples function.
    """
    return list(Global(projectile).params.keys())


def get_samples(projectile: tuple[int, int], posterior: str = "federal") -> np.ndarray:
    """
    Get the posterior samples for the given projectile (neutron or
    proton) from the KDUQ Federal or Democratic posteriors.

    See [Pruitt, et al., 2023]
    (https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602) for
    details on the KDUQ posteriors.

    :param projectile: tuple A tuple representing the projectile, with format (Ap,
                       Zp), where Ap is the mass number and Zp is the atomic number.
                       Must be either (1, 0) for neutron or (1, 1) for proton.
    :param posterior: str Which KDUQ posterior to return samples from. Must be
                      either "federal" or "democratic". Defaults to "federal".
    :returns: An array of shape (NUM_POSTERIOR_SAMPLES, num_params) containing the
              posterior samples for the given projectile, where num_params is the
              number of parameters in the Koning-Delaroche potential, and the
              parameters are ordered according to the order set by the
              get_param_names function.
    :rtype: np.ndarray"""
    if posterior == "federal":
        directory = "KDUQFederal"
    elif posterior == "democratic":
        directory = "KDUQDemocratic"
    else:
        raise ValueError("posterior must be either 'federal' or 'democratic'")

    return np.array(
        [
            list(
                Global(
                    projectile, data_dir / f"{directory}/{i}/parameters.json"
                ).params.values()
            )
            for i in range(NUM_POSTERIOR_SAMPLES)
        ]
    )


def Vv(E: float, v1: float, v2: float, v3: float, v4: float, Ef: float) -> float:
    r"""energy-dependent, volume-central strength - real term, Eq. (7)"""
    return v1 * (1 - v2 * (E - Ef) + v3 * (E - Ef) ** 2 - v4 * (E - Ef) ** 3)


def Wv(E: float, w1: float, w2: float, Ef: float) -> float:
    """energy-dependent, volume-central strength - imaginary term, Eq. (7)"""
    return w1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + w2**2)


def Wd(E: float, d1: float, d2: float, d3: float, Ef: float) -> float:
    """energy-dependent, surface-central strength - imaginary term (no real
    term), Eq. (7)
    """
    return d1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + d3**2) * np.exp(-d2 * (E - Ef))


def Vso(E: float, vso1: float, vso2: float, Ef: float) -> float:
    """energy-dependent, spin-orbit strength --- real term, Eq. (7)"""
    return vso1 * np.exp(-vso2 * (E - Ef))


def Wso(E: float, wso1: float, wso2: float, Ef: float) -> float:
    """energy-dependent, spin-orbit strength --- imaginary term, Eq. (7)"""
    return wso1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + wso2**2)


def delta_VC(
    E: float, Vcbar: float, v1: float, v2: float, v3: float, v4: float, Ef: float
) -> float:
    """energy dependent Coulomb correction term, Eq. 23"""
    return v1 * Vcbar * (v2 - 2 * v3 * (E - Ef) + 3 * v4 * (E - Ef) ** 2)


def central(
    r: float | np.ndarray,
    Vv: float,
    Rv: float,
    av: float,
    Wv: float,
    Rwv: float,
    awv: float,
    Wd: float,
    Rd: float,
    ad: float,
) -> PotentialArray:
    r"""
    Koning-Delaroche central terms at a given energy.

    This matches Eq. (7) in Koning and Delaroche (2003).

    :param r: float or np.ndarray The radius at which to evaluate the potential.
    :param Vv: float The real central depth.
    :param Rv: float The real central radius parameter.
    :param av: float The real central diffuseness parameter.
    :param Wv: float The imaginary volume depth.
    :param Rwv: float The imaginary volume radius parameter.
    :param awv: float The imaginary volume diffuseness parameter.
    :param Wd: float The imaginary surface depth.
    :param Rd: float The imaginary surface radius parameter.
    :param ad: float The imaginary surface diffuseness parameter."""
    result = (
        -Vv * woods_saxon_safe(r, Rv, av)
        - 1j * Wv * woods_saxon_safe(r, Rwv, awv)
        - 1j * (-4 * ad) * Wd * woods_saxon_prime_safe(r, Rd, ad)
    )
    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=np.complex128)
    return complex(result)


def spin_orbit(
    r: float | np.ndarray,
    Vso: float,
    Rso: float,
    aso: float,
    Wso: float,
    Rwso: float,
    awso: float,
) -> PotentialArray:
    r"""
    Koning-Delaroche spin-orbit terms at a given energy.

    This matches Eq. (7) in Koning and Delaroche (2003).

    :param r: float or np.ndarray The radius at which to evaluate the potential.
    :param Vso: float The real spin-orbit depth.
    :param Rso: float The real spin-orbit radius parameter.
    :param aso: float The real spin-orbit diffuseness parameter.
    :param Wso: float The imaginary spin-orbit depth.
    :param Rwso: float The imaginary spin-orbit radius parameter.
    :param awso: float The imaginary spin-orbit diffuseness parameter."""
    result = Vso / WAVENUMBER_PION**2 * thomas_safe(
        r, Rso, aso
    ) + 1j * Wso / WAVENUMBER_PION**2 * thomas_safe(r, Rwso, awso)
    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=np.complex128)
    return complex(result)


class Global:
    r"""Global Koning-Delaroche parameters"""

    def __init__(self, projectile: tuple, param_fpath: Path | None = None):
        r"""
        :param projectile: tuple A tuple representing the projectile, with
                           format (Ap, Zp), where Ap is the mass number and Zp
                           is the atomic number. Must be either (1, 0) for
                           neutron or (1, 1) for proton.
        :param param_fpath: Path, optional Path to the JSON file containing the
                            Koning-Delaroche parameters for the given
                            projectile, in the same format as the files in the
                            KDUQFederal and KDUQDemocratic directories. If None,
                            defaults to the file "KD_default.json" in the data
                            directory, which contains the default parameters for
                            the Koning-Delaroche potential as given in the
                            original paper by Koning and Delaroche (2003). Note
                            that the default parameters are not the same as the
                            mean of the KDUQ posteriors."""
        if param_fpath is None:
            param_fpath = Path(__file__).parent.resolve() / Path(
                "./../../data/KD_default.json"
            )

        if projectile == (1, 0):
            tag = "_n"
        elif projectile == (1, 1):
            tag = "_p"
        else:
            raise RuntimeError(
                "kduq.Global is defined only for neutron and proton projectiles"
            )

        self.projectile = projectile
        self.params = OrderedDict()

        # fermi energy
        if self.projectile == (1, 0):
            self.params["Ef_0"] = -11.2814
            self.params["Ef_A"] = 0.02646

        else:
            self.params["Ef_0"] = -8.4075
            self.params["Ef_A"] = 0.01378

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)

            if "KDHartreeFock" in data:
                # real central depth
                self.params["v1_0"] = data["KDHartreeFock"]["V1_0"]
                self.params["v1_asymm"] = data["KDHartreeFock"]["V1_asymm"]
                self.params["v1_A"] = data["KDHartreeFock"]["V1_A"]
                self.params["v2_0"] = data["KDHartreeFock"]["V2_0" + tag]
                self.params["v2_A"] = data["KDHartreeFock"]["V2_A" + tag]
                self.params["v3_0"] = data["KDHartreeFock"]["V3_0" + tag]
                self.params["v3_A"] = data["KDHartreeFock"]["V3_A" + tag]
                self.params["v4_0"] = data["KDHartreeFock"]["V4_0"]

                # real central form
                self.params["rv_0"] = data["KDHartreeFock"]["r_0"]
                self.params["rv_A"] = data["KDHartreeFock"]["r_A"]
                self.params["av_0"] = data["KDHartreeFock"]["a_0"]
                self.params["av_A"] = data["KDHartreeFock"]["a_A"]

                # imag volume depth
                self.params["w1_0"] = data["KDImagVolume"]["W1_0" + tag]
                self.params["w1_A"] = data["KDImagVolume"]["W1_A" + tag]
                self.params["w2_0"] = data["KDImagVolume"]["W2_0"]
                self.params["w2_A"] = data["KDImagVolume"]["W2_A"]

                # imag surface depth
                self.params["d1_0"] = data["KDImagSurface"]["D1_0"]
                self.params["d1_asymm"] = data["KDImagSurface"]["D1_asymm"]
                self.params["d2_0"] = data["KDImagSurface"]["D2_0"]
                self.params["d2_A"] = data["KDImagSurface"]["D2_A"]
                self.params["d2_A2"] = data["KDImagSurface"]["D2_A2"]
                self.params["d2_A3"] = data["KDImagSurface"]["D2_A3"]
                self.params["d3_0"] = data["KDImagSurface"]["D3_0"]

                # imag surface form
                self.params["rd_0"] = data["KDImagSurface"]["r_0"]
                self.params["rd_A"] = data["KDImagSurface"]["r_A"]
                self.params["ad_0"] = data["KDImagSurface"]["a_0" + tag]
                self.params["ad_A"] = data["KDImagSurface"]["a_A" + tag]

                # real spin orbit depth
                self.params["Vso1_0"] = data["KDRealSpinOrbit"]["V1_0"]
                self.params["Vso1_A"] = data["KDRealSpinOrbit"]["V1_A"]
                self.params["Vso2_0"] = data["KDRealSpinOrbit"]["V2_0"]

                # imag spin orbit form
                self.params["Wso1_0"] = data["KDImagSpinOrbit"]["W1_0"]
                self.params["Wso2_0"] = data["KDImagSpinOrbit"]["W2_0"]

                # spin orbit form
                self.params["rso_0"] = data["KDRealSpinOrbit"]["r_0"]
                self.params["rso_A"] = data["KDRealSpinOrbit"]["r_A"]
                self.params["aso_0"] = data["KDRealSpinOrbit"]["a_0"]

                # Coulomb
                if self.projectile == (1, 1):
                    self.params["rc_0"] = data["KDCoulomb"]["r_C_0"]
                    self.params["rc_A"] = data["KDCoulomb"]["r_C_A"]
                    self.params["rc_A2"] = data["KDCoulomb"]["r_C_A2"]

            elif "KDHartreeFock_V1_0" in data:
                # real central depth
                self.params["v1_0"] = data["KDHartreeFock_V1_0"]
                self.params["v1_asymm"] = data["KDHartreeFock_V1_asymm"]
                self.params["v1_A"] = data["KDHartreeFock_V1_A"]
                self.params["v2_0"] = data["KDHartreeFock_V2_0" + tag]
                self.params["v2_A"] = data["KDHartreeFock_V2_A" + tag]
                self.params["v3_0"] = data["KDHartreeFock_V3_0" + tag]
                self.params["v3_A"] = data["KDHartreeFock_V3_A" + tag]
                self.params["v4_0"] = data["KDHartreeFock_V4_0"]

                # real central form
                self.params["rv_0"] = data["KDHartreeFock_r_0"]
                self.params["rv_A"] = data["KDHartreeFock_r_A"]
                self.params["av_0"] = data["KDHartreeFock_a_0"]
                self.params["av_A"] = data["KDHartreeFock_a_A"]

                # imag volume depth
                self.params["w1_0"] = data["KDImagVolume_W1_0" + tag]
                self.params["w1_A"] = data["KDImagVolume_W1_A" + tag]
                self.params["w2_0"] = data["KDImagVolume_W2_0"]
                self.params["w2_A"] = data["KDImagVolume_W2_A"]

                # imag surface depth
                self.params["d1_0"] = data["KDImagSurface_D1_0"]
                self.params["d1_asymm"] = data["KDImagSurface_D1_asymm"]
                self.params["d2_0"] = data["KDImagSurface_D2_0"]
                self.params["d2_A"] = data["KDImagSurface_D2_A"]
                self.params["d2_A2"] = data["KDImagSurface_D2_A2"]
                self.params["d2_A3"] = data["KDImagSurface_D2_A3"]
                self.params["d3_0"] = data["KDImagSurface_D3_0"]

                # imag surface form
                self.params["rd_0"] = data["KDImagSurface_r_0"]
                self.params["rd_A"] = data["KDImagSurface_r_A"]
                self.params["ad_0"] = data["KDImagSurface_a_0" + tag]
                self.params["ad_A"] = data["KDImagSurface_a_A" + tag]

                # real spin orbit depth
                self.params["Vso1_0"] = data["KDRealSpinOrbit_V1_0"]
                self.params["Vso1_A"] = data["KDRealSpinOrbit_V1_A"]
                self.params["Vso2_0"] = data["KDRealSpinOrbit_V2_0"]

                # imag spin orbit form
                self.params["Wso1_0"] = data["KDImagSpinOrbit_W1_0"]
                self.params["Wso2_0"] = data["KDImagSpinOrbit_W2_0"]

                # spin orbit form
                self.params["rso_0"] = data["KDRealSpinOrbit_r_0"]
                self.params["rso_A"] = data["KDRealSpinOrbit_r_A"]
                self.params["aso_0"] = data["KDRealSpinOrbit_a_0"]

                # Coulomb
                if self.projectile == (1, 1):
                    self.params["rc_0"] = data["KDCoulomb_r_C_0"]
                    self.params["rc_A"] = data["KDCoulomb_r_C_A"]
                    self.params["rc_A2"] = data["KDCoulomb_r_C_A2"]
            else:
                raise ValueError("Unrecognized parameter file format for KDUQ!")

    def get_params(
        self, A: int, Z: int, Elab: float
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Return Koning-Delaroche central, spin-orbit, and Coulomb parameters."""
        return calculate_params(
            self.projectile, (A, Z), Elab, *list(self.params.values())
        )


def calculate_params(
    projectile: tuple,
    target: tuple,
    Elab: float,
    Ef_0: float,
    Ef_A: float,
    v1_0: float,
    v1_asymm: float,
    v1_A: float,
    v2_0: float,
    v2_A: float,
    v3_0: float,
    v3_A: float,
    v4_0: float,
    rv_0: float,
    rv_A: float,
    av_0: float,
    av_A: float,
    w1_0: float,
    w1_A: float,
    w2_0: float,
    w2_A: float,
    d1_0: float,
    d1_asymm: float,
    d2_0: float,
    d2_A: float,
    d2_A2: float,
    d2_A3: float,
    d3_0: float,
    rd_0: float,
    rd_A: float,
    ad_0: float,
    ad_A: float,
    Vso1_0: float,
    Vso1_A: float,
    Vso2_0: float,
    Wso1_0: float,
    Wso2_0: float,
    rso_0: float,
    rso_A: float,
    aso_0: float,
    rc_0: float = 0.0,
    rc_A: float = 0.0,
    rc_A2: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """
    Calculate the arguments for the central, spin_orbit, and
    coulomb_charged_sphere functions corresponding to the KDUQ potential
    for a given projectile, target, lab energy, and the KDUQ parameters.

    :param projectile: tuple A tuple representing the projectile, with format (Ap,
                       Zp), where Ap is the mass number and Zp is the atomic number.
    :param target: tuple A tuple representing the target, with format (A, Z), where
                   A is the mass number and Z is the atomic number.
    :param Elab: float The laboratory energy of the projectile in MeV.
    :param Ef_0: float Base Fermi energy.
    :param Ef_A: float Atomic mass number modifier for Fermi energy.
    :param v1_0, v1_asymm, ..., rc_A2: float Parameters for the Koning-Delaroche
                                       potential, including real and imaginary
                                       central depths, forms, spin-orbit terms, and
                                       Coulomb correction parameters. See Table V
                                       and the Appendix of [Pruitt, et al., 2023]
                                       (https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602)
                                       for details.
    :returns: central_params: tuple (vv, Rv, av, wv, Rwv, awv, wd, Rd, ad), where vv
              is the real central depth, Rv is the real central radius, av is the
              real central diffuseness, wv is the imaginary volume depth, Rwv is the
              imaginary volume radius, awv is the imaginary volume diffuseness, wd
              is the imaginary surface depth, Rd is the imaginary surface radius,
              and ad is the imaginary surface diffuseness.; spin_orbit_params: tuple
              (vso, Rso, aso, wso, Rwso, awso ), where vso is the real spin-orbit
              depth, Rso is the real spin-orbit radius, aso is the real spin-orbit
              diffuseness, wso is the imaginary spin-orbit depth, Rwso is the
              imaginary spin-orbit radius, and awso is the imaginary spin-orbit
              diffuseness. Note that the real and imaginary spin-orbit terms have
              the same form, so Rso = Rwso and aso = awso.; coulomb_params: tuple
              (Z*Zp, RC), where Z is the charge of the target, Zp is the charge of
              the projectile, and RC is the Coulomb radius.
    :rtype: tuple[central_params, spin_orbit_params, coulomb_params]"""

    A, Z = target
    Ap, Zp = projectile
    assert Ap == 1 and Zp in (0, 1)
    asym_factor = (A - 2 * Z) / (A)
    factor = (-1) ** (Zp + 1)  # -1 for neutron, +1 for proton
    asym_factor *= factor

    # fermi energy
    Ef = Ef_0 + Ef_A * A

    # real central depth
    v1 = v1_0 + v1_asymm * asym_factor - v1_A * A
    v2 = v2_0 + v2_A * A * factor
    v3 = v3_0 + v3_A * A * factor
    v4 = v4_0
    vv = Vv(Elab, v1, v2, v3, v4, Ef)

    # real central form
    rv = rv_0 - rv_A * A ** (-1.0 / 3.0)
    av = av_0 - av_A * A

    # imag volume depth
    w1 = w1_0 + w1_A * A
    w2 = w2_0 + w2_A * A
    wv = Wv(Elab, w1, w2, Ef)

    # imag volume form
    rwv = rv
    awv = av

    # imag surface depth
    d1 = d1_0 + d1_asymm * asym_factor
    d2 = d2_0 + d2_A / (1 + np.exp((A - d2_A3) / d2_A2))
    d3 = d3_0
    wd = Wd(Elab, d1, d2, d3, Ef)

    # imag surface form
    rd = rd_0 - rd_A * A ** (1.0 / 3.0)
    ad = ad_0 + ad_A * A * factor

    # real spin orbit depth
    vso1 = Vso1_0 + Vso1_A * A
    vso2 = Vso2_0
    vso = Vso(Elab, vso1, vso2, Ef)

    # real spin orbit form
    rso = rso_0 - rso_A * A ** (-1.0 / 3.0)
    aso = aso_0

    # imag spin orbit form
    wso1 = Wso1_0
    wso2 = Wso2_0
    wso = Wso(Elab, wso1, wso2, Ef)

    # imag spin orbit form
    rwso = rso
    awso = aso

    # Coulomb correction
    R_C = rv * A ** (1.0 / 3.0)
    if Zp == 1:
        # Coulomb radius
        rc0 = rc_0 + rc_A * A ** (-2.0 / 3.0) + rc_A2 * A ** (-5.0 / 3.0)
        R_C = rc0 * A ** (1.0 / 3.0)

        Vcbar = 1.73 / rc0 * Z * A ** (-1.0 / 3.0)
        Vc = delta_VC(Elab, Vcbar, v1, v2, v3, v4, Ef)
        vv += Vc

    coulomb_params = (Z * Zp, R_C)
    central_params = (
        vv,
        rv * A ** (1.0 / 3.0),
        av,
        wv,
        rwv * A ** (1.0 / 3.0),
        awv,
        wd,
        rd * A ** (1.0 / 3.0),
        ad,
    )
    spin_orbit_params = (
        vso,
        rso * A ** (1.0 / 3.0),
        aso,
        wso,
        rwso * A ** (1.0 / 3.0),
        awso,
    )

    return central_params, spin_orbit_params, coulomb_params


class KDUQ(SingleChannelOpticalModel):
    """
    Koning-Delaroche Uncertainty Quantification (KDUQ) optical
    potential model.
    """

    def __init__(self, projectile: tuple):
        super().__init__(params=get_param_names(projectile))
        self.projectile = projectile

    def evaluate(
        self,
        rgrid: float | np.ndarray,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        *params: float,
    ) -> tuple[PotentialArray, PotentialArray, ArrayOrScalar]:
        """Evaluate the KDUQ central, spin-orbit, and Coulomb terms."""
        central_params, spin_orbit_params, coulomb_params = calculate_params(
            tuple(reaction.projectile),  # type: ignore[arg-type]
            tuple(reaction.target),  # type: ignore[arg-type]
            kinematics.Elab,
            *params,
        )
        return (
            central(rgrid, *central_params),
            spin_orbit(rgrid, *spin_orbit_params),
            coulomb_charged_sphere(rgrid, *coulomb_params),
        )
