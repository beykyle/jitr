"""The Koning-Delaroche potential is a common optical potential for nuclear
scattering. It is provided here in simplified form specifically to address this
need.

See the [Koning-Delaroche
paper](https://www.sciencedirect.com/science/article/pii/S0375947402013210) for
details. Equation references are with respect to (w.r.t.) this paper.
"""

from pathlib import Path
from collections import OrderedDict

import json
import numpy as np

from ..utils.constants import WAVENUMBER_PION
from .potential_forms import (
    woods_saxon_safe,
    woods_saxon_prime_safe,
    thomas_safe,
    coulomb_charged_sphere,
)
from ..data import data_dir
from ..xs.elastic import DifferentialWorkspace


def get_samples_democratic(projectile: tuple):
    return np.array(
        [
            list(
                Global(
                    projectile, data_dir / f"KDUQDemocratic/{i}/parameters.json"
                ).params.values()
            )
            for i in range(416)
        ]
    )


def get_samples_federal(projectile: tuple):
    return np.array(
        [
            list(
                Global(
                    projectile, data_dir / f"KDUQFederal/{i}/parameters.json"
                ).params.values()
            )
            for i in range(416)
        ]
    )


def Vv(E, v1, v2, v3, v4, Ef):
    r"""energy-dependent, volume-central strength - real term, Eq. (7)"""
    return v1 * (1 - v2 * (E - Ef) + v3 * (E - Ef) ** 2 - v4 * (E - Ef) ** 3)


def Wv(E, w1, w2, Ef):
    """energy-dependent, volume-central strength - imaginary term, Eq. (7)"""
    return w1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + w2**2)


def Wd(E, d1, d2, d3, Ef):
    """energy-dependent, surface-central strength - imaginary term (no real
    term), Eq. (7)
    """
    return d1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + d3**2) * np.exp(-d2 * (E - Ef))


def Vso(E, vso1, vso2, Ef):
    """energy-dependent, spin-orbit strength --- real term, Eq. (7)"""
    return vso1 * np.exp(-vso2 * (E - Ef))


def Wso(E, wso1, wso2, Ef):
    """energy-dependent, spin-orbit strength --- imaginary term, Eq. (7)"""
    return wso1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + wso2**2)


def delta_VC(E, Vcbar, v1, v2, v3, v4, Ef):
    """energy dependent Coulomb correction term, Eq. 23"""
    return v1 * Vcbar * (v2 - 2 * v3 * (E - Ef) + 3 * v4 * (E - Ef) ** 2)


def central(r, vv, rv, av, wv, rwv, awv, wd, rd, ad):
    r"""simplified Koning-Delaroche without the spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.
    """
    return (
        -vv * woods_saxon_safe(r, rv, av)
        - 1j * wv * woods_saxon_safe(r, rwv, awv)
        - 1j * (-4 * ad) * wd * woods_saxon_prime_safe(r, rd, ad)
    )


def central_plus_coulomb(r, central_params, coulomb_params):
    nucl = central(r, *central_params)
    coul = coulomb_charged_sphere(r, *coulomb_params)
    return nucl + coul


def spin_orbit(r, vso, rso, aso, wso, rwso, awso):
    r"""simplified Koning-Delaroche spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.
    """
    return vso / WAVENUMBER_PION**2 * thomas_safe(
        r, rso, aso
    ) + 1j * wso / WAVENUMBER_PION**2 * thomas_safe(r, rwso, awso)


class Global:
    r"""Global Koning-Delaroche parameters"""

    def __init__(self, projectile: tuple, param_fpath: Path = None):
        r"""
        Parameters:
            projectile (tuple): (A,Z) must be neutron or proton
                (e.g. (1,0) or (1,1))
            param_fpath : path to json file encoding parameter values.
            Defaults to data/KD_default.json
        """
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

    def get_params(self, A, Z, Elab):
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
):
    """
    Calculate Koning-Delaroche global neutron-nucleus optical potential
    parameters for a given system.

    Parameters:
    ----------
    projectile : tuple
        A tuple representing the projectile, with format (Ap, Zp),
        where Ap is the mass number and Zp is the atomic number.
    target : tuple
        A tuple representing the target, with format (A, Z),
        where A is the mass number and Z is the atomic number.
    Elab : float
        The laboratory energy of the projectile in MeV.
    Ef_0 : float
        Base Fermi energy.
    Ef_A : float
        Atomic mass number modifier for Fermi energy.
    v1_0, v1_asymm, ..., rc_A2: float
        Parameters for the Koning-Delaroche potential, including
        real and imaginary central depths, forms, spin-orbit terms,
        and Coulomb correction parameters. See Table V and the Appendix
        of [Pruitt, et al., 2023]
        (https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602)
        for details.

    Returns:
    -------
    tuple
        A tuple containing the following:
        - coulomb_params: tuple
            A tuple with Coulomb parameters (Z * Zp, R_C).
        - central_params: tuple
            A tuple with central potential parameters
            (vv, rv * A**(1/3), av, wv, rwv * A**(1/3), awv, wd, rd * A**(1/3), ad)
        - spin_orbit_params: tuple
            A tuple with spin-orbit potential parameters
            (vso, rso * A**(1/3), aso, wso, rwso * A**(1/3), awso).
    """

    A, Z = target
    Ap, Zp = projectile
    assert Ap == 1 and (Zp == 1 or Zp == 0)
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

    return coulomb_params, central_params, spin_orbit_params


def calculate_diff_xs(
    workspace: DifferentialWorkspace,
    params: OrderedDict,
):
    rxn = workspace.reaction
    coulomb_params, central_params, spin_orbit_params = calculate_params(
        rxn.projectile, rxn.target, workspace.kinematics.Elab, params
    )

    return workspace.xs(
        central_plus_coulomb,
        spin_orbit,
        (central_params, coulomb_params),
        spin_orbit_params,
    )
