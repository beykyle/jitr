"""The Whitehead-Lim-Holt potential is a global mcroscopic nucleon-nucleus
optical potential

See the [Whitehead, et al., 2021]
(https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.182502)
for details. Equation references are with respect to (w.r.t.) this paper.
"""

from collections import OrderedDict
from pathlib import Path
import json

import numpy as np

from ..data import data_dir
from ..utils.constants import WAVENUMBER_PION
from .potential_forms import (
    woods_saxon_safe,
    woods_saxon_prime_safe,
    coulomb_charged_sphere,
    thomas_safe,
)
from ..xs.elastic import DifferentialWorkspace


def get_samples(projectile: tuple):
    return np.array(
        [
            list(
                Global(
                    projectile, data_dir / f"WLHSamples/{i}/parameters.json"
                ).params.values()
            )
            for i in range(1000)
        ]
    )


def spin_orbit(r, uso, rso, aso):
    r"""WLH spin-orbit terms"""
    return (uso / WAVENUMBER_PION**2) * thomas_safe(r, rso, aso)


def central(r, uv, rv, av, uw, rw, aw, ud, rd, ad):
    r"""WLH without the spin-orbit term"""
    return (
        -uv * woods_saxon_safe(r, rv, av)
        - 1j * uw * woods_saxon_safe(r, rw, aw)
        - 1j * (-4 * ad) * ud * woods_saxon_prime_safe(r, rd, ad)
    )


def central_plus_coulomb(r, central_params, coulomb_params):
    nucl = central(r, *central_params)
    coul = coulomb_charged_sphere(r, *coulomb_params)
    return nucl + coul


class Global:
    r"""Global optical potential in WLH form."""

    def __init__(self, projectile: tuple, param_fpath: Path = None):
        r"""
        Parameters:
            projectile : neutron or proton?
            param_fpath : path to json file encoding parameter values.
                Defaults to data/WLH_mean.json
        """
        if param_fpath is None:
            param_fpath = Path(__file__).parent.resolve() / Path(
                "./../../data/WLH_mean.json"
            )

        if projectile == (1, 0):
            tag = "_n"
        elif projectile == (1, 1):
            tag = "_p"
        else:
            raise RuntimeError(
                "wlh.Global is defined only for neutron and proton projectiles"
            )

        self.params = OrderedDict()
        self.projectile = projectile

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)

            if "WLHReal" in data:
                self.params["uv0"] = data["WLHReal"]["V0" + tag]
                self.params["uv1"] = data["WLHReal"]["V1" + tag]
                self.params["uv2"] = data["WLHReal"]["V2" + tag]
                self.params["uv3"] = data["WLHReal"]["V3" + tag]
                self.params["uv4"] = data["WLHReal"]["V4" + tag]
                self.params["uv5"] = data["WLHReal"]["V5" + tag]
                self.params["uv6"] = data["WLHReal"]["V6" + tag]
                self.params["rv0"] = data["WLHReal"]["r0" + tag]
                self.params["rv1"] = data["WLHReal"]["r1" + tag]
                self.params["rv2"] = data["WLHReal"]["r2" + tag]
                self.params["rv3"] = data["WLHReal"]["r3" + tag]
                self.params["av0"] = data["WLHReal"]["a0" + tag]
                self.params["av1"] = data["WLHReal"]["a1" + tag]
                self.params["av2"] = data["WLHReal"]["a2" + tag]
                self.params["av3"] = data["WLHReal"]["a3" + tag]
                self.params["av4"] = data["WLHReal"]["a4" + tag]
                self.params["uw0"] = data["WLHImagVolume"]["W0" + tag]
                self.params["uw1"] = data["WLHImagVolume"]["W1" + tag]
                self.params["uw2"] = data["WLHImagVolume"]["W2" + tag]
                self.params["uw3"] = data["WLHImagVolume"]["W3" + tag]
                self.params["uw4"] = data["WLHImagVolume"]["W4" + tag]
                self.params["rw0"] = data["WLHImagVolume"]["r0" + tag]
                self.params["rw1"] = data["WLHImagVolume"]["r1" + tag]
                self.params["rw2"] = data["WLHImagVolume"]["r2" + tag]
                self.params["rw3"] = data["WLHImagVolume"]["r3" + tag]
                self.params["rw4"] = data["WLHImagVolume"]["r4" + tag]
                self.params["rw5"] = data["WLHImagVolume"]["r5" + tag]
                self.params["aw0"] = data["WLHImagVolume"]["a0" + tag]
                self.params["aw1"] = data["WLHImagVolume"]["a1" + tag]
                self.params["aw2"] = data["WLHImagVolume"]["a2" + tag]
                self.params["aw3"] = data["WLHImagVolume"]["a3" + tag]
                self.params["aw4"] = data["WLHImagVolume"]["a4" + tag]
                self.params["ud0"] = data["WLHImagSurface"]["W0" + tag]
                self.params["ud1"] = data["WLHImagSurface"]["W1" + tag]
                self.params["ud3"] = data["WLHImagSurface"]["W2" + tag]
                self.params["ud4"] = data["WLHImagSurface"]["W3" + tag]
                self.params["rd0"] = data["WLHImagSurface"]["r0" + tag]
                self.params["rd1"] = data["WLHImagSurface"]["r1" + tag]
                self.params["rd2"] = data["WLHImagSurface"]["r2" + tag]
                self.params["ad0"] = data["WLHImagSurface"]["a0" + tag]
                self.params["uso0"] = data["WLHRealSpinOrbit"]["V0" + tag]
                self.params["uso1"] = data["WLHRealSpinOrbit"]["V1" + tag]
                self.params["rso0"] = data["WLHRealSpinOrbit"]["r0" + tag]
                self.params["rso1"] = data["WLHRealSpinOrbit"]["r1" + tag]
                self.params["aso0"] = data["WLHRealSpinOrbit"]["a0" + tag]
                self.params["aso1"] = data["WLHRealSpinOrbit"]["a1" + tag]
            elif f"WLHRealSpinOrbit_a1{tag}" in data:
                self.params["uv0"] = data["WLHReal_V0" + tag]
                self.params["uv1"] = data["WLHReal_V1" + tag]
                self.params["uv2"] = data["WLHReal_V2" + tag]
                self.params["uv3"] = data["WLHReal_V3" + tag]
                self.params["uv4"] = data["WLHReal_V4" + tag]
                self.params["uv5"] = data["WLHReal_V5" + tag]
                self.params["uv6"] = data["WLHReal_V6" + tag]
                self.params["rv0"] = data["WLHReal_r0" + tag]
                self.params["rv1"] = data["WLHReal_r1" + tag]
                self.params["rv2"] = data["WLHReal_r2" + tag]
                self.params["rv3"] = data["WLHReal_r3" + tag]
                self.params["av0"] = data["WLHReal_a0" + tag]
                self.params["av1"] = data["WLHReal_a1" + tag]
                self.params["av2"] = data["WLHReal_a2" + tag]
                self.params["av3"] = data["WLHReal_a3" + tag]
                self.params["av4"] = data["WLHReal_a4" + tag]
                self.params["uw0"] = data["WLHImagVolume_W0" + tag]
                self.params["uw1"] = data["WLHImagVolume_W1" + tag]
                self.params["uw2"] = data["WLHImagVolume_W2" + tag]
                self.params["uw3"] = data["WLHImagVolume_W3" + tag]
                self.params["uw4"] = data["WLHImagVolume_W4" + tag]
                self.params["rw0"] = data["WLHImagVolume_r0" + tag]
                self.params["rw1"] = data["WLHImagVolume_r1" + tag]
                self.params["rw2"] = data["WLHImagVolume_r2" + tag]
                self.params["rw3"] = data["WLHImagVolume_r3" + tag]
                self.params["rw4"] = data["WLHImagVolume_r4" + tag]
                self.params["rw5"] = data["WLHImagVolume_r5" + tag]
                self.params["aw0"] = data["WLHImagVolume_a0" + tag]
                self.params["aw1"] = data["WLHImagVolume_a1" + tag]
                self.params["aw2"] = data["WLHImagVolume_a2" + tag]
                self.params["aw3"] = data["WLHImagVolume_a3" + tag]
                self.params["aw4"] = data["WLHImagVolume_a4" + tag]
                self.params["ud0"] = data["WLHImagSurface_W0" + tag]
                self.params["ud1"] = data["WLHImagSurface_W1" + tag]
                self.params["ud3"] = data["WLHImagSurface_W2" + tag]
                self.params["ud4"] = data["WLHImagSurface_W3" + tag]
                self.params["rd0"] = data["WLHImagSurface_r0" + tag]
                self.params["rd1"] = data["WLHImagSurface_r1" + tag]
                self.params["rd2"] = data["WLHImagSurface_r2" + tag]
                self.params["ad0"] = data["WLHImagSurface_a0" + tag]
                self.params["uso0"] = data["WLHRealSpinOrbit_V0" + tag]
                self.params["uso1"] = data["WLHRealSpinOrbit_V1" + tag]
                self.params["rso0"] = data["WLHRealSpinOrbit_r0" + tag]
                self.params["rso1"] = data["WLHRealSpinOrbit_r1" + tag]
                self.params["aso0"] = data["WLHRealSpinOrbit_a0" + tag]
                self.params["aso1"] = data["WLHRealSpinOrbit_a1" + tag]
            else:
                raise ValueError("Unrecognized parameter file format for WLH!")

            assert set(self.params) == set(PARAMS)

    def get_params(self, A, Z, Elab):
        # fermi energy
        return calculate_params(
            self.projectile, (A, Z), Elab, *list(self.params.values())
        )


def calculate_params(
    projectile: tuple,
    target: tuple,
    Elab: float,
    uv0: float,
    uv1: float,
    uv2: float,
    uv3: float,
    uv4: float,
    uv5: float,
    uv6: float,
    rv0: float,
    rv1: float,
    rv2: float,
    rv3: float,
    av0: float,
    av1: float,
    av2: float,
    av3: float,
    av4: float,
    uw0: float,
    uw1: float,
    uw2: float,
    uw3: float,
    uw4: float,
    rw0: float,
    rw1: float,
    rw2: float,
    rw3: float,
    rw4: float,
    rw5: float,
    aw0: float,
    aw1: float,
    aw2: float,
    aw3: float,
    aw4: float,
    ud0: float,
    ud1: float,
    ud3: float,
    ud4: float,
    rd0: float,
    rd1: float,
    rd2: float,
    ad0: float,
    uso0: float,
    uso1: float,
    rso0: float,
    rso1: float,
    aso0: float,
    aso1: float,
):
    """
    Calculates WLH parameters for a given system

    Parameters:
        projectile : tuple
            (A, Z) of the projectile
        target : tuple
            (A, Z) of the target
        Elab : float
            Laboratory energy in MeV
        uv0, uv1, ..., aso1 : float
            Parameters of the WLH potential. See
            [Whitehead et al., 2021]
            (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.182502)
            for details.
    """

    A, Z = target
    Ap, Zp = projectile
    assert Ap == 1 and (Zp == 1 or Zp == 0)
    asym_factor = (A - 2 * Z) / (A)
    factor = (-1) ** (Zp + 1)  # -1 for neutron, +1 for proton

    uv = (
        uv0
        - uv1 * Elab
        + uv2 * Elab**2
        + uv3 * Elab**3
        + factor * (uv4 - uv5 * Elab + uv6 * Elab**2) * asym_factor
    )
    rv = rv0 - rv1 * A ** (-1.0 / 3) - rv2 * Elab + rv3 * Elab**2
    av = (
        av0
        - factor * av1 * Elab
        - av2 * Elab**2
        - (av3 - av4 * asym_factor) * asym_factor
    )

    uw = uw0 + uw1 * Elab - uw2 * Elab**2 + (factor * uw3 - uw4 * Elab) * asym_factor
    rw = rw0 + (rw1 + rw2 * A) / (rw3 + A + rw4 * Elab) + rw5 * Elab**2
    aw = aw0 - (aw1 * Elab) / (-aw2 - Elab) + (aw3 - aw4 * Elab) * asym_factor

    if (projectile == (1, 0) and Elab < 40) or (
        projectile == (1, 1) and Elab < 20 and A > 100
    ):
        ud = ud0 - ud1 * Elab - (ud3 - ud4 * Elab) * asym_factor
    else:
        ud = 0

    rd = rd0 - rd2 * Elab - rd1 * A ** (-1.0 / 3)
    ad = ad0

    uso = uso0 - uso1 * A
    rso = rso0 - rso1 * A ** (-1.0 / 3.0)
    aso = aso0 - aso1 * A

    R_C = rv * A ** (1.0 / 3.0)
    coulomb_params = (Z * Zp, R_C)
    central_params = (
        uv,
        rv * A ** (1.0 / 3.0),
        av,
        uw,
        rw * A ** (1.0 / 3.0),
        aw,
        ud,
        rd * A ** (1.0 / 3.0),
        ad,
    )
    spin_orbit_params = (
        uso,
        rso * A ** (1.0 / 3.0),
        aso,
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
