"""The Whitehead-Lim-Holt potential is a global mcroscopic optical potential for nuclear
scattering.

See the [paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.182502) for
details. Equation references are with respect to (w.r.t.) this paper.
"""

from pathlib import Path
import json

from ..utils.constants import MASS_PION
from .potentials import woods_saxon_safe, woods_saxon_prime_safe, coulomb_charged_sphere


def WLH_so(r, uso, rso, aso):
    r"""WLH spin-orbit terms"""
    return (uso / MASS_PION**2) / r * woods_saxon_prime_safe(r, rso, aso)


def WLH_central(r, uv, rv, av, uw, rw, aw, ud, rd, ad):
    r"""WLH without the spin-orbit term"""
    return (
        -uv * woods_saxon_safe(r, rv, av)
        - 1j * uw * woods_saxon_safe(r, rw, aw)
        - 1j * (-4 * ad) * ud * woods_saxon_prime_safe(r, rd, ad)
    )


def WLH_plus_coulomb(r, central_params, coulomb_params):
    nucl = WLH_central(r, *central_params)
    coul = coulomb_charged_sphere(r, *coulomb_params)
    return nucl + coul


class WLHGlobal:
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
                "WLHGlobal is defined only for neutron and proton projectiles"
            )

        self.projectile = projectile

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)

            if "WLHReal" in data:
                self.uv0 = data["WLHReal"]["V0" + tag]
                self.uv1 = data["WLHReal"]["V1" + tag]
                self.uv2 = data["WLHReal"]["V2" + tag]
                self.uv3 = data["WLHReal"]["V3" + tag]
                self.uv4 = data["WLHReal"]["V4" + tag]
                self.uv5 = data["WLHReal"]["V5" + tag]
                self.uv6 = data["WLHReal"]["V6" + tag]
                self.rv0 = data["WLHReal"]["r0" + tag]
                self.rv1 = data["WLHReal"]["r1" + tag]
                self.rv2 = data["WLHReal"]["r2" + tag]
                self.rv3 = data["WLHReal"]["r3" + tag]
                self.av0 = data["WLHReal"]["a0" + tag]
                self.av1 = data["WLHReal"]["a1" + tag]
                self.av2 = data["WLHReal"]["a2" + tag]
                self.av3 = data["WLHReal"]["a3" + tag]
                self.av4 = data["WLHReal"]["a4" + tag]
                self.uw0 = data["WLHImagVolume"]["W0" + tag]
                self.uw1 = data["WLHImagVolume"]["W1" + tag]
                self.uw2 = data["WLHImagVolume"]["W2" + tag]
                self.uw3 = data["WLHImagVolume"]["W3" + tag]
                self.uw4 = data["WLHImagVolume"]["W4" + tag]
                self.rw0 = data["WLHImagVolume"]["r0" + tag]
                self.rw1 = data["WLHImagVolume"]["r1" + tag]
                self.rw2 = data["WLHImagVolume"]["r2" + tag]
                self.rw3 = data["WLHImagVolume"]["r3" + tag]
                self.rw4 = data["WLHImagVolume"]["r4" + tag]
                self.rw5 = data["WLHImagVolume"]["r5" + tag]
                self.aw0 = data["WLHImagVolume"]["a0" + tag]
                self.aw1 = data["WLHImagVolume"]["a1" + tag]
                self.aw2 = data["WLHImagVolume"]["a2" + tag]
                self.aw3 = data["WLHImagVolume"]["a3" + tag]
                self.aw4 = data["WLHImagVolume"]["a4" + tag]
                self.ud0 = data["WLHImagSurface"]["W0" + tag]
                self.ud1 = data["WLHImagSurface"]["W1" + tag]
                self.ud3 = data["WLHImagSurface"]["W2" + tag]
                self.ud4 = data["WLHImagSurface"]["W3" + tag]
                self.rd0 = data["WLHImagSurface"]["r0" + tag]
                self.rd1 = data["WLHImagSurface"]["r1" + tag]
                self.rd2 = data["WLHImagSurface"]["r2" + tag]
                self.ad0 = data["WLHImagSurface"]["a0" + tag]
                self.uso0 = data["WLHRealSpinOrbit"]["V0" + tag]
                self.uso1 = data["WLHRealSpinOrbit"]["V1" + tag]
                self.rso0 = data["WLHRealSpinOrbit"]["r0" + tag]
                self.rso1 = data["WLHRealSpinOrbit"]["r1" + tag]
                self.aso0 = data["WLHRealSpinOrbit"]["a0" + tag]
                self.aso1 = data["WLHRealSpinOrbit"]["a1" + tag]
            elif f"WLHRealSpinOrbit_a1{tag}" in data:
                self.uv0 = data["WLHReal_V0" + tag]
                self.uv1 = data["WLHReal_V1" + tag]
                self.uv2 = data["WLHReal_V2" + tag]
                self.uv3 = data["WLHReal_V3" + tag]
                self.uv4 = data["WLHReal_V4" + tag]
                self.uv5 = data["WLHReal_V5" + tag]
                self.uv6 = data["WLHReal_V6" + tag]
                self.rv0 = data["WLHReal_r0" + tag]
                self.rv1 = data["WLHReal_r1" + tag]
                self.rv2 = data["WLHReal_r2" + tag]
                self.rv3 = data["WLHReal_r3" + tag]
                self.av0 = data["WLHReal_a0" + tag]
                self.av1 = data["WLHReal_a1" + tag]
                self.av2 = data["WLHReal_a2" + tag]
                self.av3 = data["WLHReal_a3" + tag]
                self.av4 = data["WLHReal_a4" + tag]
                self.uw0 = data["WLHImagVolume_W0" + tag]
                self.uw1 = data["WLHImagVolume_W1" + tag]
                self.uw2 = data["WLHImagVolume_W2" + tag]
                self.uw3 = data["WLHImagVolume_W3" + tag]
                self.uw4 = data["WLHImagVolume_W4" + tag]
                self.rw0 = data["WLHImagVolume_r0" + tag]
                self.rw1 = data["WLHImagVolume_r1" + tag]
                self.rw2 = data["WLHImagVolume_r2" + tag]
                self.rw3 = data["WLHImagVolume_r3" + tag]
                self.rw4 = data["WLHImagVolume_r4" + tag]
                self.rw5 = data["WLHImagVolume_r5" + tag]
                self.aw0 = data["WLHImagVolume_a0" + tag]
                self.aw1 = data["WLHImagVolume_a1" + tag]
                self.aw2 = data["WLHImagVolume_a2" + tag]
                self.aw3 = data["WLHImagVolume_a3" + tag]
                self.aw4 = data["WLHImagVolume_a4" + tag]
                self.ud0 = data["WLHImagSurface_W0" + tag]
                self.ud1 = data["WLHImagSurface_W1" + tag]
                self.ud3 = data["WLHImagSurface_W2" + tag]
                self.ud4 = data["WLHImagSurface_W3" + tag]
                self.rd0 = data["WLHImagSurface_r0" + tag]
                self.rd1 = data["WLHImagSurface_r1" + tag]
                self.rd2 = data["WLHImagSurface_r2" + tag]
                self.ad0 = data["WLHImagSurface_a0" + tag]
                self.uso0 = data["WLHRealSpinOrbit_V0" + tag]
                self.uso1 = data["WLHRealSpinOrbit_V1" + tag]
                self.rso0 = data["WLHRealSpinOrbit_r0" + tag]
                self.rso1 = data["WLHRealSpinOrbit_r1" + tag]
                self.aso0 = data["WLHRealSpinOrbit_a0" + tag]
                self.aso1 = data["WLHRealSpinOrbit_a1" + tag]
            else:
                raise ValueError("Unrecognized parameter file format for WLH!")

    def get_params(self, A, Z, mu, E_lab, k):
        """
        Calculates WLH parameters for a given system
        """

        N = A - Z
        delta = (N - Z) / A
        factor = 1.0
        if self.projectile == (1, 0):
            factor = -1.0

        uv = (
            self.uv0
            - self.uv1 * E_lab
            + self.uv2 * E_lab**2
            + self.uv3 * E_lab**3
            + factor * (self.uv4 - self.uv5 * E_lab + self.uv6 * E_lab**2) * delta
        )
        rv = (
            self.rv0
            - self.rv1 * A ** (-1.0 / 3)
            - self.rv2 * E_lab
            + self.rv3 * E_lab**2
        )
        av = (
            self.av0
            - factor * self.av1 * E_lab
            - self.av2 * E_lab**2
            - (self.av3 - self.av4 * delta) * delta
        )

        uw = (
            self.uw0
            + self.uw1 * E_lab
            - self.uw2 * E_lab**2
            + (factor * self.uw3 - self.uw4 * E_lab) * delta
        )
        rw = (
            self.rw0
            + (self.rw1 + self.rw2 * A) / (self.rw3 + A + self.rw4 * E_lab)
            + self.rw5 * E_lab**2
        )
        aw = (
            self.aw0
            - (self.aw1 * E_lab) / (-self.aw2 - E_lab)
            + (self.aw3 - self.aw4 * E_lab) * delta
        )

        if (self.projectile == (1, 0) and E_lab < 40) or (
            self.projectile == (1, 1) and E_lab < 20 and A > 100
        ):
            ud = self.ud0 - self.ud1 * E_lab - (self.ud3 - self.ud4 * E_lab) * delta
        else:
            ud = 0

        rd = self.rd0 - self.rd2 * E_lab - self.rd1 * A ** (-1.0 / 3)
        ad = self.ad0

        uso = self.uso0 - self.uso1 * A
        rso = self.rso0 - self.rso1 * A ** (-1.0 / 3.0)
        aso = self.aso0 - self.aso1 * A

        R_C = rv * A ** (1.0 / 3.0)
        coulomb_params = (Z * self.projectile[1], R_C)
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
