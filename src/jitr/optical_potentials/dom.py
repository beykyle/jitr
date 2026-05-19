"""Dispersive optical-model helpers and a local DOM wrapper.

This module provides analytic building blocks for a local dispersive optical
model (DOM) together with a :class:`DOM` wrapper that fits the repository's
``SingleChannelOpticalModel`` interface.

The current DOM implementation keeps the standard analytic dispersive
corrections for the energy-dependent depths. The helper functions remain useful
on their own, while the wrapper gives callers a first-class optical-model entry
point consistent with the other built-in OMP modules.
"""

from __future__ import annotations

import numpy as np
from scipy.special import exp1, expi

from .._types import ArrayOrScalar, PotentialArray
from ..reactions.reaction import Reaction
from ..utils.constants import ALPHA, HBARC
from ..utils.kinematics import ChannelKinematics
from .omp import SingleChannelOpticalModel, _as_potential_array
from .potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)

PARAM_NAMES: tuple[str, ...] = (
    "v0",
    "v1",
    "rv",
    "av",
    "wv0",
    "wv1",
    "rw",
    "aw",
    "ws0",
    "ws1",
    "ws2",
    "rd",
    "ad",
    "vso0",
    "wso0",
    "wso1",
    "rso",
    "aso",
    "rC",
)


def get_param_names() -> list[str]:
    """Return the DOM parameter names in ``calculate_params`` order.

    Returns:
        Ordered list of DOM parameter names.
    """

    return list(PARAM_NAMES)


def central(
    r: ArrayOrScalar,
    V: float,
    R: float,
    a: float,
    Wv: float,
    Delta_Vv: float,
    Rw: float,
    aw: float,
    Ws: float,
    Delta_Vs: float,
    Rd: float,
    ad: float,
) -> PotentialArray:
    r"""Evaluate the local DOM central potential.

    The real part contains the standard Woods-Saxon volume term plus analytic
    dispersive corrections tied to the imaginary volume and surface strengths.

    Args:
        r: Radial coordinate or grid in fm.
        V: Real volume depth in MeV.
        R: Real volume radius in fm.
        a: Real volume diffuseness in fm.
        Wv: Imaginary volume depth in MeV.
        Delta_Vv: Dispersive correction to the real volume depth in MeV.
        Rw: Imaginary volume radius in fm.
        aw: Imaginary volume diffuseness in fm.
        Ws: Imaginary surface depth in MeV.
        Delta_Vs: Dispersive correction to the real surface depth in MeV.
        Rd: Surface radius in fm.
        ad: Surface diffuseness in fm.

    Returns:
        Complex central potential evaluated on ``r``.
    """

    volume = V * woods_saxon_safe(r, R, a)
    imag_volume = (Delta_Vv + 1j * Wv) * woods_saxon_safe(r, Rw, aw)
    imag_surface = -(4 * ad * (Delta_Vs + 1j * Ws)) * woods_saxon_prime_safe(r, Rd, ad)
    result = -volume - imag_volume - imag_surface
    return _as_potential_array(result)


def spin_orbit(
    r: ArrayOrScalar,
    Vso: float,
    Wso: float,
    Rso: float,
    aso: float,
) -> PotentialArray:
    r"""Evaluate the local DOM spin-orbit term.

    Args:
        r: Radial coordinate or grid in fm.
        Vso: Real spin-orbit depth in MeV.
        Wso: Imaginary spin-orbit depth in MeV.
        Rso: Spin-orbit radius in fm.
        aso: Spin-orbit diffuseness in fm.

    Returns:
        Complex Thomas-form spin-orbit potential.
    """

    result = 2 * (Vso + 1j * Wso) * thomas_safe(r, Rso, aso)
    return _as_potential_array(result)


def central_plus_coulomb(
    r: ArrayOrScalar,
    central_params: tuple[float, ...],
    coulomb_params: tuple[float, ...],
) -> PotentialArray:
    """Evaluate the total central-plus-Coulomb potential for proton scattering.

    Args:
        r: Radial coordinate or grid in fm.
        central_params: Arguments for :func:`central`.
        coulomb_params: Arguments for :func:`coulomb_charged_sphere`.

    Returns:
        Complex central potential plus Coulomb term.
    """

    result = central(r, *central_params) + coulomb_charged_sphere(r, *coulomb_params)
    return _as_potential_array(result)


def V_depth(DeltaE: ArrayOrScalar, V0: float, v01: float) -> ArrayOrScalar:
    """Return the real volume depth parameterisation.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        V0: Zero-offset real depth in MeV.
        v01: Exponential slope in MeV^-1.

    Returns:
        Real volume depth at ``DeltaE``.
    """

    return V0 * np.exp(-v01 * DeltaE)


def Vso_depth(DeltaE: ArrayOrScalar, Vso: float, v01: float) -> ArrayOrScalar:
    """Return the real spin-orbit depth parameterisation.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Vso: Zero-offset spin-orbit depth in MeV.
        v01: Exponential slope in MeV^-1.

    Returns:
        Real spin-orbit depth at ``DeltaE``.
    """

    return Vso * np.exp(-v01 * DeltaE)


def Wv_depth(DeltaE: ArrayOrScalar, Wv0: float, wv1: float) -> ArrayOrScalar:
    """Return the imaginary volume depth parameterisation.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Wv0: Saturation depth in MeV.
        wv1: Energy scale in MeV.

    Returns:
        Imaginary volume depth at ``DeltaE``.
    """

    return Wv0 * DeltaE**2 / (DeltaE**2 + wv1**2)


def Wso_depth(DeltaE: ArrayOrScalar, Wso: float, wso1: float) -> ArrayOrScalar:
    """Return the imaginary spin-orbit depth parameterisation.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Wso: Saturation depth in MeV.
        wso1: Energy scale in MeV.

    Returns:
        Imaginary spin-orbit depth at ``DeltaE``.
    """

    return Wso * DeltaE**2 / (DeltaE**2 + wso1**2)


def Ws_depth(
    DeltaE: ArrayOrScalar,
    Ws0: float,
    ws1: float,
    ws2: float,
) -> ArrayOrScalar:
    """Return the imaginary surface depth parameterisation.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Ws0: Surface-depth scale in MeV.
        ws1: Quartic energy scale in MeV.
        ws2: Exponential damping scale in MeV^-1.

    Returns:
        Imaginary surface depth at ``DeltaE``.
    """

    return Ws0 * DeltaE**4 / (DeltaE**4 + ws1**4) * np.exp(-ws2 * np.abs(DeltaE))


def Delta_Vv_depth(DeltaE: ArrayOrScalar, Wv0: float, wv1: float) -> ArrayOrScalar:
    """Return the analytic dispersion correction to the volume depth.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Wv0: Saturation imaginary volume depth in MeV.
        wv1: Energy scale in MeV.

    Returns:
        Real dispersive correction to the volume depth.
    """

    return Wv0 * wv1 * DeltaE / (DeltaE**2 + wv1**2)


def Delta_Vso_depth(
    DeltaE: ArrayOrScalar,
    Wso: float,
    wso1: float,
) -> ArrayOrScalar:
    """Return the analytic dispersion correction to the spin-orbit depth.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Wso: Saturation imaginary spin-orbit depth in MeV.
        wso1: Energy scale in MeV.

    Returns:
        Real dispersive correction to the spin-orbit depth.
    """

    return Wso * wso1 * DeltaE / (DeltaE**2 + wso1**2)


def Delta_Vs_depth(
    DeltaE: ArrayOrScalar,
    Ws0: float,
    ws1: float,
    ws2: float,
) -> ArrayOrScalar:
    """Return the analytic surface-dispersion correction.

    Args:
        DeltaE: Energy offset from the Fermi energy in MeV.
        Ws0: Surface-depth scale in MeV.
        ws1: Quartic energy scale in MeV.
        ws2: Exponential damping scale in MeV^-1.

    Returns:
        Analytic surface-dispersion correction.
    """

    return delta_Vs_analytic(DeltaE, Ws0, ws1, ws2)


def calculate_params(
    projectile: tuple[int, int],
    target: tuple[int, int],
    Ecm: float,
    Ef: float,
    v0: float,
    v1: float,
    rv: float,
    av: float,
    wv0: float,
    wv1: float,
    rw: float,
    aw: float,
    ws0: float,
    ws1: float,
    ws2: float,
    rd: float,
    ad: float,
    vso0: float,
    wso0: float,
    wso1: float,
    rso: float,
    aso: float,
    rC: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Assemble central, Coulomb, and spin-orbit DOM parameters.

    Args:
        projectile: Projectile ``(A, Z)`` tuple. Must be a nucleon.
        target: Target ``(A, Z)`` tuple.
        Ecm: Center-of-mass energy in MeV.
        Ef: Fermi energy in MeV.
        v0: Real volume depth scale in MeV.
        v1: Real depth exponential slope in MeV^-1.
        rv: Real volume reduced radius in fm.
        av: Real volume diffuseness in fm.
        wv0: Imaginary volume depth scale in MeV.
        wv1: Imaginary volume energy scale in MeV.
        rw: Imaginary volume reduced radius in fm.
        aw: Imaginary volume diffuseness in fm.
        ws0: Imaginary surface depth scale in MeV.
        ws1: Imaginary surface quartic energy scale in MeV.
        ws2: Imaginary surface exponential damping scale in MeV^-1.
        rd: Surface reduced radius in fm.
        ad: Surface diffuseness in fm.
        vso0: Real spin-orbit depth scale in MeV.
        wso0: Imaginary spin-orbit depth scale in MeV.
        wso1: Imaginary spin-orbit energy scale in MeV.
        rso: Spin-orbit reduced radius in fm.
        aso: Spin-orbit diffuseness in fm.
        rC: Coulomb reduced radius in fm.

    Returns:
        ``((central_params, coulomb_params), spin_orbit_params)``.
    """

    A, Z = target
    Ap, Zp = projectile
    is_proton = Zp == 1
    assert Ap == 1 and Zp in (0, 1)

    R0 = rv * A ** (1 / 3)
    Rw = rw * A ** (1 / 3)
    Rd = rd * A ** (1 / 3)
    Rso = rso * A ** (1 / 3)
    RC = rC * A ** (1 / 3)

    Ec = coulomb_correction(Z * Zp, RC) if is_proton else 0.0
    delta_E = Ecm - Ec - Ef

    V0 = float(V_depth(delta_E, v0, v1))
    Vso = float(Vso_depth(delta_E, vso0, v1))
    Wv = float(Wv_depth(delta_E, wv0, wv1))
    Ws = float(Ws_depth(delta_E, ws0, ws1, ws2))
    Wso = float(Wso_depth(delta_E, wso0, wso1))
    Delta_Vv = float(Delta_Vv_depth(delta_E, wv0, wv1))
    Delta_Vso = float(Delta_Vso_depth(delta_E, wso0, wso1))
    Delta_Vs = float(Delta_Vs_depth(delta_E, ws0, ws1, ws2))

    central_params = (
        V0,
        R0,
        av,
        Wv,
        Delta_Vv,
        Rw,
        aw,
        Ws,
        Delta_Vs,
        Rd,
        ad,
    )
    spin_orbit_params = (Vso + Delta_Vso, Wso, Rso, aso)
    coulomb_params = (float(Z * Zp), RC)

    return central_params, spin_orbit_params, coulomb_params


def coulomb_correction(Zz: int, RC: float) -> float:
    r"""Return the proton Coulomb energy correction.

    Args:
        Zz: Product of projectile and target charge numbers.
        RC: Coulomb radius in fm.

    Returns:
        Coulomb energy correction in MeV.
    """

    return 6.0 * Zz * ALPHA * HBARC / (5 * RC)


def delta_Vs_analytic(
    delta_E: ArrayOrScalar,
    Ws0: float,
    ws1: float,
    ws2: float,
) -> ArrayOrScalar:
    r"""Return the analytic surface-dispersion correction.

    This expression matches the Mahaux-Sartor-style dispersive correction for
    the surface depth

    .. math::

       W_s(E) = W_{s0}\,\frac{E^4}{E^4 + w_{s1}^4}\,\exp(-w_{s2}|E|).

    Args:
        delta_E: Energy offset from the Fermi energy in MeV.
        Ws0: Surface-depth scale in MeV.
        ws1: Quartic energy scale in MeV.
        ws2: Exponential damping scale in MeV^-1.

    Returns:
        Surface-dispersion correction evaluated at ``delta_E``.
    """

    dE = np.atleast_1d(np.asarray(delta_E, dtype=float))
    scalar = np.ndim(delta_E) == 0
    out = np.zeros_like(dE)

    nz = dE != 0.0
    if not np.any(nz):
        return out.item() if scalar else out

    x = dE[nz]
    ax = np.abs(x)

    if ws2 == 0.0:
        J1 = np.zeros_like(x)
    else:
        z1 = ws2 * ax
        J1 = (-np.exp(-z1) * expi(z1) - np.exp(z1) * exp1(z1)) / (2.0 * ax)

    if ws2 == 0.0:
        J2 = np.pi * (ws1**2 + x**2) / (2.0 * np.sqrt(2.0) * ws1**3)
    else:
        roots = ws1 * np.array([np.exp(1j * np.pi / 4), np.exp(1j * 3 * np.pi / 4)])
        F = np.exp(-ws2 * roots) * exp1(-ws2 * roots)
        contrib = (
            (roots[None, :] ** 2 + x[:, None] ** 2) / (4.0 * roots[None, :] ** 3) * F
        )
        J2 = 2.0 * np.real(np.sum(contrib, axis=1))

    pre1 = x**4 / (x**4 + ws1**4)
    pre2 = ws1**4 / (x**4 + ws1**4)
    out[nz] = (2.0 * Ws0 * x / np.pi) * (pre1 * J1 + pre2 * J2)

    return out.item() if scalar else out


def extract_params(
    reaction: Reaction,
    kinematics: ChannelKinematics,
    *params: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Extract DOM parameters from explicit reaction and kinematics inputs.

    Args:
        reaction: Reaction carrying projectile and target information.
        kinematics: Channel kinematics for the current energy point.
        *params: DOM parameters in :func:`get_param_names` order.

    Returns:
        ``(central_params, spin_orbit_params, coulomb_params)``.
    """
    Ef = reaction.Ef if reaction.Ef is not None else 0.0
    return calculate_params(
        (reaction.projectile.A, reaction.projectile.Z),
        (reaction.target.A, reaction.target.Z),
        kinematics.Ecm,
        Ef,
        *params,
    )


class DOM(SingleChannelOpticalModel):
    """Local dispersive optical model compatible with ``SingleChannelOpticalModel``.

    The DOM parameter order is given by :func:`get_param_names`.
    """

    def __init__(self) -> None:
        super().__init__(params=get_param_names())

    def evaluate(
        self,
        rgrid: ArrayOrScalar,
        reaction: Reaction,
        kinematics: ChannelKinematics,
        *params: float,
    ) -> tuple[PotentialArray, PotentialArray, ArrayOrScalar]:
        """Evaluate the DOM central, spin-orbit, and Coulomb terms.

        Args:
            rgrid: Radial coordinate or grid in fm.
            reaction: Reaction carrying projectile and target information.
            kinematics: Channel kinematics for the current energy point.
            *params: DOM parameters in :func:`get_param_names` order.

        Returns:
            ``(central_term, spin_orbit_term, coulomb_term)``.
        """

        if len(params) != self.n_params:
            raise ValueError(
                f"DOM expects {self.n_params} parameters in get_param_names() order, "
                f"got {len(params)}."
            )

        Ef = reaction.Ef if reaction.Ef is not None else 0.0
        central_params, spin_orbit_params, coulomb_params = calculate_params(
            (reaction.projectile.A, reaction.projectile.Z),
            (reaction.target.A, reaction.target.Z),
            kinematics.Ecm,
            Ef,
            *params,
        )
        return (
            central(rgrid, *central_params),
            spin_orbit(rgrid, *spin_orbit_params),
            coulomb_charged_sphere(rgrid, *coulomb_params),
        )
