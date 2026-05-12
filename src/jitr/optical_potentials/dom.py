"""
KD/Washu inspired local energy dependet dispersive optical.

Allows for energy dependent diffuseness in the surface and spin orbit terms.

Dispersion corrections are enforced analytically, but only using the energy
dependence of the depths of the imaginary terms, ignoring the energy dependence
of the diffuseness. This is an approximation.
"""

import numpy as np
from scipy.special import exp1, expi

from jitr.utils.constants import ALPHA, HBARC

from .potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)


def central(r, V, R, a, Wv, Delta_Vv, Rw, aw, Ws, Delta_Vs, Rd, ad):
    r"""
    The form of central term, including dispersive corrections to the real part
    of the potential. The real part includes the standard Woods-Saxon form,
    plus dispersive corrections to the volume and surface terms. The imaginary
    part includes both volume and surface terms, with the surface term having a
    derivative form.

    The dispersive corrections are included as additional terms in the real
    part of the potential, with the same radial dependence as the corresponding
    imaginary terms. The overall sign convention is such that the potential is
    negative for an attractive potential and positive for a repulsive
    potential.

    Parameters:
    ----------
    r : float or array-like
        The radial distance from the center of the nucleus in fm
    V : float
        The depth of the real part of the potential in MeV
    R : float
        The radius parameter for the real part of the potential in fm
    a : float
        The diffuseness parameter for the real part of the potential in fm
    Wv : float
        The depth of the volume imaginary part of the potential in MeV
    Delta_Vv : float
        The dispersive correction to the volume real part of the potential in
        MeV
    Rw : float
        The radius parameter for the imaginary volume part of the potential in
        fm
    aw : float
        The diffuseness parameter for the imaginary volume part of the
        potential in fm
    Ws : float
        The depth of the surface imaginary part of the potential in MeV
    Delta_Vs : float
        The dispersive correction to the surface real part of the potential in
        MeV
    Rd : float
        The radius parameter for the surface imaginary part of the potential in
        fm
    ad : float
        The diffuseness parameter for the surface imaginary part of the
        potential in fm
    """
    volume = V * woods_saxon_safe(r, R, a)
    imag_volume = (Delta_Vv + 1j * Wv) * woods_saxon_safe(r, Rw, aw)
    imag_surface = -(4 * ad * (Delta_Vs + 1j * Ws)) * woods_saxon_prime_safe(r, Rd, ad)
    return -volume - imag_volume - imag_surface


def spin_orbit(r, Vso, Wso, Rso, aso):
    """
    The form of the spin-orbit term, including dispersive corrections to the
    real part of the potential. The real part includes the standard Thomas
    form, plus dispersive corrections to the spin-orbit term. The imaginary
    part has the same radial dependence as the real part, with a depth
    parameter Wso.

    Parameters:
    ----------
    r : float or array-like
        The radial distance from the center of the nucleus in fm
    Vso : float
        The depth of the real part of the spin-orbit potential in MeV
    Wso : float
        The depth of the imaginary part of the spin-orbit potential in MeV
    Rso : float
        The radius parameter for the spin-orbit potential in fm
    aso : float
        The diffuseness parameter for the spin-orbit potential in fm
    """
    return 2 * (Vso + 1j * Wso) * thomas_safe(r, Rso, aso)


def central_plus_coulomb(
    r,
    central_params,
    coulomb_params,
):
    """
    The total central potential, including the Coulomb term for proton.

    Parameters:
    ----------
    r : float or array-like
        The radial projectile-target c.m. frame distance in fm
    central_params : tuple
        A tuple containing the parameters for the central potential (V, R, a,
        Wv Delta_Vv, Ws, Delta_Vs, Rw, aw)
    coulomb_params : tuple
        A tuple containing the parameters for the Coulomb potential (Zz, RC)
    """
    coulomb = coulomb_charged_sphere(r, *coulomb_params)
    centr = central(r, *central_params)
    return centr + coulomb


# depths
def V_depth(DeltaE, V0, v01):
    return V0 * np.exp(-v01 * DeltaE)


def Vso_depth(DeltaE, Vso, v01):
    return Vso * np.exp(-v01 * DeltaE)


def Wv_depth(DeltaE, Wv0, wv1):
    return Wv0 * DeltaE**2 / (DeltaE**2 + wv1**2)


def Wso_depth(DeltaE, Wso, wso1):
    return Wso * DeltaE**2 / (DeltaE**2 + wso1**2)


def Ws_depth(DeltaE, Ws0, ws1, ws2):
    return Ws0 * DeltaE**4 / (DeltaE**4 + ws1**4) * np.exp(-ws2 * np.abs(DeltaE))


def Delta_Vv_depth(DeltaE, Wv0, wv1):
    return Wv0 * wv1 * DeltaE / (DeltaE**2 + wv1**2)


def Delta_Vso_depth(DeltaE, Wso, wso1):
    return Wso * wso1 * DeltaE / (DeltaE**2 + wso1**2)


def Delta_Vs_depth(DeltaE, Ws0, ws1, ws2):
    return delta_Vs_analytic(DeltaE, Ws0, ws1, ws2)


def calculate_params(
    projectile: tuple,
    target: tuple,
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
):
    """
    Calculate the parameters for the optical model potential.

    Parameters:
    ----------
    projectile : tuple
        A tuple containing the mass number and charge of the projectile.
    target : tuple
        A tuple containing the mass number and charge of the target.
    Ecm : float
        The center-of-mass energy of the reaction in MeV
    Ef : float, optional
        The Fermi energy in MeV
    V0 : float, optional
        The depth of the real part of the potential at zero energy in MeV
    v1 : float, optional
        The energy dependence parameter for the real part of the potential in
        1/MeV
    rv : float, optional
        The radius parameter for the real part of the potential in fm
    av : float, optional
        The diffuseness parameter for the real part of the potential in fm
    wv0 : float, optional
        The depth of the volume imaginary part of the potential at zero energy
        in MeV
    wv1 : float, optional
        The energy dependence parameter for the volume imaginary part of the
        potential in MeV
    rw : float, optional
        The radius parameter for the imaginary part of the potential in fm
    aw : float, optional
        The diffuseness parameter for the imaginary part of the potential in fm
    ws0 : float, optional
        The depth of the surface imaginary part of the potential at zero energy
        in MeV
    ws1 : float, optional
        The energy dependence parameter for the surface imaginary part of the
        potential in MeV
    ws2 : float, optional
        The exponential decay parameter for the surface imaginary part of the
        potential in 1/MeV
    rd : float, optional
        The radius parameter for the surface imaginary part of the potential in
        fm
    ad : float, optional
        The diffuseness parameter for the surface imaginary part of the
        potential in fm
    vso0 : float, optional
        The depth of the real part of the spin-orbit potential at zero energy
        in MeV
    wso0 : float, optional
        The depth of the imaginary part of the spin-orbit potential at zero
        energy in MeV
    wso1 : float, optional
        The energy dependence parameter for the spin-orbit potential in MeV
    rso : float, optional
        The radius parameter for the spin-orbit potential in fm
    aso : float, optional
        The diffuseness parameter for the spin-orbit potential in fm
    rC : float, optional
        The radius parameter for the Coulomb potential in fm
    """

    # set up reaction
    A, Z = target
    Ap, Zp = projectile
    is_proton = Zp == 1
    assert Ap == 1 and Zp in (0, 1)

    R0 = rv * A ** (1 / 3)
    Rw = rw * A ** (1 / 3)
    Rd = rd * A ** (1 / 3)
    Rso = rso * A ** (1 / 3)
    RC = rC * A ** (1 / 3)

    # Coulomb correction
    Ec = coulomb_correction(Z * Zp, RC) if is_proton else 0.0
    delta_E = Ecm - Ec - Ef

    # real, local depths
    # (static with energy dpendence due to exchange/locality)
    V0 = V_depth(delta_E, v0, v1)
    Vso = Vso_depth(delta_E, vso0, v1)

    # dynamic imaginary parts (energy dependent)
    Wv = Wv_depth(delta_E, wv0, wv1)
    Ws = Ws_depth(delta_E, ws0, ws1, ws2)
    Wso = Wso_depth(delta_E, wso0, wso1)

    # dispersive corrections (energy dependent)
    Delta_Vv = Delta_Vv_depth(delta_E, wv0, wv1)
    Delta_Vso = Delta_Vso_depth(delta_E, wso0, wso1)
    Delta_Vs = Delta_Vs_depth(delta_E, ws0, ws1, ws2)

    central_params = (V0, R0, av, Wv, Delta_Vv, Rw, aw, Ws, Delta_Vs, Rd, ad)
    spin_orbit_params = (Vso + Delta_Vso, Wso, Rso, aso)
    coulomb_params = (Z * Zp, RC)

    return (central_params, coulomb_params), spin_orbit_params


def coulomb_correction(Zz, RC):
    r"""
    Coulomb correction for proton energy

    Parameters:
    ----------
    Zz : int
        The product of the charge numbers of the projectile and target (Z * Zp)
    RC : float
        The radius parameter for the Coulomb potential in fm
    """
    return 6.0 * Zz * ALPHA * HBARC / (5 * RC)


def delta_Vs_analytic(delta_E, Ws0, ws1, ws2):
    """
    Analytic expression for the dispersive correction to the surface imaginary
    potential, derived from the dispersion relation and the assumed energy
    dependence of the surface imaginary potential. This expression is based on
    the work of Mahaux and Sartor (1991) and is valid for a surface imaginary
    potential of the form W_s(E) = W_s0 * E^4 / (E^4 + ws1^4) * exp(-ws2 *
    |E|).

    Parameters:
    ----------
    delta_E : float or array-like
        The energy difference (E - E_Fermi) in MeV
    Ws0 : float
        The depth of the surface imaginary potential at zero energy in MeV
    ws1 : float
        The energy dependence parameter for the surface imaginary potential in
        MeV
    ws2 : float
        The exponential decay parameter for the surface imaginary potential in
        1/MeV

    """
    dE = np.atleast_1d(np.asarray(delta_E, dtype=float))
    scalar = np.ndim(delta_E) == 0
    out = np.zeros_like(dE)

    nz = dE != 0.0
    if not np.any(nz):
        return out.item() if scalar else out

    x = dE[nz]
    ax = np.abs(x)

    # ── J1 ────────────────────────────────────────────────────────────────
    if ws2 == 0.0:
        J1 = np.zeros_like(x)
    else:
        z1 = ws2 * ax
        J1 = (-np.exp(-z1) * expi(z1) - np.exp(z1) * exp1(z1)) / (2.0 * ax)

    # ── J2 ────────────────────────────────────────────────────────────────
    if ws2 == 0.0:
        J2 = np.pi * (ws1**2 + x**2) / (2.0 * np.sqrt(2.0) * ws1**3)
    else:
        r = ws1 * np.array([np.exp(1j * np.pi / 4), np.exp(1j * 3 * np.pi / 4)])
        F = np.exp(-ws2 * r) * exp1(-ws2 * r)
        contrib = (r[None, :] ** 2 + x[:, None] ** 2) / (4.0 * r[None, :] ** 3) * F
        J2 = 2.0 * np.real(np.sum(contrib, axis=1))

    pre1 = x**4 / (x**4 + ws1**4)
    pre2 = ws1**4 / (x**4 + ws1**4)
    out[nz] = (2.0 * Ws0 * x / np.pi) * (pre1 * J1 + pre2 * J2)

    return out.item() if scalar else out


def extract_params(ws, *params):
    rxn = ws.reaction
    projectile = rxn.projectile
    target = rxn.target
    Ecm = ws.kinematics.Ecm
    Ef = rxn.Ef
    central_params, spin_orbit_params = calculate_params(
        projectile, target, Ecm, Ef, *params
    )
    return central_params, spin_orbit_params
