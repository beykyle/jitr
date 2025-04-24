from numba import njit
from dataclasses import dataclass
from scipy.special import eval_legendre, lpmv, gamma
import numpy as np
import pickle

from ..utils import constants
from ..rmatrix import Solver


@dataclass
class ElasticXS:
    r"""
    Holds differential cross section, analyzing power, total cross section and
    reaction cross secton, all at a given energy
    """

    dsdo: np.ndarray
    Ay: np.ndarray
    t: np.float64
    rxn: np.float64


@njit
def integral_elastic_xs(
    k: float,
    Splus: np.array,
    Sminus: np.array,
    ls: np.array,
    sigma_l: np.array = 0,
):
    xsrxn = 0.0
    xst = 0.0

    for l in range(Splus.shape[0]):
        xsrxn += (l + 1) * (1 - np.absolute(Splus[l]) ** 2) + l * (
            1 - np.absolute(Sminus[l]) ** 2
        )
        xst += (l + 1) * (1 - np.real(Splus[l])) + l * (1 - np.real(Sminus[l]))

    xsrxn *= 10 * np.pi / k**2
    xst *= 10 * 2 * np.pi / k**2

    return xst, xsrxn


@njit
def differential_elastic_xs(
    k: float,
    angles: np.array,
    splus: np.array,
    sminus: np.array,
    ls: np.array,
    P_l_costheta: np.array,
    P_1_l_costheta: np.array,
    f_c: np.array = 0,
    sigma_l: np.array = 0,
):
    r"""
    Calculates differential, total and reaction cross section for spin-1/2 spin-0 scattering
    following Herman, et al., 2007, https://doi.org/10.1016/j.nds.2007.11.003
    """
    a = np.zeros_like(angles, dtype=np.complex128) + f_c
    b = np.zeros_like(angles, dtype=np.complex128)
    xsrxn = 0.0
    xst = 0.0

    for l in range(splus.shape[0]):
        a += (
            P_l_costheta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2j * k)
            * ((l + 1) * (splus[l] - 1) + l * (sminus[l] - 1))
        )
        b += (
            P_1_l_costheta[l, :]
            * np.exp(2j * sigma_l[l])
            / (2j * k)
            * (splus[l] - sminus[l])
        )

        xsrxn += (l + 1) * (1 - np.absolute(splus[l]) ** 2) + l * (
            1 - np.absolute(sminus[l]) ** 2
        )
        xst += (l + 1) * (1 - np.real(splus[l])) + l * (1 - np.real(sminus[l]))

    dsdo = (np.absolute(a) ** 2 + np.absolute(b) ** 2) * 10
    Ay = np.imag(a.conj() * b) * 10 / dsdo
    xsrxn *= 10 * np.pi / k**2
    xst *= 10 * 2 * np.pi / k**2

    return dsdo, Ay, xst, xsrxn
