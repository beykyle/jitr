import numpy as np
from numba import njit

from .free_solutions import (
    CoulombAsymptotics,
    H_plus,
    H_minus,
    H_plus_prime,
    H_minus_prime,
)


@njit
def complex_det(matrix: np.array):
    d = np.linalg.det(matrix @ np.conj(matrix).T)
    return np.sqrt(d)


@njit
def block(matrix: np.array, block, block_size):
    """
    get submatrix with coordinates block from matrix, where
    each block is defined by block_size elements along each dimension
    """
    i, j = block
    n, m = block_size
    return matrix[i * n : i * n + n, j * m : j * m + m]


def second_derivative_op(s, channel, interaction, args=()):
    r"""second derivative operator of reduced, scaled radial Schrodinger equation"""
    return (
        eval_scaled_interaction(s, interaction, channel, args)
        + channel.l * (channel.l + 1) / s**2
        - 1.0
    )


def schrodinger_eqn_ivp_order1(s, y, channel, interaction, args=()):
    r"""
    callable for scipy.integrate.solve_ivp; converts SE to
    2 coupled 1st order ODEs
    """
    u, uprime = y
    return [uprime, second_derivative_op(s, channel, interaction, args) * u]


def smatrix(Rl, a, l, eta, asym=CoulombAsymptotics):
    """
    Calculates channel S-Matrix from channel R-matrix (logarithmic
    derivative of channel wavefunction at channel radius)
    """
    return (
        H_minus(a, l, eta, asym=asym) - a * Rl * H_minus_prime(a, l, eta, asym=asym)
    ) / (H_plus(a, l, eta, asym=asym) - a * Rl * H_plus_prime(a, l, eta, asym=asym))


def delta(Sl):
    """
    returns the phase shift and attentuation factor in degrees
    """
    delta = np.log(Sl) / 2.0j  # complex phase shift in radians
    return np.rad2deg(np.real(delta)), np.rad2deg(np.imag(delta))


def eval_scaled_interaction(s, interaction, ch, args):
    return interaction(s / ch.k, *args) / ch.E


def eval_scaled_nonlocal_interaction(s, sp, interaction, ch, args):
    return interaction(s / ch.k, sp / ch.k, *args) / ch.E
