import numpy as np


def scalar(J2: int, pi: bool):
    r"""For spin-0 spin-0 coupling, there is only one possible orbital angular momentum
        for a given total angular momentum J:
            - J = l

    Parameters:
        J2 (int): double the total angular momentum in the channel
        pi (bool): parity (+ is True, - is False)

    Returns:
        l (np.ndarray): l in each channel (just one channel with l = J)
        couplings (np.ndarray): just 1
    """
    J = J2 / 2
    ls = np.array([J])
    return ls, np.array([[1.0]])


def spin_orbit_spin_half(J2: int , pi: bool):
    r"""For a spin-1/2 nucleon scattering off a spin-0 nucleus with spin-obit coupling,
    there are 2 orbital angular momenta that couple to total channel spin J:
        - l = J - 1/2 (unless J = 0)
        - l - J + 1/2

    Parameters:
        J2 (int): double the total angular momentum in the channel
        pi (bool): parity (+ is True, - is False)

    Returns:
        l (np.ndarray): l in each channel
        couplings (np.ndarray): expectation value of l dot s in each channel
    """
    J = J2 / 2
    if J == 0:
        ls = np.array([J + 1/2])
    else:
        ls = np.array([J - 1/2, J + 1/2])

    return ls, np.diag([(J * (J + 1) - l * (l + 1) - 0.5 * (0.5 + 1)) for l in ls])
