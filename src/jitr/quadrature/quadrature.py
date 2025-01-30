import numpy as np
import scipy.special as sc


def laguerre(n: np.int32, a: np.float64, s: np.float64, quadrature):
    r"""
    nth Lagrange-Laguerre function, scaled by a. Eq. 3.70 in Baye, 2015
    with alpha = 0.

    Note: n is indexed from 1 (constant function is not part of basis)
    """
    assert n <= quadrature.nbasis and n >= 1

    x = s / a
    xn = quadrature.abscissa[n - 1]

    return (
        (-1) ** n
        / np.sqrt(xn)
        * sc.special.eval_laguerre(n, x)
        / (x - xn)
        * x
        * np.exp(-x / 2)
    )


def legendre(n: np.int32, a: np.float64, s: np.float64, quadrature):
    r"""
    nth Lagrange-Legendre polynomial shifted onto [0,a_i] and regularized by
    s.  Eq. 3.122 in Baye, 2015

    Note: n is indexed from 1 (constant function is not part of basis)
    """
    assert n <= quadrature.nbasis and n >= 1
    N = quadrature.nbasis
    x = s / a
    xn = quadrature.abscissa[n - 1]

    return (
        (-1.0) ** (N - n)
        * np.sqrt((1 - xn) / xn)
        * sc.eval_legendre(N, 2.0 * x - 1.0)
        * x
        / (x - xn)
    )


def generate_laguerre_quadrature(nbasis: int):
    r"""
    @returns zeros and weights for Gauss quadrature using the Lagrange-Laguerre
    basis. See Ch. 3.3 of Baye, 2015
    """
    return np.polynomial.laguerre.laggauss(nbasis)


def generate_legendre_quadrature(nbasis: int):
    r"""
    @returns zeros and weights for Gauss quadrature using the Lagrange-Legendre
    basis shifted and scaled onto [0,a]. See Ch. 3.4 of Baye, 2015
    """
    x, w = np.polynomial.legendre.leggauss(nbasis)
    x = 0.5 * (x + 1)
    w = 0.5 * w
    return x, w


class LagrangeLaguerreQuadrature:
    r"""
    Lagrange Laguerre mesh for the Schrödinger equation following ch. 3.3 of
    Baye, D.  (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel;
    e.g r  -> s_i = r * k_i, and each channel's equation is then divided by
    it's asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(
        self,
        abscissa: np.array,
        weights: np.array,
        overlap: np.array = None,
    ):
        """
        Constructs the Schrödinger equation in a basis of Lagrange Laguerre
        functions
        """
        self.nbasis = len(abscissa)
        assert len(abscissa) == len(weights)
        self.abscissa = abscissa
        self.weights = weights

        if overlap is None:
            # Eq. 3.71 in Baye, 2015
            imj = np.arange(self.nbasis) - np.arange(self.nbasis)[:, np.newaxis]
            self.overlap = (-1.0) ** imj / np.sqrt(np.outer(abscissa, abscissa))
        else:
            self.overlap = overlap

    def kinetic_operator_element(
        self, n: np.int32, m: np.int32, a: np.float64, l: np.int32
    ):
        """
        @returns the (n,m)th matrix element for the kinetic energy operator at
        channel radius a = k*r with orbital angular momentum l
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        N = self.nbasis

        # Eq. 3.77 in Baye, 2015
        correction = (-1) ** (n - m) / 4 / np.sqrt(xn * xm)

        if n == m:
            # Eq. 3.75 in [Baye, 2015], scaled by 1/E and with r->s=kr
            centrifugal = l * (l + 1) / (a * xn) ** 2
            radial = -1.0 / (12 * xn**2) * (xn**2 - 2 * (2 * N + 1) * xn - 4) / a**2
            return radial - correction + centrifugal
        else:
            # Eq. 3.76 in [Baye, 2015], scaled by 1/E and with r->s=kr
            return (-1) ** (n - m) * (xn + xm) / np.sqrt(xn * xm) / (
                xn - xm
            ) ** 2 / a**2 - correction

    def kinetic_matrix(self, a: np.float64, l: np.int32):
        r"""
        @returns the kinetic operator matrix in the Lagrange Laguerre basis
        """
        F = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)
        for n in range(1, self.nbasis + 1):
            for m in range(n, self.nbasis + 1):
                F[n - 1, m - 1] = self.kinetic_operator_element(n, m, a, l)
        F = F + np.triu(F, k=1).T
        return F


class LagrangeLegendreQuadrature:
    r"""
    Lagrange Legendre mesh for the Schrödinger equation following ch. 3.4 of
    Baye, D.  (2015). The Lagrange-mesh method. Physics reports, 565, 1-107,
    with the only difference being the domain is scaled in each channel; e.g.  r
    -> s_i = r * k_i, and each channel's equation is then divided by it's
    asymptotic kinetic energy in the channel T_i = E_inc - E_i
    """

    def __init__(
        self,
        abscissa: np.array,
        weights: np.array,
        overlap: np.array = None,
    ):
        """
        Constructs the Schrödinger equation in a basis of Lagrange Legendre
        functions
        """
        self.nbasis = len(abscissa)
        assert len(abscissa) == len(weights)
        self.abscissa = abscissa
        self.weights = weights

        if overlap is None:
            self.overlap = np.diag(np.ones(self.nbasis))
        else:
            self.overlap = overlap

    def kinetic_operator_element(
        self, n: np.int32, m: np.int32, a: np.float64, l: np.int32
    ):
        """
        @returns the (n,m)th matrix element for the kinetic energy + Bloch
        operator at channel radius a = k*r with orbital angular momentum l
        """
        assert n <= self.nbasis and n >= 1
        assert m <= self.nbasis and m >= 1

        xn, xm = self.abscissa[n - 1], self.abscissa[m - 1]
        N = self.nbasis

        if n == m:
            # Eq. 3.128 in [Baye, 2015], scaled by 1/E and with r->s=kr
            centrifugal = l * (l + 1) / (a * xn) ** 2
            radial = (
                ((4 * N**2 + 4 * N + 3) * xn * (1 - xn) - 6 * xn + 1)
                / (3 * xn**2 * (1 - xn) ** 2)
                / a**2
            )
            return radial + centrifugal
        else:
            # Eq. 3.129 in [Baye, 2015], scaled by 1/E and with r->s=kr
            return (
                (-1.0) ** (n + m)
                * (
                    (N**2 + N + 1.0)
                    + (xn + xm - 2 * xn * xm) / (xn - xm) ** 2
                    - 1.0 / (1.0 - xn)
                    - 1.0 / (1.0 - xm)
                )
                / np.sqrt(xn * xm * (1.0 - xn) * (1.0 - xm))
                / a**2
            )

    def kinetic_matrix(self, a: np.float64, l: np.int32):
        r"""
        @returns the kinetic operator matrix in the Lagrange Legendre basis
        """
        F = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)
        for n in range(1, self.nbasis + 1):
            for m in range(n, self.nbasis + 1):
                F[n - 1, m - 1] = self.kinetic_operator_element(n, m, a, l)
        F = F + np.triu(F, k=1).T
        return F
