from fractions import Fraction
from sympy.physics.quantum.cg import CG


class Memoize:
    # https://stackoverflow.com/a/1988826/1142217
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


def clebsch_gordan(
    j1: Fraction, m1: Fraction, j2: Fraction, m2: Fraction, j3: Fraction, m3: Fraction
):
    """
    Computes <j1 m1 j2 m2 | j3 m3>, where all spins are given as Fractions
    """
    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
    # https://mattpap.github.io/scipy-2011-tutorial/html/numerics.html
    return CG(j1, m1, j2, m2, j3, m3).doit().evalf()


clebsch_gordan = Memoize(clebsch_gordan)
