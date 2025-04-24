from fractions import Fraction
from sympy.physics.quantum.cg import CG, Wigner6j


def triangle_rule(I1: Fraction, I2: Fraction):
    steps = int(I1 + I2 - abs(I1 - I2))
    return [abs(I1 - I2) + n for n in range(0, steps + 1, 1)]


class Memoize:
    # https://stackoverflow.com/a/1988826/1142217
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


def clebsch(
    j1: Fraction, m1: Fraction, j2: Fraction, m2: Fraction, j3: Fraction, m3: Fraction
):
    """
    Computes <j1 m1 j2 m2 | j3 m3>.
    """
    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
    # https://mattpap.github.io/scipy-2011-tutorial/html/numerics.html
    return CG(j1, m1, j2, m2, j3, m3).doit().evalf()


def racah(a: Fraction, b: Fraction, c: Fraction, d: Fraction, e: Fraction, f: Fraction):
    """
    Computes the Racah W-coefficient W(a, b, c, d; e, f).
    """
    return Wigner6j(a, b, c, d, e, f).doit().evalf()


racah = Memoize(racah)
clebsch = Memoize(clebsch)
