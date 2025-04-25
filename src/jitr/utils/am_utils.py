from fractions import Fraction
from sympy import S as _to_symbolic_


def _int_or_frac_to_symbol_(j):
    jj = Fraction(j)
    return _to_symbolic_(jj.numerator) / jj.denominator


class _memoize_:
    # https://stackoverflow.com/a/1988826/1142217
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]
