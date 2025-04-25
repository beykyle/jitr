from fractions import Fraction
from sympy.physics.wigner import racah as _racah_sympy_
from sympy.physics.quantum.cg import CG as _cg_sympy_

from .am_utils import _memoize_, _int_or_frac_to_symbol_


def triangle_rule(I1: Fraction, I2: Fraction):
    steps = int(I1 + I2 - abs(I1 - I2))
    return [abs(I1 - I2) + n for n in range(0, steps + 1, 1)]


def format_sbasis_channel_latex(l: int, Jtot: Fraction, S: Fraction):
    """ Formats a channel as ^{2s+1}L_{Jtot} """
    wave = lwaves[l]
    jfrac = f"{Jtot.numerator}/{Jtot.denominator}"
    s = f"{int(2*S + 1)}"
    return f"^{{{s}}} {wave} _{{{jfrac}}}"


def format_jbasis_channel_latex(l: int, Jtot: Fraction, J: Fraction):
    """ Formats a channel as ^{2J+1}L_{Jtot} """
    wave = lwaves[l]
    jfrac = f"{Jtot.numerator}/{Jtot.denominator}"
    j = f"{int(2*J + 1)}"
    return f"^{{{j}}} {wave} _{{{jfrac}}}"



def ClebschGordan(
    j1: Fraction, m1: Fraction, j2: Fraction, m2: Fraction, j3: Fraction, m3: Fraction
):
    """
    Computes <j1 m1 j2 m2 | j3 m3>.
    """
    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
    # https://mattpap.github.io/scipy-2011-tutorial/html/numerics.html
    return complex(
        _cg_sympy_(
            _int_or_frac_to_symbol_(j1),
            _int_or_frac_to_symbol_(m1),
            _int_or_frac_to_symbol_(j2),
            _int_or_frac_to_symbol_(m2),
            _int_or_frac_to_symbol_(j3),
            _int_or_frac_to_symbol_(m3),
        )
        .doit()
        .evalf()
    )


def RacahW(
    a: Fraction, b: Fraction, c: Fraction, d: Fraction, e: Fraction, f: Fraction
):
    """
    Computes the Racah W-coefficient W(a, b, c, d; e, f).
    """

    return complex(
        _racah_sympy_(
            _int_or_frac_to_symbol_(a),
            _int_or_frac_to_symbol_(b),
            _int_or_frac_to_symbol_(c),
            _int_or_frac_to_symbol_(d),
            _int_or_frac_to_symbol_(e),
            _int_or_frac_to_symbol_(f),
            prec=64,
        )
        .doit()
        .evalf()
    )


RacahW = _memoize_(RacahW)
ClebschGordan = _memoize_(ClebschGordan)


lwaves = [
    "s",
    "p",
    "d",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "q",
    "r",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
