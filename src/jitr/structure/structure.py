from fractions import Fraction
from enum import Enum
from dataclasses import dataclass


class Parity(Enum):
    positive = True
    negative = False

    def __mul__(self, other):
        if not isinstance(other, Parity):
            return NotImplemented
        return Parity(self.value and other.value)


@dataclass
class Level:
    E: float
    I: Fraction
    pi: Parity

    def __iter__(self):
        return iter((self.E, self.I, self.pi))
