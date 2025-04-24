from fractions import Fraction
from enum import Enum
from dataclasses import dataclass


class Parity(Enum):
    positive = True
    negative = False

    def __mul__(self, other):
        """
        Multiplies the current Parity instance with another Parity instance or a number.

        Parameters
        ----------
            other (Parity or int/float): The other operand to multiply with.

        Returns
        -------
            Parity: The result of the multiplication.

        Raises
        ------
            TypeError: If the other operand is not a Parity instance or a number.
        """
        if isinstance(other, Parity):
            return Parity(self.value and other.value)
        else:
            f = 1 if self.value else -1
            return Parity(other * f >= 0)


@dataclass
class Level:
    """
    Represents a level with energy, spin, and parity.

    Parameters
    ----------
    E : float
        The energy of the level.
    I : Fraction
        The spin of the level.
    pi : Parity
        The parity of the level.
    """

    E: float
    I: Fraction
    pi: Parity

    def __iter__(self):
        return iter((self.E, self.I, self.pi))
