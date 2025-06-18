
import numpy as np
from typing import Sequence

class CC_Constants:
    amu = 931.49432  # MeV/c^2
    hbarc = 197.32705  # MeV·fm
    finec = 137.03599  # Fine-structure constant

    def __init__(self, mass_t: float, mass_p: float, E_lab: complex, E_states: Sequence[complex]):
        self.mass_t = float(mass_t)
        self.mass_p = float(mass_p)
        self.E_lab = np.complex128(E_lab)
        self.E_states = np.array(E_states, dtype=np.complex128)

        self.mu = (self.mass_t * self.mass_p * self.amu**2) / ((self.mass_t + self.mass_p) * self.amu)
        self.h2_mass = np.complex128(self.hbarc**2 / (2 * self.mu))

    @property
    def reduced_mass(self) -> float:
        return self.mu

    @property
    def radial_m_coeff(self) -> float:
        return self.mass_t ** (1 / 3)

    def E_lab_to_COM(self) -> np.ndarray:
        return self.E_lab * (self.mass_t / (self.mass_t + self.mass_p)) - self.E_states

    def k(self) -> np.ndarray:
        E_COM = self.E_lab_to_COM()
        return np.sqrt(E_COM / self.h2_mass)