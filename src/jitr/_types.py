"""Shared type aliases used throughout jitr."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

#: 1-D array of 64-bit floats.
FloatArray: TypeAlias = npt.NDArray[np.float64]

#: 1-D (or N-D) array of 128-bit complex numbers.
ComplexArray: TypeAlias = npt.NDArray[np.complex128]

#: A real scalar or a real-valued NumPy array — used for radial grid inputs and
#: real-valued potential outputs.
ArrayOrScalar: TypeAlias = float | FloatArray

#: A complex scalar or a complex-valued NumPy array — used for complex potential
#: outputs.
PotentialArray: TypeAlias = complex | ComplexArray
