""" Evaluate polynomials and their derivatives."""
import numpy as np

def poly1d(x, coeffs, start_i=0):
    """
    coeffs[i] = coefficient of x^i, shape (N,)
    x: float or array of shape (M,) → returns float or array of shape (M,)
    """
    x = np.asarray(x, dtype=float)
    i = np.arange(start_i, len(coeffs) + start_i)
    return (x[..., None] ** i) @ coeffs
    #       └─────────────────┘
    #       scalar → (N,)  ─→  scalar
    #       (M,)   → (M,N) ─→  (M,)


def poly2d(x, y, coeffs, start_i=0, start_j=0):
    """
    coeffs[i,j] = coefficient of x^i y^j, shape (N,N)
    x, y: floats or arrays of shape (M,) → returns float or array of shape (M,)
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = np.arange(start_i, coeffs.shape[0] + start_i)
    j = np.arange(start_j, coeffs.shape[1] + start_j)
    xi = x[..., None] ** i  # scalar → (N,)  |  (M,) → (M,N)
    yj = y[..., None] ** j  # scalar → (N,)  |  (M,) → (M,N)
    return np.einsum("...i,ij,...j->...", xi, coeffs, yj)
    #                 └──────────────────────────────┘
    #   scalar: '  i,ij,  j-> '  →  scalar
    #   array:  'mi,ij, mj->m'  →  (M,)


def poly1d_deriv(coeffs, start_i=0):
    """∂/∂x of Σ_i c[i] x^(start_i+i).  Returns (new_coeffs, new_start_i)."""
    i = np.arange(start_i, start_i + len(coeffs))
    out = coeffs * i
    if start_i == 0:
        return out[1:], 0  # the constant term differentiates to zero — drop it
    return out, start_i - 1


def poly2d_deriv(coeffs, start_i=0, start_j=0, wrt="y"):
    """∂/∂x or ∂/∂y of Σ_{i,j} c[i,j] x^(start_i+i) y^(start_j+j).

    Returns (new_coeffs, new_start_i, new_start_j) — drop straight back into poly2d.
    """
    if wrt == "y":
        j = np.arange(start_j, start_j + coeffs.shape[1])
        out = coeffs * j[None, :]
        if start_j == 0:
            return out[:, 1:], start_i, 0
        return out, start_i, start_j - 1
    if wrt == "x":
        i = np.arange(start_i, start_i + coeffs.shape[0])
        out = coeffs * i[:, None]
        if start_i == 0:
            return out[1:, :], 0, start_j
        return out, start_i - 1, start_j
    raise ValueError(f"wrt must be 'x' or 'y', got {wrt!r}")
