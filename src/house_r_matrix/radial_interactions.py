#import numpy as np
import jax.numpy as jnp
import jax.numpy as jnp
from jax import jit, vmap
from jax import Array


MAX_ARG = jnp.log(1 / 1e-16)  # ~36.8


def woods_saxon_safe(r, R, a):
    """
    JAX-compatible Woods-Saxon shape:
    f(r) = 1 / (1 + exp((r - R)/a)) with safe overflow handling.
    """
    x = (r - R) / a
    x = jnp.asarray(x, dtype=jnp.complex64)

    # Apply masking logic without mutation
    safe_exp = jnp.where(jnp.real(x) <= MAX_ARG, 1.0 / (1.0 + jnp.exp(x)), 0.0)

    return safe_exp


def woods_saxon_prime_safe(r, R, a):
    """
    JAX-compatible derivative of the Woods-Saxon shape:
    f'(r) = -exp(x) / [a * (1 + exp(x))^2]
    """
    x = (r - R) / a
    x = jnp.asarray(x, dtype=jnp.complex64)

    expx = jnp.exp(x)
    num = -expx
    denom = a * (1.0 + expx) ** 2
    df = num / denom

    # Use where to safely mask overflow
    df_safe = jnp.where(jnp.real(x) <= MAX_ARG, df, 0.0)

    return df_safe

@jit
def woods_saxon_potential(r, V0, R, a):
    """
    Full Woods-Saxon potential V(r) = -V0 * f(r)
    """
    return - V0 * woods_saxon_safe(r, R, a)

@jit
def woods_saxon_prime(r : Array, 
                      V0 : float, 
                      R : float, 
                      a :float):
    """
    Derivative V'(r) = V0 * f'(r)
    """
    return V0 * woods_saxon_prime_safe(r, R, a)

@jit
def woods_saxon_deformed_interaction(r : Array, 
                                     delta_lambda : float, 
                                     V0 : float, 
                                     R : float, 
                                     a : float):
    """
    First-order deformed Woods-Saxon interaction:
    V_def(r) = V0 * δλ / sqrt(4π) * f'(r)
    """
    prefactor = delta_lambda / jnp.sqrt(4 * jnp.pi)
    return V0 * prefactor * woods_saxon_prime_safe(r, R, a)


