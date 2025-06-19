import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from rmatrix import Rmatrix_free
from cc_couplings import CC_Couplings
from cc_constants import CC_Constants
from typing import Callable
from jax_solver import batched_local_diagonal, batched_local_couplings
from jax_solver import S_matrix_solver_with_warmup

class Jax_Rmatrix(Rmatrix_free):
    """
    A Schrödinger equation solver using the R-matrix method on a Lagrange mesh,
    accelerated with JAX for GPU compatibility.
    """
    def __init__(self, mass_t, mass_p, E_lab, E_states, I_states, pi_xp, pi_xt, J_tot_max, nbasis: int, basis="Legendre", **args):
        super().__init__(nbasis, basis, **args)

        self.channel_radius = 40.0  # set this *before* it's used
        self.nbasis = nbasis
        # Set up coupling class and quantum numbers
        self.couplings_class = CC_Couplings(mass_t, mass_p, E_states, I_states, pi_xp, pi_xt, J_tot_max)
        self.E_dict, self.l_dict, self.k_dict = self.couplings_class.generate_energy_centrifugal_mom_batched(E_lab)
        
        # Reduced mass constant
        self.constants_class = CC_Constants(mass_t, mass_p, E_lab, I_states)
        self.h2_mass = self.constants_class.h2_mass

        # Scale abscissa and convert to JAX array
        abscissa = self.kernel.quadrature.abscissa * self.channel_radius
        self.abscissa_scaled = jnp.array(abscissa, dtype=jnp.complex64)

        # Couplings
        self.couplings_dict, self.keys = self.couplings_class.generate_couplings_batched(2.0)
        self.couplings_dict = {
            key: jnp.array(val, dtype=jnp.complex64) for key, val in self.couplings_dict.items()
        }

        # Free matrix (precomputed)
        self.free_matrix_batch = self.batched_free_matrix(
            self.channel_radius, self.l_dict, self.E_dict, self.h2_mass, self.keys
        )
        self.free_matrix_batch = {
            key: jnp.array(val, dtype=jnp.complex64) for key, val in self.free_matrix_batch.items()
        }

        #preallocate the kernels for fast update of diagonal and coupling interactions
        self.local_diagonal_kernels = batched_local_diagonal(self.keys)
        self.local_coupling_kernels = batched_local_couplings(self.keys, self.nbasis)

        #preallocate the batch size dictionary for fast filling
        self.batch_size_dict = self.batch_size()

        # JIT- pre-compiled solver for the S-matrix
        self.S_matrix_solver = S_matrix_solver_with_warmup(self.keys, self.couplings_class.batched_dict, 
                                                           nbasis, self.channel_radius,
                                                           self.h2_mass)




    @staticmethod
    @jit
    def matrix_local(f: Callable, abscissa_scaled: jnp.ndarray, *args) -> jnp.ndarray:
        """
        Apply local operator f(r, *args) to the Lagrange mesh points scaled by a.
        This is fully JAX-compatible and JIT-compilable.
        """
        r_vals = abscissa_scaled   # shape (nbasis,)
        return f(r_vals, *args)  # should return shape (nbasis,)
    
    def batch_size(self) -> dict:
        """
        Returns the size of the batch for JAX operations.
        """
        batch_sz_dict = {key: val.shape[0] for key, val in self.couplings_dict.items()}
        return batch_sz_dict
    
    def update_and_solve(self, jax_diag_interaction: Callable, jax_couplings_interaction: Callable, *args):
        abscissa_scaled = self.abscissa_scaled
        nbasis = self.nbasis
        for key in self.keys:
            free_matrix = self.free_matrix_batch[key]
            batch_sz = self.batch_size_dict[key]
            couplings = self.couplings_dict[key]
            diagonal_filling_kernel = self.local_diagonal_kernels[key]
            coupling_filling_kernel = self.local_coupling_kernels[key]
            pre_compiled_solver = self.S_matrix_solver._precompiled_smatrix_funcs[key]
            local_diag = jnp.broadcast_to(Jax_Rmatrix.matrix_local(jax_diag_interaction, abscissa_scaled, *args),
                                           (batch_sz, nbasis))
            
            
            
            






        



