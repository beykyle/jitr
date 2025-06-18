import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict, Sequence
from jax import Array
import numpy as np
from functools import partial



@jax.jit
def rmatrix_fn(A_batch: Array,
               b: Array,
               nchannels: int,
               nbasis: int,
               a: float,
               hbar_2mu: float) -> Array:
    """
    Computes the batched R-matrix:
        R_ij = hbar²/(2μ) * a * b_m * C_imjn * b_n
    where C = A⁻¹.
    See Eq. (15) in Descouvemont, 2016.

    Args:
        A_batch: (batch_size, nchannels*nbasis, nchannels*nbasis)
        b: (nbasis,)
        nchannels, nbasis: channel and basis size
        a: channel radius
        hbar_2mu: ħ² / 2μ

    Returns:
        R_batch: (batch_size, nchannels, nchannels)
    """
    C_batch = jnp.linalg.inv(A_batch)
    C_blocks = C_batch.reshape(C_batch.shape[0], nchannels, nbasis, nchannels, nbasis)
    C_blocks = jnp.transpose(C_blocks, (0, 1, 3, 2, 4))  # (B, i, j, m, n)

    R_batch = hbar_2mu * a * jnp.einsum('m,bijnm,n -> bij', b, C_blocks, b)

    return R_batch



def make_compiled_rmatrix_fn(nchannels: int,
                             nbasis: int,
                             a: float,
                             hbar_2mu: float):
    """
    Returns a version of rmatrix_fn with fixed static arguments.
    """
    @jax.jit
    def compiled_fn(A_batch: Array, b: Array) -> Array:
        return rmatrix_fn(A_batch, b, nchannels, nbasis, a, hbar_2mu)
    return compiled_fn


class R_matrix_solver_with_warmup:
    def __init__(self,
                 keys: np.ndarray,
                 quantum_num_dic: Dict,
                 nbasis: int,
                 a: float,
                 hbar_2mu: float):
        self.keys = jnp.asarray(keys, dtype=jnp.int32)
        self.quantum_num_dic = {k: jnp.array(v, dtype=jnp.complex64) for k, v in quantum_num_dic.items()}
        self.nbasis = nbasis
        self.a = a
        self.hbar_2mu = hbar_2mu
        self._precompiled_rmatrix_funcs = self.precompile_functions()

    
    def generate_dummy_arrays(self,
                              nchannels: int,
                              nbasis: int,
                              batch_size: int) -> tuple[Array, Array]:
        """
        Generate dummy arrays for precompilation with correct shape.
        """
        dim = nchannels * nbasis
        A_dummy = jnp.tile(jnp.eye(dim, dtype=jnp.complex64), (batch_size, 1, 1))
        b_dummy = jnp.ones((nbasis,), dtype=jnp.complex64)
        return A_dummy, b_dummy

    
    def precompile_functions(self) -> Dict[str, callable]:
        compiled_funcs = {}
        for key in self.keys:
            quantum_matrix = jnp.array(self.quantum_num_dic[key], dtype=jnp.complex64)
            batch_size, nchannels, _ = quantum_matrix.shape
            A_dummy, b_dummy = self.generate_dummy_arrays(nchannels, self.nbasis, batch_size)
            # Make and compile specialized function
            fn = make_compiled_rmatrix_fn(nchannels, self.nbasis, self.a, self.hbar_2mu)
            _ = fn(A_dummy, b_dummy)  # trigger compilation
            compiled_funcs[key] = fn

        return compiled_funcs

    def evaluate(self, key: int, A_batch: Array, b: Array) -> Array:
        """
        Use the precompiled function for the given key.
        """
        return self._precompiled_rmatrix_funcs[key](A_batch, b)
    



@jax.jit
def smatrix_fn(A_batch: Array,
               b : Array,
               Hp : Array,
               Hpp : Array,
               Hm : Array,
               Hmp : Array,
               nchannels: int,
               nbasis : int,
               a : float,
               hbar_2mu : float):
    """
    Computes the batched S-matrix:
        R_ij = hbar²/(2μ) * a * b_m * C_imjn * b_n
        S_ij = [Hp_ij + Hpp_ik @ R_kj]^-1 @ [Hm_ij + Hmp_ik @ R_kj]
    where C = A⁻¹.
    See Eq. (15) in Descouvemont, 2016.

    Args:
        A_batch: (batch_size, nchannels*nbasis, nchannels*nbasis)
        b: (nbasis,)
        Hp : (batch_size, nchannels, nchannels) : H^+ Coulomb funtions at the channel radius r 
        Hpp : (batch_size, nchannels, nchannels) : derivative of the H^+ Coulomb funtions at the channel radius r
        Hm : (batch_size, nchannels, nchannels) : H^- Coulomb funtions at the channel radius r
        Hmp : (batch_size, nchannels, nchannels) : derivative of the H^- Coulomb funtions at the channel radius r
        nchannels, nbasis: channel and basis size
        a: channel radius
        hbar_2mu: ħ² / 2μ

    Returns:
        S_batch: (batch_size, nchannels, nchannels)
    """

    C_batch = jnp.linalg.inv(A_batch)
    C_blocks = C_batch.reshape(C_batch.shape[0], nchannels, nbasis, nchannels, nbasis)
    C_blocks = jnp.transpose(C_blocks, (0, 1, 3, 2, 4))  # (B, i, j, m, n)

    R_batch = hbar_2mu * a * jnp.einsum('m,bijnm,n -> bij', b, C_blocks, b)

    Zp = Hp - a * jnp.einsum('bij,bjk->bik', R_batch, Hpp)
    Zm = Hm - a * jnp.einsum('bij,bjk->bik', R_batch, Hmp)  

    S_batch = jnp.linalg.solve(Zp, Zm)

    return S_batch



def make_compiled_smatrix_fn(nchannels: int,
                             nbasis: int,
                             a: float,
                             hbar_2mu: float):
    """
    Returns a version of smatrix_fn with fixed static arguments.
    """
    @jax.jit
    def compiled_fn(A_batch: Array, b: Array, Hp: Array, Hpp: Array, Hm: Array, Hmp: Array) -> Array:
        return smatrix_fn(A_batch, b, Hp, Hpp, Hm, Hmp, nchannels, nbasis, a, hbar_2mu)
    return compiled_fn




class S_matrix_solver_with_warmup:
    def __init__(self,
                 keys: np.ndarray,
                 quantum_num_dic: Dict,
                 nbasis: int,
                 a: float,
                 hbar_2mu: float):
        self.keys = keys
        self.quantum_num_dic = quantum_num_dic
        self.nbasis = nbasis
        self.a = a
        self.hbar_2mu = hbar_2mu
        self._precompiled_smatrix_funcs = self.precompile_functions()

    
    def generate_dummy_arrays(self,
                              nchannels: int,
                              nbasis: int,
                              batch_size: int) -> tuple[Array, Array, Array, Array, Array, Array]:
        """
        Generate dummy arrays for precompilation with correct shape.
        """
        dim = nchannels * nbasis
        A_dummy = jnp.tile(jnp.eye(dim, dtype=jnp.complex64), (batch_size, 1, 1))
        b_dummy = jnp.ones((nbasis,), dtype=jnp.complex64)
        Hp_dummy = jnp.broadcast_to(jnp.eye(nchannels, dtype=jnp.complex64),
                            (batch_size, nchannels, nchannels))
        Hpp_dummy = jnp.broadcast_to(jnp.eye(nchannels, dtype=jnp.complex64),
                            (batch_size, nchannels, nchannels))
        Hm_dummy = jnp.broadcast_to(jnp.eye(nchannels, dtype=jnp.complex64),
                            (batch_size, nchannels, nchannels))
        Hmp_dummy = jnp.broadcast_to(jnp.eye(nchannels, dtype=jnp.complex64),
                            (batch_size, nchannels, nchannels))
        return A_dummy, b_dummy, Hp_dummy, Hpp_dummy, Hm_dummy, Hmp_dummy

    
    def precompile_functions(self) -> Dict[str, callable]:
        compiled_funcs = {}
        for key in self.keys:
            quantum_matrix = jnp.array(self.quantum_num_dic[key], dtype=jnp.complex64)
            batch_size, nchannels, _ = quantum_matrix.shape
            A_dummy, b_dummy, Hp_dummy, Hpp_dummy, Hm_dummy, Hmp_dummy = self.generate_dummy_arrays(nchannels, self.nbasis, batch_size)
            # Make and compile specialized function
            fn = make_compiled_smatrix_fn(nchannels, self.nbasis, self.a, self.hbar_2mu)
            _ = fn(A_dummy, b_dummy, Hp_dummy, Hpp_dummy, Hm_dummy, Hmp_dummy )  # trigger compilation
            compiled_funcs[key] = fn

        return compiled_funcs

    def evaluate(self, key:int, A_batch: Array, b: Array, Hp: Array, 
                 Hpp: Array, Hm: Array, Hmp: Array) -> Array:
        """
        Use the precompiled function for the given key.
        """
        return self._precompiled_smatrix_funcs[key](A_batch, b, Hp, Hpp, Hm, Hmp)







def make_diagonal_kernel(nchannels: int):
    #@partial(jax.jit, static_argnums=1)  # `nchannels` must be static
    def update_local_diagonal_interaction(diag_vals: jnp.ndarray, nchannels: int) -> jnp.ndarray:
        """
        @parameters:
           diag_vals: an array of diagonal values of shape (nbasis,)
           nchannels: the number of channels (int)
        Returns a (nchannels*N, nchannels*N) matrix with the same diag(diag_vals) in each diagonal block.
      """
        block = jnp.diag(diag_vals.astype(jnp.complex64))                       # (nbasis, nbasis)
        eye = jnp.eye(nchannels, dtype=jnp.complex64)                          # (nchannels, nchannels)
        mat = jnp.kron(eye, block)                        # (nchannels*nbasis, nchannels*nbasis)
        return mat

    return jax.jit(jax.vmap(partial(update_local_diagonal_interaction, nchannels=nchannels), in_axes=0))

def batched_local_diagonal(keys: Sequence[int]):
    f"""
    @parameters:
         keys: an array of integers corresponding to the nchannels of each batch
     @returns:
        local_diagonal_kernels: a dictionary of fused kernel functions (vmap) with fixed nchannels
        and nbasis, where the operation is a batched along the first axis. Such that the input
        to the stored function is an array of diagonal values of shape diag_values= (batch_size, nbasis) 
        and an array of scale_matrix = (batch_size, nchannels, nchannels)
    """
    local_diagonal_kernels = {}
    for nch in keys:
        local_diagonal_kernels[int(nch)] = make_diagonal_kernel(int(nch))
    return local_diagonal_kernels



def make_coupling_kernel(nchannels: int, nbasis: int):
    def update_local_coupling_interaction(diag_vals: jnp.ndarray,
                                          scale_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a (nchannels*nbasis, nchannels*nbasis) matrix where each (i, j) block = scale_matrix[i,j] * diag(diag_vals)
        """
        block = jnp.diag(diag_vals.astype(jnp.complex64))  # (nbasis, nbasis)
        scaled_blocks = scale_matrix[:, None, :, None] * block[None, :, None, :]  # (nch, nb, nch, nb)
        return scaled_blocks.reshape(nchannels * nbasis, nchannels * nbasis)

    # Vectorize over batch dim 0 for both inputs
    return jax.jit(jax.vmap(update_local_coupling_interaction, in_axes=(0, 0)))


def batched_local_couplings(keys: Sequence[int], nbasis: int) -> Dict[int, callable]:
    f"""
      @parameters:
          keys: an array of integers corresponding to the nchannels of each batch
          nbasis: the fixed number of lagrange basis functions
      @returns:
          local_coupling_kernels: a dictionary of fused kernel functions (vmap) with fixed nchannels
          and nbasis, where the operation is a batched along the first axis. Such that the input
          to the stored function is an array of diagonal values of shape diag_values= (batch_size, nbasis) 
          and an array of scale_matrix = (batch_size, nchannels, nchannels)
      """
    local_coupling_kernels = {}
    for nch in keys:
        local_coupling_kernels[int(nch)] = make_coupling_kernel(int(nch), nbasis)
    return local_coupling_kernels


