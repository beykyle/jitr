import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from jax import Array
import numpy as np



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
        self.keys = keys
        self.quantum_num_dic = quantum_num_dic
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





@jax.jit
def update_local_diagonal_interaction(diag_vals: jnp.ndarray, nchannels: int) -> jnp.ndarray:
    """
    Returns a (nchannels*N, nchannels*N) matrix with the same diag(diag_vals) in each diagonal block.
    """
    N = diag_vals.shape[0]
    block = jnp.diag(diag_vals)                       # (N, N)
    eye = jnp.eye(nchannels)                          # (B, B)
    mat = jnp.kron(eye, block)                        # (B*N, B*N)
    return mat

#batch it over the first dimension, you just need to pass a batched array of diagonal values!
batched_insert_diag = jax.jit(
    jax.vmap(update_local_diagonal_interaction, in_axes=(0, None))
)



@jax.jit
def update_local_coupling_interaction(diag_vals: jnp.ndarray,
                                      scale_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Returns a (B*N, B*N) matrix where each (i, j) block = scale_matrix[i,j] * diag(diag_vals)
    
    diag_vals: (N,)
    scale_matrix: (B, B)
    """
    N = diag_vals.shape[0]
    B = scale_matrix.shape[0]

    block = jnp.diag(diag_vals)  # (N, N)
    scaled_blocks = scale_matrix[:, None, :, None] * block[None, :, None, :]  # (B, N, B, N)

    return scaled_blocks.reshape(B * N, B * N)

batched_insert_couplings = jax.jit(
    jax.vmap(update_local_coupling_interaction, in_axes=(0, 0))
)
