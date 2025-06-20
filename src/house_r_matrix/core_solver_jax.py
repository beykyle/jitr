from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from jax import device_put
from typing import Callable



class Core_Solver:
    def __init__(self,
                 free_matrix_batch : np.ndarray,
                 b : np.ndarray,
                 Hp_batch: np.ndarray,
                 Hpp_batch: np.ndarray,
                 Hm_batch: np.ndarray,
                 Hmp_batch: np.ndarray,
                 appended_couplings_batched: np.ndarray, 
                 hbar_2mu : float,
                 a : float,
                 nbasis : int,
                 nchannels : int,
                 nbatch : int,
                 n_int : int):
        
        devices = jax.devices()
        gpu_available = any(device.platform == 'gpu' for device in devices)
        if gpu_available:
            print("GPU available. Using GPU.")
            self.device = jax.devices("gpu")[0]
        else:
            print(" No GPU found. Using CPU.")
            self.device = jax.devices("cpu")[0]
        
        self.nbasis = nbasis
        self.nchannels = nchannels
        self.batch_size = nbatch
        self.hbar_2mu = hbar_2mu
        self.a = a
        self.n_int = n_int
        # send to device
        self.free_matrix_batch = device_put(np.array(free_matrix_batch, dtype=np.complex64) , device = self.device) # (nbatch, nchannels*nbasis, nchannels*nbasis)
        self.b = device_put(np.array(b, dtype=np.complex64), device = self.device)   # (nbasis,)
        self.Hp_batch = device_put(np.array(Hp_batch, dtype=np.complex64), device=self.device)  # (nbatch, nchannels, nchannels)
        self.Hpp_batch = device_put(np.array(Hpp_batch, dtype=np.complex64), device=self.device) # (nbatch, nchannels, nchannels)
        self.Hm_batch = device_put(np.array(Hm_batch, dtype=np.complex64), device=self.device)  # (nbatch, nchannels, nchannels)
        self.Hmp_batch = device_put(np.array(Hmp_batch, dtype=np.complex64), device=self.device)  # (nbatch, nchannels, nchannels)
        self.appended_couplings_batched = device_put(np.array(appended_couplings_batched, dtype=np.complex64), device=self.device)  # (n_int, nbatch, nchannels, nchannels)

        self.fn_core, self.fn_interaction = self.precompile_functions()
    

    """
    class Core_Solver:
    Solves the coupled-channel Schrödinger equation using the R-matrix method for a homogeneous
    batched set of coupled-channel blocks in the GPU (when available). It sends to device
    all the precomputed matrices and quantities and computes the S-matrix in a fully vectorized manner.
    @paramters:
        - free_matrix_batch: (nbatch, nchannels*nbasis, nchannels*nbasis) : precomputed free matrix
        - b: (nbasis,) : vector of basis functions at the channel radius
        - Hp_batch: (nbatch, nchannels, nchannels) : H^+ Coulomb functions at the channel radius r
        - Hpp_batch: (nbatch, nchannels, nchannels) : derivative of the H^+ Coulomb functions at the channel radius r
        - Hm_batch: (nbatch, nchannels, nchannels) : H^- Coulomb functions at the channel radius r
        - Hmp_batch: (nbatch, nchannels, nchannels) : derivative of the H^- Coulomb functions at the channel radius r
        - appended_couplings_batched: (n_int, nbatch, nchannels, nchannels) : precomputed appended couplings for each interaction
                                     the first axis is the number of interactions that wish to be included.
        - hbar_2mu: float : ħ² / 2μ, where μ is the reduced mass of the system
        - a: float : channel radius
        - nbasis: int : number of basis functions
        - nchannels: int : number of channels
        - nbatch: int : number of batches
        - n_int: int : number of interactions (currently not used, but reserved for future use)

    @functions:
        solver: computes the S-matrix for a given appended block array.
        generate_coupling_interaction: generates the coupling interaction matrix for the appended couplings and
                                       the appended block array.
        make_coupling_kernel: creates a kernel function to update the local coupling interaction.
        smatrix_fn: computes the S-matrix for a given set of parameters.
        make_compiled_smatrix_fn: creates a compiled version of the smatrix_fn with fixed static arguments.
        precompile_functions: precompiles the core and interaction functions for faster execution.
        generate_dummy_arrays: generates dummy arrays for precompilation with correct shape.

    """
    
        

    @partial(jax.jit, static_argnames=())  # no static args in this case
    def solver(self, appended_block_arr: np.ndarray) -> Array:

        """
        Computes the S-matrix for a given appended block array.
        @parameters:
            appended_block_arr: (n_int, nbatch, nbasis, nbasis) : on-the-fly computed blocks of the radial 
                                 part of the interaction, the elements of the first axis must be the interaction
                                blocks that correspond to the appended_couplings_batched. 
            
        @returns:
            S_batch: (nbatch, nchannels, nchannels)
        """

        #send to device
        app_block_jax = device_put(np.array(appended_block_arr, dtype=np.complex64), device=self.device)  # (nbatch, nbasis, nbasis)

        A_batch = self.free_matrix_batch + Core_Solver.generate_coupling_interaction(
            self.appended_couplings_batched, app_block_jax, self.fn_interaction)
        
        S_batch = self.fn_core(A_batch, self.b, self.Hp_batch, self.Hpp_batch, self.Hm_batch, self.Hmp_batch)
        
        
        return S_batch  # (nbatch, nchannels, nchannels)
    
    
    
    @staticmethod
    @jax.jit
    def generate_coupling_interaction(appended_couplings_jax: Array,
                                    appended_block_jax: Array,
                                    precomp_fill_fn: Callable) -> Array:
        """
        appended_couplings_jax: (n_int, batch, nchannels, nchannels)
        appended_block_jax:     (n_int, batch, nbasis, nbasis)
        precomp_fill_fn: helper function to generate the r-matrices vmapped over axis 0 (batch)
        """
        # vmap over B (outer batch), rely on precomp_fill_fn to handle inner batch
        per_block_V = jax.vmap(precomp_fill_fn)(appended_block_jax, appended_couplings_jax)  # (B, m*n, m*n)

        # Sum over B
        V_total = jnp.sum(per_block_V, axis=0)

        return V_total # (batch_size, nchannels*nbasis, nchannels*nbasis)
        
    
    @staticmethod
    def make_coupling_kernel(nchannels: int, nbasis: int):
        def update_local_coupling_interaction(block: Array,
                                            couplings_matrix: Array) -> Array:
            """
            Returns a (nchannels*nbasis, nchannels*nbasis) matrix where each (i, j) block = couplings_matrix[i,j] * diag(diag_vals)
            """
            scaled_blocks = couplings_matrix[:, None, :, None] * block[None, :, None, :]  # (nch, nb, nch, nb)
            return scaled_blocks.reshape(nchannels * nbasis, nchannels * nbasis)

        # Vectorize over batch dim 0 for both inputs
        return jax.jit(jax.vmap(update_local_coupling_interaction, in_axes=(0, 0)))

    
    @staticmethod
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


    @staticmethod
    def make_compiled_smatrix_fn(nchannels: int,
                                nbasis: int,
                                a: float,
                                hbar_2mu: float):
        """
        Returns a version of smatrix_fn with fixed static arguments.
        """
        @jax.jit
        def compiled_fn(A_batch: Array, b: Array, Hp: Array, Hpp: Array, Hm: Array, Hmp: Array) -> Array:
            return Core_Solver.smatrix_fn(A_batch, b, Hp, Hpp, Hm, Hmp, nchannels, nbasis, a, hbar_2mu)
        return compiled_fn
    

    def precompile_functions(self):

        A_dummy, b_dummy, Hp_dummy, Hpp_dummy, Hm_dummy, Hmp_dummy, couplings_dummy, block_dummy = self.generate_dummy_arrays()
        # Make and compile specialized function
        fn_core = Core_Solver.make_compiled_smatrix_fn(self.nchannels, self.nbasis, self.a, self.hbar_2mu)
        _ = fn_core(A_dummy, b_dummy, Hp_dummy, Hpp_dummy, Hm_dummy, Hmp_dummy )  # trigger compilation
        fn_interaction = Core_Solver.make_coupling_kernel(self.nchannels, self.nbasis)
        _  = fn_interaction(block_dummy, couplings_dummy)  # trigger compilation

        return fn_core, fn_interaction
    
    
    def generate_dummy_arrays(self) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        """
        Generate dummy arrays for precompilation with correct shape.
        """
        dim = self.nchannels * self.nbasis
        A_dummy = jnp.tile(jnp.eye(dim, dtype=jnp.complex64), (self.batch_size, 1, 1))
        b_dummy = jnp.ones((self.nbasis,), dtype=jnp.complex64)
        Hp_dummy = jnp.broadcast_to(jnp.eye(self.nchannels, dtype=jnp.complex64),
                            (self.batch_size, self.nchannels, self.nchannels))
        Hpp_dummy = jnp.broadcast_to(jnp.eye(self.nchannels, dtype=jnp.complex64),
                            (self.batch_size, self.nchannels, self.nchannels))
        Hm_dummy = jnp.broadcast_to(jnp.eye(self.nchannels, dtype=jnp.complex64),
                            (self.batch_size, self.nchannels, self.nchannels))
        Hmp_dummy = jnp.broadcast_to(jnp.eye(self.nchannels, dtype=jnp.complex64),
                            (self.batch_size, self.nchannels, self.nchannels))
        couplings_dummy = jnp.broadcast_to(jnp.eye(self.nchannels, dtype=jnp.complex64),
                            (self.batch_size, self.nchannels, self.nchannels))
        block_dummy = jnp.broadcast_to(jnp.eye(self.nbasis, dtype=jnp.complex64),
                            (self.batch_size, self.nbasis, self.nbasis))
        
        return A_dummy, b_dummy, Hp_dummy, Hpp_dummy, Hm_dummy, Hmp_dummy, couplings_dummy, block_dummy