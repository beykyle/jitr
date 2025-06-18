import numpy as np
from kernel import Kernel



def block(matrix: np.array, block, block_size):
    """
    get submatrix with coordinates block from matrix, where
    each block is defined by block_size elements along each dimension
    """
    i, j = block
    n, m = block_size
    return matrix[i * n : i * n + n, j * m : j * m + m]


class Solver:
    r"""
    A Schrodinger equation solver using the R-matrix method on a Lagrange mesh
    """
    def __init__(
            self, 
            nbasis: np.int32,
            basis = "Legendre",
            **args,
    ):
        r""""
        Contructs an R-matrix solver on a Lagrange mesh
        @parameters:
                nbasis(int) : size of basis; e.g. number of quadrature points for integration
                ecom(float) : center of mass frame scattering energy
                basis(str) : what basis mesh to use
        """

        self.kernel = Kernel(nbasis, basis)

    def precompute_boundaries(self, a : np.float64):
        r"""
        Precompute boundary values of Lagrange basis functions for a set of channel radii
        @parameters:
                a(float) : channel radius
        """
        nbasis = self.kernel.quadrature.basis
        return np.array([
            self.kernel.f(n, a, a) for n in range(1, nbasis + 1)], 
            dtype = np.complex128
            )
    
    def get_channel_block(self, matrix: np.ndarray, i: np.int32, j:np.int32 = None):
        N = self.kernel.quadrature.nbasis
        if j is None:
            j = i
        return block(matrix, (i, j), (N, N))
    
    def kinetic_matrix(
            self,
            a: np.float64,
            l: np.ndarray,
            hbar_2mu: np.float64,
    ):
        r"""
        @return:
            kinetic matrix (np.ndarray): the full (Nb)x(Nb) kinetic energy matrix
        @parameters:
            a : channel radius (r_f)
            l : orbital angular momentum in each channel 
            hbar_2mu : reduced mass of the system. currently restricted to systems
                       in the same mass paritition
        """
        Nb = self.kernel.quadrature.nbasis
        Nch = np.size(l)
        sz = Nb * Nch
        F = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(Nch):
            Fij = hbar_2mu * self.kernel.quadrature.kinetic_matrix(a, l[i])
            F[(i * Nb) : (i + 1) * Nb, (i * Nb) : (i + 1) * Nb] += Fij
        
        return F
    
    def energy_matrix(
            self,
            a: np.float64,
            l: np.ndarray,
            E: np.ndarray
    ):
        r"""
        @returns:
            energy_matrix (np.ndarray) : the full (nchannels x nbasis)^2
            Diagonal in channel space but possibly not in Lagrange
            space. (E_i < f_n | f_m >)
        @parameters:
            a : channel radius (r_f)
            l : orbital angular momentum in each channel
            E : channel energies (E_i)
        """
        assert np.size(l) == np.size(E) 
        
        Nb = self.kernel.quadrature.nbasis
        Nch = np.size(l)
        sz = Nb * Nch
        F = np.zeros((sz, sz), dtype=np.complex128)
        
        for i in range(Nch):
            F[(i * Nb) : (i + 1) * Nb, (i * Nb) : (i + 1) * Nb] += ( self.kernel.overlap * E[i])

        return F
    
    def free_matrix(
            self,
            a: np.float64,
            l: np.ndarray,
            E: np.ndarray,
            hbar_2mu: np.float64,
            coupled : bool = True
    ):
        
        """
        Precompute the free matrix (kinetic + energy), whic only depends on the
        channel orbital angular momenta l and dimensionless channel radius a
        @parameters:
            a : channel radius (r_f)
            l : orbital angular momentum in each channel
            E : channel energies (E_i)
            hbar_2mu : reduced mass of the system. currently restricted to systems
                       in the same mass partition
            coupled : whether to return the full matrix or just the block
                      diagonal elements  (elements off of the channel diagonal
                      are all 0 for the free matrix). If False, returna list of 
                      Nch (Nb, Nb) matrices, where Nch is the number of channels
                      and Nb is the number of basis elements, otherwise returns
                      the full (Nch * Nb, Nch * Nb) matrix
        """

        free_matrix = self.kinetic_matric(a, l, hbar_2mu) + self.energy_matrix(a, l, E)

        if coupled:
            return free_matrix
        else: 
            return [self.get_channel_block(free_matrix, i) for i in range(l.size)]
        

    def interaction_matrix(
            self,
            a: np.float64,
            couplings : np.ndarray,
            local_interaction = None,
            local_args = None,
            nonlocal_interaction = None,
            nonlocal_args = None,
    ):
        r"""
        Returns the full (Nch * Nb,  Nch * Nb) interaction in the Lagrange basis, where
        each channel is an (Nb, Nb) block (Nb being the basis size), and there are
        Nch * Nch such blocks, for N channels. Uses radial coordinate r. 
        @ parameters:
            a (float): channel radius (r_f)
            couplings (array): an array of geometric matrix elements for each
                        coupled block of waves.
            local_interaction (callable): the local potential, a function of r, *args
            local_args : the arguments that get passed into local interaction
            nonlocal_interaction (callable) : the nonlocal potential, a function of r, r' and *args
            nonlocal_args : the args that get passed into nonlocal interaction
        """

        # allocate matrix to store the full interaction in the Lagrange basis
        nb = self.kernel.quadrature.nbasis
        nch = couplings.shape[0]
        sz = nb * nch
        V = np.zeros((sz, sz), dtype=np.complex128)

        #scaling
        channel_radius_r = a

        if local_interaction is not None:
            #matrix local just gives the diagonal elements of each block 
            #compute this once since it is the same for all channels, only scaled by 
            #the geometric couplings
            Vl_diag = np.diag(self.kernel.matrix_local(
                local_interaction, channel_radius_r, args = local_args
            ))

            #manually put them in the diagonal of each block and multiply by the coupling matrix
            #elements
            for i in range(nch):
                for j in range(nch):
                    V[i * nb : (i + 1) * nb, j * nb : (j + 1) * nb] = couplings[i, j] * Vl_diag

        ########### NOT TESTED FOR NONLOCAL
        if nonlocal_interaction is not None:
            #matrix nonlocal gives us an (Nch, Nch, Nb, Nb) array
            #which we can just reshape into the shape we want
            V += (
                self.kernel.matrix_nonlocal(
                    nonlocal_interaction, channel_radius_r, args=nonlocal_args
                )
                .reshape(nch, nch, nb, nb)
                .swapaxes(1, 2)
                .reshape(sz, sz, order="C")
            )
        ########### NOT TESTED FOR NONLOCAL

        return V