import numpy as np
from cc_constants import CC_Constants
import scipy.special as sc
from mpmath import coulombf, coulombg


from scipy.special import spherical_jn, spherical_yn

def F(r, ell, k):
    rho = k * r
    return rho * spherical_jn(int(ell), rho)

def G(r, ell, k):
    rho = k * r
    return  rho * spherical_yn(int(ell), rho)

def F_prime(r, ell, k):
    rho = k * r
    return k * (spherical_jn(int(ell), rho)) + k * rho * spherical_jn(int(ell), rho, derivative=True)

def G_prime(r, ell, k):
    rho = k * r
    return  k * (spherical_yn(int(ell), rho)) + k * rho * spherical_yn(int(ell), rho, derivative=True)

def H_plus(r, ell, k):
    return G(r, ell, k) + 1j * F(r, ell, k)

def H_minus(r, ell, k):
    return G(r, ell, k) - 1j * F(r, ell, k)

def H_plus_prime(r, ell, k):
    return G_prime(r, ell, k) + 1j * F_prime(r, ell, k)
 
def H_minus_prime(r, ell, k):
    return G_prime(r, ell, k) - 1j* F_prime(r, ell, k)

def free_solution(r, ell, k):
    """
    Free solution to the radial Schrödinger equation.
    """
    return H_minus(r, ell, k) - H_plus(r, ell, k)

def free_solution_prime(r, ell, k):
    """
    Derivative of the free solution to the radial Schrödinger equation.
    """
    return H_minus_prime(r, ell, k) - H_plus_prime(r, ell, k)



class CC_Asymptotics:

    @staticmethod
    def sqrt_k_matrix_per_shape(homogeneous_k_arr):
        """
        Returns wave number arrays (k) and their inverses (1/k) for each parity block.
        """

        sqrt_k_matrices = []
        sqrt_k_inv_matrices = []
        for k_arr in homogeneous_k_arr:
            k_arr = np.asarray(k_arr, dtype=np.complex128)
            sqrt_k = np.diag(np.sqrt(k_arr))
            sqrt_k_inv = np.diag(1.0 / np.sqrt(k_arr))

            sqrt_k_matrices.append(sqrt_k)
            sqrt_k_inv_matrices.append(sqrt_k_inv)

        return sqrt_k_matrices, sqrt_k_inv_matrices

    @staticmethod
    def CC_Bessel_per_shape(r, homogeneous_l_arr, homogeneous_k_arr):
        """
        Computes the Bessel functions (H±) and their derivatives (H±') at fixed radius r
        for all coupled-channel blocks, split by parity.
        Returns:
            (Hp_pos, Hpp_pos, Hm_pos, Hmp_pos,
             Hp_neg, Hpp_neg, Hm_neg, Hmp_neg)
        Each of these is a list of diagonal matrices (np.array of shape (n_channels, n_channels))
        """
        def compute_block(l_block, k_block):

            #s = k * r  # dimensionless radial variable
            Hp  = [H_plus(r, li, ki)        for li, ki in zip(l_block, k_block)]
            Hpp = [H_plus_prime(r, li, ki)  for li, ki in zip(l_block, k_block)]
            Hm  = [H_minus(r, li, ki)       for li, ki in zip(l_block, k_block)]
            Hmp = [H_minus_prime(r, li, ki) for li, ki in zip(l_block, k_block)]
            return Hp, Hpp, Hm, Hmp

        Hp, Hpp, Hm, Hmp = [], [], [], []

        for l_block, k_block in zip(homogeneous_l_arr, homogeneous_k_arr):
            Hp_i, Hpp_i, Hm_i, Hmp_i = compute_block(l_block, k_block)
            Hp.append(np.diag(Hp_i))
            Hpp.append(np.diag(Hpp_i))
            Hm.append(np.diag(Hm_i))
            Hmp.append(np.diag(Hmp_i))


        return np.array(Hp), np.array(Hpp), np.array(Hm), np.array(Hmp)
    
    @staticmethod
    def generate_sqrt_mom_batched(k_batched, keys):
        """
        Computes the square root of the wave number matrices for each coupled-channel block.
        Returns:
            sqrt_k_dic: Dictionary of square root matrices for each key.
            sqrt_k_inv_dic: Dictionary of inverse square root matrices for each key.
        """

        sqrt_k_dic, sqrt_k_inv_dic = {}, {}
        for key in keys:
            sqrt_k_dic[key], sqrt_k_inv_dic[key] = CC_Asymptotics.sqrt_k_matrix_per_shape(k_batched[key])
        
        return sqrt_k_dic, sqrt_k_inv_dic
    
    @staticmethod
    def generate_bessel_batched(r, l_batched, k_batched, keys):
        """
        Computes the Bessel functions (H±) and their derivatives (H±') at fixed radius r
        for all coupled-channel blocks, split by parity.
        Returns:
            (Hp_pos, Hpp_pos, Hm_pos, Hmp_pos,
             Hp_neg, Hpp_neg, Hm_neg, Hmp_neg)
        Each of these is a list of diagonal matrices (np.array of shape (n_channels, n_channels))
        """
        Hp_dic, Hpp_dic, Hm_dic, Hmp_dic = {}, {}, {}, {}

        for key in keys:
            Hp_dic[key], Hpp_dic[key], Hm_dic[key], Hmp_dic[key] = CC_Asymptotics.CC_Bessel_per_shape(r, l_batched[key], k_batched[key])

        return Hp_dic, Hpp_dic, Hm_dic, Hmp_dic

        



