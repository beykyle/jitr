import numpy as np
import scipy.constants as sc
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.wigner import wigner_6j
from scipy.special import sph_harm

from sympy import N

def sympy_to_complex(expr, prec=64):
    """Convert a SymPy expression to a complex number with optional precision."""
    result = expr.doit()
    real_part, imag_part = result.as_real_imag()
    return complex(N(real_part, prec), N(imag_part, prec))


class Spin_projections_padded:
    def __init__(self, angles, solution_number = 0):
        self.angles = angles

        #the solution number we are interested in, since we want the solution where the target is in the ground state
        # we fix it to the first column
        self.solution_number = solution_number

    @staticmethod
    def get_projection_range(spin):
        """Returns the range of magnetic quantum numbers for a given spin."""
        return np.arange(-int(round(2*spin))/2, int(round(2*spin))/2 + 1)
    
    @staticmethod
    def precompute_clebsch_gordan(l_f, s_f, j_f, l_i, s_i, j_i, J, I_f, I_i, m_s_f, m_s_i):
        """Precompute Clebsch-Gordan coefficients for all relevant combinations."""

        # Generate all projection combinations for quantum numbers
        m_f_range = Spin_projections_padded.get_projection_range(l_f)
        m_j_f_range = Spin_projections_padded.get_projection_range(j_f)
        m_j_i_range = Spin_projections_padded.get_projection_range(j_i)
        M_range = Spin_projections_padded.get_projection_range(J)
        m_I_i_range = Spin_projections_padded.get_projection_range(I_i)
        m_I_f_range = Spin_projections_padded.get_projection_range(I_f)
    

        # Precompute the Clebsch-Gordan coefficients as arrays, forcing them to numerical values
        cg_lsj_f = np.array([
            [sympy_to_complex(clebsch_gordan(l_f, s_f, j_f, m_f, m_s_f, m_j_f)) for m_j_f in m_j_f_range]
            for m_f in m_f_range
        ], dtype=complex)

        cg_lsj_i = np.array(
          [sympy_to_complex(clebsch_gordan(l_i, s_i, j_i, 0, m_s_i, m_j_i).conjugate()) for m_j_i in m_j_i_range]
        , dtype=complex)

        cg_jIJ_f = np.array([
            [
                [sympy_to_complex(clebsch_gordan(j_f, I_f, J, m_j_f, m_I_f, M)) for m_I_f in m_I_f_range]
            for M in M_range]
            for m_j_f in m_j_f_range
        ], dtype=complex)

        cg_jIJ_i = np.array([
            [  
            [sympy_to_complex(clebsch_gordan(j_i, I_i, J, m_j_i, m_I_i, M).conjugate()) for m_I_i in m_I_i_range] 
            for M in M_range]
            for m_j_i in m_j_i_range
        ], dtype=complex)
    
        return cg_lsj_f, cg_lsj_i, cg_jIJ_f, cg_jIJ_i, m_f_range, m_j_f_range, m_j_i_range, M_range
    
    @staticmethod
    def sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, m_s_f, m_s_i, I_f, I_i, angles):
        """Sum over all quantum projections to compute the total amplitude."""
    
        # Precompute Clebsch-Gordan coefficients and projection ranges
        cg_lsj_f, cg_lsj_i, cg_jIJ_f, cg_jIJ_i, m_f_range, m_j_f_range, m_j_i_range, M_range = Spin_projections_padded.precompute_clebsch_gordan(
        l_f, 0, j_f, l_i, 0, j_i, J, I_f, I_i, m_s_f, m_s_i
        )

    
        # Precompute spherical harmonics
        Y_i = np.conj(sph_harm(0, l_i, 0, 0))
        Y_f = np.array([((-1.0)**m_f)*sph_harm(m_f, l_f, 0, angles) for m_f in m_f_range])
        #print(Y_f)
        cg_final= np.einsum('ij,ik,klm->jlm', Y_f, cg_lsj_f, cg_jIJ_f)
        cg_initial = np.einsum('i,ijk->jk', cg_lsj_i, cg_jIJ_i)


        prefactor_per_angle = (
               4 * np.pi *  Y_i* np.einsum('ij,lik->ljk', cg_initial, cg_final)
            ) / (2j)
    

        # Sum over all prefactor elements per angle
        #shapes [m_I_i_proj][m_I_f_proj][angles]
        return np.transpose(prefactor_per_angle, (1,2,0))
    
    def precompute_spin_projections_per_shape_padded(self, homogeneous_matrix_arr):
        """
        Precompute the spin projections for all quantum numbers and angles.
        """
        
        dim_J = len(homogeneous_matrix_arr)                                     # number of arrays in the homogeneous matrix
        nch = homogeneous_matrix_arr[0].shape[0]                                # number of channels
        col_vector_temp = homogeneous_matrix_arr[-1][:,self.solution_number]    #choose the correct column corresponding to the target in ground state
        max_I_f = np.max(col_vector_temp[:,1])                                  #get the maximum I_f value
        max_m_I = len(Spin_projections_padded.get_projection_range(max_I_f))           #get the projection range for the maximum I_f value

        pre_spin_proj = np.zeros((dim_J, nch, max_m_I, max_m_I, len(self.angles)), dtype=np.complex128)  #preallocate the array for the spin projections

        for i, array in enumerate(homogeneous_matrix_arr):
            col_vector = array[:,self.solution_number]
            k = 0
            for l_f, I_f, l_i, I_i, J in col_vector:
                j_i = l_i
                j_f = l_f
                spin_proj = Spin_projections_padded.sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, 0, 0, I_f, I_i, self.angles)
                m_I_i_temp, m_I_f_temp, angles_temp = spin_proj.shape
                #print(m_I_i_temp, m_I_f_temp, angles_temp)
                pre_spin_proj[i, k, :m_I_i_temp, :m_I_f_temp, :angles_temp] = spin_proj
                k += 1

        return pre_spin_proj
    
    def precompute_spin_projections_padded(self, batched_dict, keys):
        
        spin_proj_batched = {}
        
        for key in keys:
            spin_proj_batched[key] = self.precompute_spin_projections_per_shape_padded(batched_dict[key])
        return spin_proj_batched, keys
    



# class Spin_projections:
#     def __init__(self, quantum_numbers_pos, quatum_numbers_neg, angles, solution_number = 0):
#         self.quantum_numbers_pos = quantum_numbers_pos
#         self.quantum_numbers_neg = quatum_numbers_neg
#         self.angles = angles

#         #the solution number we are interested in, since we want the solution where the target is in the ground state
#         # we fix it to the first column
#         self.solution_number = solution_number

#     @staticmethod
#     def get_projection_range(spin):
#         """Returns the range of magnetic quantum numbers for a given spin."""
#         return np.arange(-int(round(2*spin))/2, int(round(2*spin))/2 + 1)
    
#     @staticmethod
#     def precompute_clebsch_gordan(l_f, s_f, j_f, l_i, s_i, j_i, J, I_f, I_i, m_s_f, m_s_i):
#         """Precompute Clebsch-Gordan coefficients for all relevant combinations."""

#         # Generate all projection combinations for quantum numbers
#         m_f_range = Spin_projections.get_projection_range(l_f)
#         m_j_f_range = Spin_projections.get_projection_range(j_f)
#         m_j_i_range = Spin_projections.get_projection_range(j_i)
#         M_range = Spin_projections.get_projection_range(J)
#         m_I_i_range = Spin_projections.get_projection_range(I_i)
#         m_I_f_range = Spin_projections.get_projection_range(I_f)
    

#         # Precompute the Clebsch-Gordan coefficients as arrays, forcing them to numerical values
#         cg_lsj_f = np.array([
#             [sympy_to_complex(clebsch_gordan(l_f, s_f, j_f, m_f, m_s_f, m_j_f)) for m_j_f in m_j_f_range]
#             for m_f in m_f_range
#         ], dtype=complex)

#         cg_lsj_i = np.array(
#           [sympy_to_complex(clebsch_gordan(l_i, s_i, j_i, 0, m_s_i, m_j_i).conjugate()) for m_j_i in m_j_i_range]
#         , dtype=complex)

#         cg_jIJ_f = np.array([
#             [
#                 [sympy_to_complex(clebsch_gordan(j_f, I_f, J, m_j_f, m_I_f, M)) for m_I_f in m_I_f_range]
#             for M in M_range]
#             for m_j_f in m_j_f_range
#         ], dtype=complex)

#         cg_jIJ_i = np.array([
#             [  
#             [sympy_to_complex(clebsch_gordan(j_i, I_i, J, m_j_i, m_I_i, M).conjugate()) for m_I_i in m_I_i_range] 
#             for M in M_range]
#             for m_j_i in m_j_i_range
#         ], dtype=complex)
    
#         return cg_lsj_f, cg_lsj_i, cg_jIJ_f, cg_jIJ_i, m_f_range, m_j_f_range, m_j_i_range, M_range
    
#     @staticmethod
#     def sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, m_s_f, m_s_i, I_f, I_i, angles):
#         """Sum over all quantum projections to compute the total amplitude."""
    
#         # Precompute Clebsch-Gordan coefficients and projection ranges
#         cg_lsj_f, cg_lsj_i, cg_jIJ_f, cg_jIJ_i, m_f_range, m_j_f_range, m_j_i_range, M_range = Spin_projections.precompute_clebsch_gordan(
#         l_f, 0, j_f, l_i, 0, j_i, J, I_f, I_i, m_s_f, m_s_i
#         )

    
#         # Precompute spherical harmonics
#         Y_i = np.conj(sph_harm(0, l_i, 0, 0))
#         Y_f = np.array([((-1.0)**m_f)*sph_harm(m_f, l_f, 0, angles) for m_f in m_f_range])
#         #print(Y_f)
#         cg_final= np.einsum('ij,ik,klm->jlm', Y_f, cg_lsj_f, cg_jIJ_f)
#         cg_initial = np.einsum('i,ijk->jk', cg_lsj_i, cg_jIJ_i)


#         prefactor_per_angle = (
#                4 * np.pi *  Y_i* np.einsum('ij,lik->ljk', cg_initial, cg_final)
#             ) / (2j)
    

#         # Sum over all prefactor elements per angle
#         #shapes [m_I_i_proj][m_I_f_proj][angles]
#         return np.transpose(prefactor_per_angle, (1,2,0))
    
#     def precompute_spin_projections(self):
#         """
#         Precompute the spin projections for all quantum numbers and angles.
#         """
#         projections_pos = []
#         projections_neg = []

#         # Loop over positive parity states
#         for i, array in enumerate(self.quantum_numbers_pos):
#             temp_proj_per_J_tot = []
#             col_vector = array[:,self.solution_number]
#             for l_f, I_f, l_i, I_i, J in col_vector:
#                 #spin=0
#                 j_i = l_i
#                 j_f = l_f
#                 spin_proj = Spin_projections.sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, 0, 0, I_f, I_i, self.angles)
#                 temp_proj_per_J_tot.append(spin_proj)
#             projections_pos.append(temp_proj_per_J_tot)

#         # Loop over negative parity states
#         for i, array in enumerate(self.quantum_numbers_neg):
#             temp_proj_per_J_tot = []
#             col_vector = array[:,self.solution_number]
#             for l_f, I_f, l_i, I_i, J in col_vector:
#                 #spin=0
#                 j_i = l_i
#                 j_f = l_f
#                 spin_proj = Spin_projections.sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, 0, 0, I_f, I_i, self.angles)
#                 temp_proj_per_J_tot.append(spin_proj)
#             projections_neg.append(temp_proj_per_J_tot)

#         return projections_pos, projections_neg
    
# def sympy_to_complex(expr):
#     """Convert a SymPy expression to a Python complex number."""
#      # Use doit() to explicitly evaluate the Clebsch-Gordan coefficient
#     result = expr.doit()  # Forces the symbolic result to be evaluated
#     real_part, imag_part = result.as_real_imag()
#     return complex(float(real_part), float(imag_part))


# import numpy as np
# import scipy.constants as sc
# from sympy.physics.wigner import clebsch_gordan
# from sympy.physics.wigner import wigner_6j
# from scipy.special import sph_harm


# from sympy import N

# def sympy_to_complex(expr, prec=64):
#     """Convert a SymPy expression to a complex number with optional precision."""
#     result = expr.doit()
#     real_part, imag_part = result.as_real_imag()
#     return complex(N(real_part, prec), N(imag_part, prec))

# class Spin_projections_padded:
#     def __init__(self, quantum_numbers_pos, quatum_numbers_neg, angles, solution_number = 0):
#         self.quantum_numbers_pos = quantum_numbers_pos
#         self.quantum_numbers_neg = quatum_numbers_neg
#         self.angles = angles

#         #the solution number we are interested in, since we want the solution where the target is in the ground state
#         # we fix it to the first column
#         self.solution_number = solution_number

#     @staticmethod
#     def get_projection_range(spin):
#         """Returns the range of magnetic quantum numbers for a given spin."""
#         return np.arange(-int(round(2*spin))/2, int(round(2*spin))/2 + 1)
    
#     @staticmethod
#     def precompute_clebsch_gordan(l_f, s_f, j_f, l_i, s_i, j_i, J, I_f, I_i, m_s_f, m_s_i):
#         """Precompute Clebsch-Gordan coefficients for all relevant combinations."""

#         # Generate all projection combinations for quantum numbers
#         m_f_range = Spin_projections_padded.get_projection_range(l_f)
#         m_j_f_range = Spin_projections_padded.get_projection_range(j_f)
#         m_j_i_range = Spin_projections_padded.get_projection_range(j_i)
#         M_range = Spin_projections_padded.get_projection_range(J)
#         m_I_i_range = Spin_projections_padded.get_projection_range(I_i)
#         m_I_f_range = Spin_projections_padded.get_projection_range(I_f)
    

#         # Precompute the Clebsch-Gordan coefficients as arrays, forcing them to numerical values
#         cg_lsj_f = np.array([
#             [sympy_to_complex(clebsch_gordan(l_f, s_f, j_f, m_f, m_s_f, m_j_f)) for m_j_f in m_j_f_range]
#             for m_f in m_f_range
#         ], dtype=complex)

#         cg_lsj_i = np.array(
#           [sympy_to_complex(clebsch_gordan(l_i, s_i, j_i, 0, m_s_i, m_j_i).conjugate()) for m_j_i in m_j_i_range]
#         , dtype=complex)

#         cg_jIJ_f = np.array([
#             [
#                 [sympy_to_complex(clebsch_gordan(j_f, I_f, J, m_j_f, m_I_f, M)) for m_I_f in m_I_f_range]
#             for M in M_range]
#             for m_j_f in m_j_f_range
#         ], dtype=complex)

#         cg_jIJ_i = np.array([
#             [  
#             [sympy_to_complex(clebsch_gordan(j_i, I_i, J, m_j_i, m_I_i, M).conjugate()) for m_I_i in m_I_i_range] 
#             for M in M_range]
#             for m_j_i in m_j_i_range
#         ], dtype=complex)
    
#         return cg_lsj_f, cg_lsj_i, cg_jIJ_f, cg_jIJ_i, m_f_range, m_j_f_range, m_j_i_range, M_range
    
#     @staticmethod
#     def sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, m_s_f, m_s_i, I_f, I_i, angles):
#         """Sum over all quantum projections to compute the total amplitude."""
    
#         # Precompute Clebsch-Gordan coefficients and projection ranges
#         cg_lsj_f, cg_lsj_i, cg_jIJ_f, cg_jIJ_i, m_f_range, m_j_f_range, m_j_i_range, M_range = Spin_projections_padded.precompute_clebsch_gordan(
#         l_f, 0, j_f, l_i, 0, j_i, J, I_f, I_i, m_s_f, m_s_i
#         )

    
#         # Precompute spherical harmonics
#         Y_i = np.conj(sph_harm(0, l_i, 0, 0))
#         Y_f = np.array([((-1.0)**m_f)*sph_harm(m_f, l_f, 0, angles) for m_f in m_f_range])
#         #print(Y_f)
#         cg_final= np.einsum('ij,ik,klm->jlm', Y_f, cg_lsj_f, cg_jIJ_f)
#         cg_initial = np.einsum('i,ijk->jk', cg_lsj_i, cg_jIJ_i)


#         prefactor_per_angle = (
#                4 * np.pi *  Y_i* np.einsum('ij,lik->ljk', cg_initial, cg_final)
#             ) / (2j)
    

#         # Sum over all prefactor elements per angle
#         #shapes [m_I_i_proj][m_I_f_proj][angles]
#         return np.transpose(prefactor_per_angle, (1,2,0))
    
#     def precompute_spin_projections(self):
#         """
#         Precompute the spin projections for all quantum numbers and angles.
#         """
        
#         dim_J_pos = len(self.quantum_numbers_pos)
#         dim_J_neg = len(self.quantum_numbers_neg)
#         n_channels = max(arr.shape[0] for arr in self.quantum_numbers_pos)
#         col_vector_temp = self.quantum_numbers_pos[-1][:,self.solution_number]
#         max_I = np.max(col_vector_temp[:,1])
#         max_m_I = len(Spin_projections_padded.get_projection_range(max_I))

#         pre_spin_proj_pos = np.zeros((dim_J_pos, n_channels, max_m_I, max_m_I, len(self.angles)), dtype=np.complex128)
#         pre_spin_proj_neg = np.zeros((dim_J_neg, n_channels, max_m_I, max_m_I, len(self.angles)), dtype=np.complex128)
        
#         #self.quantum_numbers_pos[-1].shape)

#         # Loop over positive parity states
#         for i, array in enumerate(self.quantum_numbers_pos):
#             col_vector = array[:,self.solution_number]
#             k = 0
#             for l_f, I_f, l_i, I_i, J in col_vector:
#                 #spin=0
#                 j_i = l_i
#                 j_f = l_f
#                 spin_proj = Spin_projections_padded.sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, 0, 0, I_f, I_i, self.angles)
#                 m_I_i_temp, m_I_f_temp, angles_temp = spin_proj.shape
#                 #print(m_I_i_temp, m_I_f_temp, angles_temp)
#                 pre_spin_proj_pos[i, k, :m_I_i_temp, :m_I_f_temp, :angles_temp] = spin_proj
#                 k += 1
#         # Loop over negative parity states
#         for i, array in enumerate(self.quantum_numbers_neg):
#             col_vector = array[:,self.solution_number]
#             k = 0
#             for l_f, I_f, l_i, I_i, J in col_vector:
#                 #spin=0
#                 j_i = l_i
#                 j_f = l_f
#                 spin_proj = Spin_projections_padded.sum_over_projections_vectorized(l_f, j_f, J, l_i, j_i, 0, 0, I_f, I_i, self.angles)
#                 m_I_i_temp, m_I_f_temp, angles_temp = spin_proj.shape
#                 pre_spin_proj_neg[i, k, :m_I_i_temp, :m_I_f_temp, :angles_temp] = spin_proj
#                 k += 1


#         return pre_spin_proj_pos, pre_spin_proj_neg