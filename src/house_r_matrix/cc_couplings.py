import numpy as np
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.wigner import wigner_6j
from cc_constants import CC_Constants
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from cc_constants import CC_Constants


def hat(a):
    """
    Normalization coefficient: sqrt(2a + 1)
    Returns np.complex128
    """
    return np.complex128(np.sqrt(2 * a + 1))


def coupling_matrix_elements(I_i, l_i, j_i, I_f, l_f, j_f, s, J, Q):
    '''
    Given the quantum numbers of one specific choice of initial and final channels calculate the coupling
    coefficient. This is the general form, for the moment l_i=j_i, s=0, l_f=j_f and both I_f, I_i must be even.
    '''
    K_f = 0  ## for I=0,2,4 only
    K_i = 0  ## for I=0,2,4 only
    coupling_initial_final = (1j**(l_i-l_f))*(-1)**(Q+J+l_i+I_f)*hat(I_i)*hat(l_i)*hat(Q) \
               * wigner_6j(l_i,I_i,J,I_f,l_f,Q).n(17) *clebsch_gordan(l_i,Q,l_f,0,0,0).n(17)\
               *clebsch_gordan(I_i,Q,I_f,K_i,0,K_f).n(17)
    
    return coupling_initial_final


class CC_Couplings:
    def __init__(self, mass_t, mass_p, E_states, I_states, pi_xp, pi_xt, J_tot_max):
        assert len(E_states) == len(I_states), "E_states and I_states must match in length"

        self.mass_t = mass_t
        self.mass_p = mass_p
        self.E_states = E_states
        self.I_states = I_states
        self.pi_xp = pi_xp
        self.pi_xt = pi_xt
        self.J_tot_max = J_tot_max
        self.batched_dict, self.keys = self.generate_quantum_num_batched()


    
    #energy independent
    @staticmethod
    def allowed_parity_waves_grouping(J_tot, I, pi_xp, pi_xt):
        l_values = np.arange(abs(J_tot - I), J_tot + I + 1)
        parities = ((-1) ** l_values) * pi_xt * pi_xp

        pos_par_l = [(int(l), I) for l in l_values[parities == 1]]
        neg_par_l = [(int(l), I) for l in l_values[parities == -1]]

        return pos_par_l, neg_par_l

    #energy independent
    def matrix_quantum_numbers(self, J_tot):
        stacked_pos, stacked_neg = [], []

        for I in self.I_states:
            pos_l, neg_l = self.allowed_parity_waves_grouping(J_tot, I, self.pi_xp, self.pi_xt)
            stacked_pos.extend(pos_l)
            stacked_neg.extend(neg_l)

        pos = np.array(stacked_pos, dtype=int) if stacked_pos else np.empty((0, 2), dtype=int)
        neg = np.array(stacked_neg, dtype=int) if stacked_neg else np.empty((0, 2), dtype=int)

        I_initial = self.I_states[0]

        if not np.any(pos[:, 1] == I_initial):
            pos = np.empty((0, 2), dtype=int)
        if not np.any(neg[:, 1] == I_initial):
            neg = np.empty((0, 2), dtype=int)

        def build_matrix(channel_array):
            if channel_array.shape[0] == 0:
                return np.empty((0, 0, 4), dtype=int)
            return np.concatenate([
                channel_array[:, None, :].repeat(len(channel_array), axis=1),
                channel_array[None, :, :].repeat(len(channel_array), axis=0)
            ], axis=2)

        matrix_pos = build_matrix(pos)
        matrix_neg = build_matrix(neg)

        return matrix_pos, matrix_neg


    # energy independent
    def matrix_J_flat(self):

        pos_matrix_array, neg_matrix_array, matrix_tot = [], [], []

        for J_tot in range(int(self.J_tot_max) + 1):
            pos_matrix, neg_matrix = self.matrix_quantum_numbers(J_tot)
            if pos_matrix.size:
                J_tot_col_pos = np.full((*pos_matrix.shape[:2], 1), J_tot)
                pos_matrix_array.append(np.concatenate((pos_matrix, J_tot_col_pos), axis=2))
            if neg_matrix.size:
                J_tot_col_neg = np.full((*neg_matrix.shape[:2], 1), J_tot)
                neg_matrix_array.append(np.concatenate((neg_matrix, J_tot_col_neg), axis=2))
        for i in range(max(len(pos_matrix_array), len(neg_matrix_array))):
            if i < len(pos_matrix_array):
                matrix_tot.append(pos_matrix_array[i])
            if i < len(neg_matrix_array):
                matrix_tot.append(neg_matrix_array[i])

        return matrix_tot
    
    # energy independent
    def couplings_per_shape(self, Q, homogeneous_matrix_arr):
        batch_size, nch, nch, _ = homogeneous_matrix_arr.shape
        V = np.zeros((batch_size, nch, nch), dtype=np.complex128)
        for k, matrix in enumerate(homogeneous_matrix_arr):
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    l_f, I_f, l_i, I_i, J_tot = matrix[i][j]
                    V[k][i][j] = coupling_matrix_elements(I_i, l_i, 0, I_f, l_f, 0, 0, J_tot, Q)
        return V

    
    #energy dependent, pass the E_lab and homogeneous_matrix_arr
    def energy_centrifugal_mom_per_shape(self, E_lab, homogeneous_matrix_arr):
        
        E_arr, l_arr, k_arr = [], [], []
        contants_class = CC_Constants(self.mass_t, self.mass_p, E_lab, self.E_states)

        I_states = self.I_states
        E_com = contants_class.E_lab_to_COM()
        hbar_2mu = contants_class.h2_mass

        for matrix in homogeneous_matrix_arr:
            E_arr_temp, l_arr_temp, k_arr_temp = [], [], []
            nch = matrix.shape[0]
            for i in range(nch):
                l_f, I_f, l_i, I_i, J_tot = matrix[i][0]
                idx = np.where(I_states == I_f)[0][0]
                E_ch = float(E_com[idx])
                k_ch = np.sqrt(E_ch / hbar_2mu)

                E_arr_temp.append(E_ch)
                l_arr_temp.append(l_f)
                k_arr_temp.append(k_ch)
            E_arr.append(E_arr_temp)
            l_arr.append(l_arr_temp)
            k_arr.append(k_arr_temp)
        
        return np.array(E_arr), np.array(l_arr), np.array(k_arr)
    
    
    #energy independent
    def generate_quantum_num_batched(self):
        '''
        Generate the array of quantum numbers that are coupled in the CC equations.
        '''
        #generate the array of all the couplings, this is an inhomogeneous multi-dimensional array
        flat_couplings = self.matrix_J_flat()
        batched_dict = CC_Couplings.group_by_shape_numpy_ready(flat_couplings)
        keys = list(batched_dict.keys())

        return batched_dict, keys
    

    def generate_couplings_batched(self, Q):
        '''
        Generate the couplings for the CC equations in a batched manner.
        '''
        #generate the array of all the couplings, this is an inhomogeneous multi-dimensional array
        couplings_batched_dic = {}
        for key in self.keys:
            couplings_batched_dic[key] = self.couplings_per_shape(Q, self.batched_dict[key])

        return couplings_batched_dic, self.keys
    
    def generate_energy_centrifugal_mom_batched(self, E_lab):
        '''
        Generate the couplings for the CC equations in a batched manner.
        '''
        #generate the array of all the couplings, this is an inhomogeneous multi-dimensional array
        E_batched, l_batched, k_batched = {}, {}, {}
        for key in self.keys:
            E_batched[key], l_batched[key], k_batched[key] = self.energy_centrifugal_mom_per_shape(E_lab, self.batched_dict[key])

        return E_batched, l_batched, k_batched, self.keys
    
    
    @staticmethod
    def group_by_shape_numpy_ready(array_list):
        '''
        group an inhomogenerous list of numpy arrays by their dimensions. To be used in batching.
        '''
        shape_groups = defaultdict(list)
        order = []
        for arr in array_list:
            m = arr.shape[0]
            shape_groups[m].append(arr)
            if m not in order:
                order.append(m)

        grouped_arrays = {}
        for m in order:
            grouped_arrays[m] = np.stack(shape_groups[m], axis=0)  # shape (K_group, m, m, 5)

        return grouped_arrays
    




