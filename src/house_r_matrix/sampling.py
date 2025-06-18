from scipy.stats import qmc
import numpy as np

class Sampling:
    def __init__(self, scale_training, samples, *central_val):
        '''
        the inputs of this class are the scale of the training width, as a fraction of the mean values, the number of samples
        and the mean values themselves.
        '''
        self.scale_training = scale_training
        self.samples = samples
        self.central_val = central_val
        
    def latinhypercube(self):
        '''
        this class method samples the parameter space using a latinhypercube routine for a given set of inputs.
        '''
        alpha_central = np.array(self.central_val)
        zero_indices = np.where(alpha_central == 0)[0]
        non_zero_indices = np.where(alpha_central != 0)[0]

        # Handle non-zero central values
        alpha_non_zero_central = alpha_central[non_zero_indices]
        alpha_width = np.abs(alpha_non_zero_central * self.scale_training)
    
        alpha_lower = alpha_non_zero_central - alpha_width
        alpha_higher = alpha_non_zero_central + alpha_width
    
        alpha_bounds = np.array([alpha_lower, alpha_higher]).T

        if len(non_zero_indices) > 0:
            sampler = qmc.LatinHypercube(d=len(non_zero_indices))
            sample_parameter = sampler.random(self.samples)
            non_zero_samples = qmc.scale(sample_parameter, alpha_bounds[:, 0], alpha_bounds[:, 1])
        else:
            non_zero_samples = np.empty((self.samples, 0))
    
        # Initialize the full sample array with zeros
        training_array = np.zeros((self.samples, len(alpha_central)))

        # Place the non-zero samples in their respective positions
        training_array[:, non_zero_indices] = non_zero_samples
    
        return training_array