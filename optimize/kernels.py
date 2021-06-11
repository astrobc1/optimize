# Maths
import numpy as np
from scipy.linalg import cho_solve, cho_factor

# optimize deps
import optimize.maths as optmath

# LLVM
import numba


####################
#### BASE TYPES ####
####################

class NoiseKernel:
    """A base noise kernel class defined through a covariance matrix. This class is not useful to instantiate on its own.
    
    Attributes:
        data (Dataset): The dataset using this noise kernel.
        par_names (list of strings, optional): The parameter names for this kernel, optional.
    """
    
    def __init__(self, data=None, par_names=None):
        self.data = data
        self.par_names = [] if par_names is None else par_names

    def compute_cov_matrix(self, pars, *args, **kwargs):
        raise NotImplementedError(f"Must implemenent the method compute_cov_matrix for class {self.__class__.__name__}.")
    
    def initialize(self, p0):
        """Default wrapper to initialize the kernel, storing the parameters.
        """
        self.p0 = p0
    
class UnCorrelatedNoiseKernel(NoiseKernel):
    """Behaves as a trait for now.
    """
    pass

class CorrelatedNoiseKernel(NoiseKernel):
    """Behaves as a trait for now.
    """
    pass

class StationaryNoiseKernel(CorrelatedNoiseKernel):
    """Noise kernel which only dependes on the relative difference between 2 values. Also has trait like behavior.
    """
    
    def compute_dist_matrix(self, x1, x2):
        """Computes the stationary distance matrix, D_ij = |x_i - x_j|.

        Args:
            x1 (np.ndarray): The first vector.
            x2 (np.ndarray): The second vector.

        Returns:
            np.ndarray: The stationary distance matrix D_ij
        """
        return optmath.compute_stationary_dist_matrix(x1, x2)
    
    def initialize(self, p0, x1=None, xpred=None):
        """Initializes the noise kernel by computing the stationary distance matrix.

        Args:
            p0 (Parameters): The parameters to use.
            x1 (np.ndarray, optional): The first vector. Defaults to data.x.
            xpred (np.ndarray, optional): The vector to make predictions on. Defaults to x1.
        """
        super().initialize(p0)
        if x1 is None:
            x1 = self.data.x
        if xpred is None:
            xpred = x1
        self.dist_matrix = self.compute_dist_matrix(x1, xpred)


###################
#### QP KERNEL ####
###################

class QuasiPeriodic(StationaryNoiseKernel):
    """A Quasiperiodic kernel. The hyperparameters may be called anything, but must be in the order of amplitude, exp length scale, period length scale, and period.
    """
    
    def compute_cov_matrix(self, pars):
        """Computes the QP kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The covariance matrix K.
        """
        
        # Alias params
        amp = pars[self.par_names[0]].value
        exp_length = pars[self.par_names[1]].value
        per_length = pars[self.par_names[2]].value
        per = pars[self.par_names[3]].value

        # Compute exp decay term
        decay_term = -0.5 * self.dist_matrix**2 / exp_length**2
        
        # Compute periodic term
        periodic_term = -0.5 * np.sin((np.pi / per) * self.dist_matrix)**2 / per_length**2
        
        # Add and include amplitude
        K = amp**2 * np.exp(decay_term + periodic_term)
        
        return K
