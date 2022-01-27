# Base Python
import functools

# Maths
import numpy as np

# Plots
import matplotlib.pyplot as plt

# LLVM
import numba

####################
#### BASE TYPES ####
####################

class NoiseKernel:
    """A base noise kernel class defined through a covariance matrix. This class is not useful to instantiate on its own.
    
    Attributes:
        par_names (list of strings, optional): The parameter names for this kernel, optional.
    """

    par_names = []
    
    def __init__(self, par_names=None):
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

    #########################
    #### DISTANCE MATRIX ####
    #########################

    def compute_dist_matrix(self, x1, x2):
        """Computes the stationary distance matrix, D_ij = |x_i - x_j|.

        Args:
            x1 (np.ndarray): The first vector.
            x2 (np.ndarray): The second vector.

        Returns:
            np.ndarray: The stationary distance matrix D_ij
        """
        return self.compute_stationary_dist_matrix(x1, x2)

    
    #@functools.lru_cache
    @staticmethod
    def compute_stationary_dist_matrix(x1, x2):
        return StationaryNoiseKernel._compute_stationary_dist_matrix(x1, x2)

    @staticmethod
    @numba.njit(nogil=True)
    def _compute_stationary_dist_matrix(x1, x2):
        """Computes the distance matrix, D_ij = |x_i - x_j|.

        Args:
            x1 (np.ndarray): The first vec to use (x_i).
            x2 (np.ndarray): The second vec to use (x_j).

        Returns:
            np.ndarray: The distance matrix D_ij.
        """
        n1 = len(x1)
        n2 = len(x2)
        out = np.zeros(shape=(n1, n2), dtype=numba.float64)
        for i in range(n1):
            for j in range(n2):
                out[i, j] = np.abs(x1[i] - x2[j])
        return out


###################
#### QP KERNEL ####
###################

class QuasiPeriodic(StationaryNoiseKernel):
    """A Quasiperiodic kernel. The hyperparameters may be called anything, but must be in the order of amplitude, exp length scale, period length scale, and period.
    """

    par_names = ["amplitude", "exponential length scale", "period length scale", "period"]
    
    def compute_cov_matrix(self, pars, x1, x2):
        """Computes the QP kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The covariance matrix K.
        """

        # Distance matrix
        dist_matrix = self.compute_dist_matrix(x1, x2)
        
        # Alias params
        amp = pars[self.par_names[0]].value
        exp_length = pars[self.par_names[1]].value
        per_length = pars[self.par_names[2]].value
        per = pars[self.par_names[3]].value

        # Compute exp decay term
        decay_term = -0.5 * dist_matrix**2 / exp_length**2
        
        # Compute periodic term
        periodic_term = -0.5 * np.sin((np.pi / per) * dist_matrix)**2 / per_length**2
        
        # Add and include amplitude
        K = amp**2 * np.exp(decay_term + periodic_term)
        
        return K
