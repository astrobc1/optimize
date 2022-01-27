# Maths
import numpy as np
from scipy.linalg import cho_solve, cho_factor

####################
#### BASE TYPES ####
####################

class NoiseProcess:
    """A base noise process class defined through a covariance matrix. This class is not useful to instantiate on its own.
    
    Attributes:
        name (str, optional): The name of this noise process. Defaults to None.
    """

    def compute_cov_matrix(self, *args, **kwargs):
        raise NotImplementedError(f"Must implemenent the method compute_cov_matrix for class {self.__class__.__name__}.")

    def compute_data_errors(self, *args, **kwargs):
        raise NotImplementedError(f"Must implemenent the method compute_data_errors for class {self.__class__.__name__}.")
    
    def __repr__(self):
        return "Base Noise Process"


class UnCorrelatedNoiseProcess(NoiseProcess):
    """ Trait.
    """
    pass


class CorrelatedNoiseProcess(NoiseProcess):
    """ Trait.
    """
    pass
        

#####################
#### WHITE NOISE ####
#####################

class WhiteNoiseProcess(UnCorrelatedNoiseProcess):
    
    def compute_cov_matrix(self, pars, data_errors):
        K = np.diag(data_errors**2)
        return K
    

##########################
#### GAUSSIAN PROCESS ####
##########################

class GaussianProcess(CorrelatedNoiseProcess):
    """A noise kernel defined through a single GP and diagonal error terms with an additional "jitter" parameter. Each jitter parameter must be named "jitter_label" where label is the data label.
    """

    def __init__(self, kernel=None):
        self.kernel = kernel

    def compute_cov_matrix(self, pars, x1, x2, data_errors=None, include_uncorrelated_error=True):
        
        # Compute GP kernel
        K = self.kernel.compute_cov_matrix(pars, x1, x2)

        # Uncorrelated errors (intrinsic error bars and additional per-data label jitter)
        if include_uncorrelated_error:
            assert K.shape[0] ==  K.shape[1]
            assert K.shape[0] == data_errors.size
            np.fill_diagonal(K, np.diagonal(K) + data_errors**2)
        
        return K
    
    def predict(self, pars, linpred, xdata, xpred=None, data_errors=None):
        
        # Get grids
        if xpred is None:
            xpred = xdata
        
        # Get K
        K = self.compute_cov_matrix(pars, xdata, xdata, data_errors, include_uncorrelated_error=True)
        
        # Compute version of K without intrinsic data error
        Ks = self.compute_cov_matrix(pars, xpred, xdata, data_errors=None, include_uncorrelated_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)

        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()

        Kss = self.compute_cov_matrix(pars, xpred, xpred, include_uncorrelated_error=False)
        B = cho_solve(L, Ks.T)
        error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())

        return mu, error
    

########################
#### IMPORT KERNELS ####
########################

from .kernels import QuasiPeriodic