# Maths
import numpy as np
from scipy.linalg import cho_solve, cho_factor

####################
#### BASE TYPES ####
####################

class NoiseProcess:
    """A base noise process class defined through a covariance matrix. This class is not useful to instantiate on its own.
    
    Attributes:
        data (Dataset): The dataset for this noise process.
        label (str, optional): The name of this noise process. Defaults to None.
    """
    
    def __init__(self, data=None, label=None):
        self.data = data
        self.label = label

    def compute_cov_matrix(self, pars):
        raise NotImplementedError(f"Must implemenent the method compute_cov_matrix for class {self.__class__.__name__}.")
    
    def __repr__(self):
        return f"Noise process: {self.__class__.__name__}"
    
    def initialize(self, p0):
        self.p0 = p0
        
    def compute_noise_components(self, *args, **kwargs):
        return {}

class UnCorrelatedNoiseProcess(NoiseProcess):
    """ Trait.
    """
    pass

class CorrelatedNoiseProcess(NoiseProcess):
    
    def __init__(self, data=None, kernel=None, label=None):
        super().__init__(data=data, label=label)
        self.kernel = kernel
        
    def initialize(self, p0, x1=None, xpred=None):
        """Default initializer for correlated noise kernels. By default, only the distance matrix is constructed.

        Args:
            x1 (np.ndarray, optional): The x1 vector. Defaults to the Data grid (self.x).
            xpred (np.ndarray, optional): The vector to make predictions on. Defaults to x1.
        """
        super().initialize(p0)
        if x1 is None:
            x1 = self.data.x
        if xpred is None:
            xpred = x1
        self.kernel.initialize(p0, x1, xpred)
        
    def compute_residuals(self, pars, linpred):
        """Computes the final residuals (linpred - noise process)

        Args:
            pars (Parameters): The parameters to use.
            linpred (np.ndarray): The linear predictor.

        Returns:
            np.ndarray: The final residuals.
        """
        return linpred - self.realize(pars, linpred=linpred)
        
class StationaryNoiseProcess(CorrelatedNoiseProcess):
    """Trait.
    """
    pass


#####################
#### WHITE NOISE ####
#####################

class WhiteNoiseProcess(UnCorrelatedNoiseProcess):
    
    def compute_cov_matrix(self, pars):
        data_errors = self.compute_data_errors(pars)
        K = np.diag(data_errors**2)
        return K
    
    def compute_data_errors(self, pars):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            np.ndarray: The final data errors.
        """
    
        # Get intrinsic data errors
        errors = self.data.get_errors()
        
        # Add jitter in quadrature
        errors = np.sqrt(errors**2 + pars[f"jitter_{self.data.label}"].value**2)
        
        return errors


##########################
#### GAUSSIAN PROCESS ####
##########################

class GaussianProcess(CorrelatedNoiseProcess):
    """A noise kernel defined through a single GP and diagonal error terms with an additional "jitter" parameter. Each jitter parameter must be named "jitter_label" where label is the data label.
    """

    def compute_cov_matrix(self, pars, include_uncorr_error=True):
        
        # Compute GP kernel
        K = self.kernel.compute_cov_matrix(pars)
        
        # Uncorrelated errors (intrinsic error bars and additional per-data label jitter)
        if include_uncorr_error:
            assert K.shape[0] ==  K.shape[1]
            data_errors = self.compute_data_errors(pars)
            assert K.shape[0] == data_errors.size
            np.fill_diagonal(K, np.diagonal(K) + data_errors**2)
        
        return K
    
    def compute_data_errors(self, pars, include_corr_error=False, linpred=None):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.
            include_gp_error (bool, optional): Whether or not to include the gp error. Defaults to False.
            linear_pred (np.ndarray): The linear predictor; the data vector containing noise.
            
        Returns:
            np.ndarray: The data errors.
        """
    
        # Get intrinsic data errors
        errors = self.data.get_apriori_errors()
        
        # Add any jitter
        errors += pars[f"jitter_{next(iter(self.data.items()))[1].label}"].value
            
        # Add in quadrature the gp error
        if include_corr_error:
            gp_error = self.compute_corr_error(pars, linpred=linpred)
            errors = np.sqrt(errors**2 + gp_error**2)

        return errors
    
    def realize(self, pars, linpred, xpred=None):
        
        # Get grids
        xdata = self.data.x
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(pars, x1=xdata, xpred=xdata)
        K = self.compute_cov_matrix(pars)
        
        # Compute version of K without intrinsic data error
        self.initialize(pars, x1=xpred, xpred=xdata)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()
        
        return mu
    
    def compute_noise_components(self, pars, linpred, xpred=None):
        if xpred is None:
            xpred = self.data.x
        comps = {}
        comps[self.label] = self.compute_gp_with_error(pars, linpred, xpred)
        return comps
    
    def compute_corr_error(self, pars, linpred, xpred=None):
            
        # Get grids
        xdata = self.data.x
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(pars, x1=xdata, xpred=xdata)
        K = self.compute_cov_matrix(pars)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)

        self.initialize(pars, x1=xdata, xpred=xdata)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)
            
        self.initialize(pars, x1=xpred, xpred=xpred)
        Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
        
        B = cho_solve(L, Ks.T)
        
        error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())
        
        return error
    
    def compute_gp_with_error(self, pars, linpred, xpred=None):
        
        # Get grids
        xdata = self.data.x
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(pars, x1=xdata, xpred=xdata)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without intrinsic data error
        self.initialize(pars, x1=xpred, xpred=xdata)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()
        
        self.initialize(pars, x1=xpred, xpred=xpred)
        Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
        
        B = cho_solve(L, Ks.T)
        
        error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())
        
        return mu, error

########################
#### IMPORT KERNELS ####
########################

from .kernels import QuasiPeriodic