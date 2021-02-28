import numpy as np
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse
from numba import jit, njit
import matplotlib.pyplot as plt

class NoiseKernel:
    """A base noise kernel class. This class is not useful to instantiate on its own.
    
    Attributes:
        data_list (list): A list containing the data objects which utilize this noise kernel.
        par_names (list): A list of parameters for this kernel, must be in order of their .
        
    Examples:
        To form one noise kernel for all data sets, construct as 
    """
    
    def __init__(self, data, par_names=None):
        """Base constructor for a generic noise kernel. This should be called by any sub-class.

        Args:
            data (CompositeData): The composite data objects which utilize this noise kernel.
            par_names (list): A list of parameter names. They must be provided in the order specified by the appropriate kernel.
        """
        self.data = data
        self.par_names = [] if par_names is None else par_names
        self.x = self.data.get_vec('x')
        self.data_inds = {}
        for instname in self.data:
            self.data_inds[instname] = self.data.get_inds(instname)
        
    def compute_cov_matrix(self, x1, x2, **kwargs):
        raise NotImplementedError("Must implement a compute_cov_matrix method.")
    
    def get_intrinsic_data_errors(self):
        """Generates the intrinsic data errors (measured apriori).

        Returns:
            np.ndarray: The intrinsic data error bars.
        """
        errors = np.array([], dtype=float)
        x = np.array([], dtype=float)
        for data in self.data.values():
            x = np.concatenate((x, data.x))
            errors = np.concatenate((errors, data.yerr))
        ss = np.argsort(x)
        errors = errors[ss]
        return errors
        
    @property
    def has_correlated_noise(self):
        return isinstance(self, CorrelatedNoiseKernel)
        
class WhiteNoise(NoiseKernel):
    """A noise kernel for white noise, where all diagonal terms in the covariance matrix are zero. The noise kernel is computed by adding a jitter term and the intrinsic error bars in quadrature.
    """
    
    def compute_cov_matrix(self, pars, **kwargs):
        """Computes the covariance matrix for white noise by filling the diagonal with provided errors.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The covariance matrix.
        """
        
        # Intrinsic data error bars
        errors = self.compute_data_errors(pars)
        
        # Number of data points
        n = len(errors)
        
        # Init cov matrix
        cov_matrix = np.zeros(shape=(n, n), dtype=float)
        
        # Fill diagonal with squared errors
        np.fill_diagonal(cov_matrix, errors**2)
        
        # Return covariance matrix
        return cov_matrix
    
    def compute_data_errors(self, pars, include_white_error=True):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The final data errors.
        """
    
        # Get intrinsic data errors
        errors = self.get_intrinsic_data_errors()
        
        # Square
        errors = errors**2
        
        # Add per-instrument jitter terms in quadrature
        if include_white_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                pname = 'jitter_' + data.label
                errors[inds] += pars[pname].value**2
                    
        # Square root
        errors = errors**0.5

        return errors
    
    
class CorrelatedNoiseKernel(NoiseKernel):
    
    def compute_cov_matrix(self, pars, include_white_error=True, **kwargs):
        raise NotImplementedError("Must implement the method compute_cov_matrix")
    
    def compute_data_errors(self, pars, include_white_error=True, include_kernel_error=True, kernel_error=None, residuals_with_noise=None):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The final data errors.
        """
    
        # Get intrinsic data errors
        errors = self.get_intrinsic_data_errors()
        
        # Square
        errors = errors**2
        
        # Add per-instrument jitter terms in quadrature
        if include_white_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                pname = 'jitter_' + data.label
                errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_kernel_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                if kernel_error is None:
                    _, _kernel_error = self.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, return_kernel_error=True)
                    errors[inds] += _kernel_error**2
                else:
                    errors[inds] += kernel_error[inds]**2
                    
        # Square root
        errors = errors**0.5

        return errors
    
    def compute_dist_matrix(self, x1=None, x2=None):
        """Default wrapper to compute the cov matrix.

        Args:
            x1 (np.ndarray, optional): The x1 vector. Defaults to the Data grid.
            x2 (np.ndarray, optional): The x2 vector. Defaults to the Data grid.
        """
        if x1 is None:
            x1 = self.x
        if x2 is None:
            x2 = self.x
        self.dist_matrix = self._compute_dist_matrix(x1, x2)
    
    
    @staticmethod
    #@njit
    def _compute_dist_matrix(x1, x2):
        """Computes the distance matrix, D_ij = |x_i - x_j|

        Args:
            x1 (np.ndarray): The first vec to use.
            x2 (np.ndarray): The second vec to use.

        Returns:
            np.ndarray: The distance matrix.
        """
        n1 = len(x1)
        n2 = len(x2)
        out = np.zeros(shape=(n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                out[i, j] = np.abs(x1[i] - x2[j])
        return out
    

class GaussianProcess(CorrelatedNoiseKernel):
    """A generic Gaussian process kernel.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.compute_dist_matrix()
        except:
            pass
        
    def realize(self, pars, residuals_with_noise, xpred=None, xres=None, return_kernel_error=False, **kwargs):
        """Realize the GP (sample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            residuals (np.ndarray): The residuals before the GP is subtracted.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            errors (np.ndarray): The errorbars, added in quadrature.
            return_kernel_error (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization (i.e, minimiation surface is purely concave or convex).
        """
        
        # Resolve the grids to use.
        if xres is None:
            xres = self.x
        if xpred is None:
            xpred = xres
        
        # Get K
        self.compute_dist_matrix(xres, xres)
        K = self.compute_cov_matrix(pars, include_white_error=True, include_kernel_error=False)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres)
        Ks = self.compute_cov_matrix(pars, include_white_error=False, include_kernel_error=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_kernel_error:
            self.compute_dist_matrix(xpred, xpred)
            Kss = self.compute_cov_matrix(pars, include_white_error=False, include_kernel_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.compute_dist_matrix()
            return mu, unc
        else:
            self.compute_dist_matrix()
            return mu


class QuasiPeriodic(GaussianProcess):
    """A Quasiperiodic GP.
    """
    
    def compute_cov_matrix(self, pars, include_white_error=True, include_kernel_error=False, kernel_error=None):
        
        # Alias params
        amp = pars[self.par_names[0]].value
        exp_length = pars[self.par_names[1]].value
        per = pars[self.par_names[2]].value
        per_length = pars[self.par_names[3]].value

        # Compute exp decay term
        decay_term = -0.5 * self.dist_matrix**2 / exp_length**2
        
        # Compute periodic term
        periodic_term = -0.5 * np.sin((np.pi / per) * self.dist_matrix)**2 / per_length**2
        
        # Add and include amplitude
        cov_matrix = amp**2 * np.exp(decay_term + periodic_term)
        
        # Include errors on the diagonal
        if include_white_error:
            data_errors = self.compute_data_errors(pars, include_white_error=include_white_error, include_kernel_error=include_kernel_error, kernel_error=kernel_error)
            np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + data_errors**2)
        
        return cov_matrix