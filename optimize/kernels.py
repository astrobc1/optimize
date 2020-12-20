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
        """Constructs a generic GP. Should be called by any class that extends this class.

        Args:
            data (MixedData): The mixed data objects which utilize this noise kernel. The combined and sorted x variables will form the axes for the covariance matrix.
            par_names (list): A list of parameter names. They must be provided in the order specified by the appropriate kernel.
        """
        self.data = data
        self.par_names = [] if par_names is None else par_names
        self.x = self.data.get_vec('x')
        self.data_inds = {}
        for instname in self.data:
            self.data_inds[instname] = self.data.get_inds(instname)
        
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
        
    def compute_cov_matrix(self, x1, x2):
        raise NotImplementedError("Must implement a compute_cov_matrix method.")
    
    def compute_data_errors(self, pars):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The errors
        """
        errors = self.get_data_errors()
        errors **= 2
        for label in self.data:
            errors[self.data_inds[label]] += pars['jitter_' + label].value**2
        errors **= 0.5
        return errors
    
    def get_data_errors(self):
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
        
    @staticmethod
    def _compute_dist_matrix(x1, x2):
        """Computes the (possibly not square) distance matrix, D. D_ij = |x_i - x_j|

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
    
    
class WhiteNoise(NoiseKernel):
    """A noise kernel for white noise, where all diagonal terms in the covariance matrix are zero. The noise kernel is computed by adding a jitter term and the intrinsic error bars in quadrature.
    """
    
    def compute_cov_matrix(self, pars, apply_errors=True):
        """Computes the covariance matrix for white noise by filling the diagonal with provided errors.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The covariance matrix.
        """
        errors = self.compute_data_errors(pars)
        n = len(errors)
        cov_matrix = np.zeros(shape=(n, n), dtype=float)
        np.fill_diagonal(cov_matrix, errors**2)
        return cov_matrix
    
class GaussianProcess(NoiseKernel):
    """A generic Gaussian process kernel.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_dist_matrix()

    def realize(self, pars, residuals, xpred=None, xres=None, return_unc=False, **kwargs):
        """Realize the GP (sample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            residuals (np.ndarray): The residuals before the GP is subtracted.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            errors (np.ndarray): The errorbars, added in quadrature.
            return_unc (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization (i.e, minimiation surface is purely concave or convex).
        """
        
        # Resolve the grids to use.
        if xres is None:
            xres = self.data.get_vec('x')
        if xpred is None:
            xpred = xres
        
        # Get K
        self.compute_dist_matrix(xres, xres)
        K = self.compute_cov_matrix(pars, apply_errors=True)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres)
        Ks = self.compute_cov_matrix(pars, apply_errors=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_unc:
            self.compute_dist_matrix(xpred, xpred)
            Kss = self.compute_cov_matrix(pars, apply_errors=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            return mu, unc
        else:
            return mu
      

class QuasiPeriodic(GaussianProcess):
    """A Quasiperiodic GP.
    """
    
    def compute_cov_matrix(self, pars, apply_errors=True):
        
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
        if apply_errors:
            errors = self.compute_data_errors(pars)
            errors_quad = np.diag(cov_matrix) + errors**2
            np.fill_diagonal(cov_matrix, errors_quad)
        
        return cov_matrix