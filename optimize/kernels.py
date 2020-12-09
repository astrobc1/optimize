import numpy as np
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse
from numba import jit, njit
import matplotlib.pyplot as plt

class NoiseKernel:
    """A base noise kernel class. This class is not useful to instantiate on its own.
    """
        
    def compute_dist_matrix(self, x1, x2):
        self.dist_matrix = self._compute_dist_matrix(x1, x2)
    
    
class WhiteNoise(NoiseKernel):
    """A noise kernel for white noise, where all diagonal terms in the covariance matrix are zero. The noise kernel is computed by adding a jitter term and the intrinsic error bars in quadrature.
    """
    
    def compute_cov_matrix(self, pars, errors):
        n = len(errors)
        cov_matrix = np.zeros(shape=(n, n), dtype=float)
        np.fill_diagonal(cov_matrix, errors**2)
        return cov_matrix
    
class GaussianProcess(NoiseKernel):
    """A generic Gaussian process kernel.
    """
        
    def realize(self, pars, xpred, xres, res, errors, stddev=False):
        """Realize the GP (sample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            res (np.ndarray): The residuals before the GP is subtracted.
            errors (np.ndarray): The instrinsic errorbars.
            stddev (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization (i.e, minimiation surface is purely concave or convex).
        """
        
        # Get K
        self.compute_dist_matrix(xres, xres)
        K = self.compute_cov_matrix(pars, errors=errors)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres)
        Ks = self.compute_cov_matrix(pars, errors=None)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, res)
        mu = np.dot(Ks, alpha).flatten()

        if stddev:
            self.compute_dist_matrix(xpred, xpred)
            Kss = self.compute_cov_matrix(pars, errors=None)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            return mu, stddev
        else:
            return mu
        
    @staticmethod
    def _compute_dist_matrix(x1, x2):
        """Computes the distance matrix, D. D_ij = |t_i - t_j|

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
        

class QuasiPeriodic(GaussianProcess):
    
    def compute_cov_matrix(self, pars, errors=None):
        
        # Alias params
        amp = pars["gp_amp"].value
        exp_length = pars["gp_decay"].value
        per = pars["gp_per"].value
        per_length = pars["gp_per_length"].value
        
        # Compute exp decay term
        decay_term = -0.5 * self.dist_matrix**2 / exp_length**2
        
        # Compute periodic term
        periodic_term = -0.5 * np.sin((2 * np.pi / per) * self.dist_matrix)**2 / per_length**2
        
        # Add and include amplitude
        cov_matrix = amp**2 * np.exp(decay_term + periodic_term)
        
        # Include errors on the diagonal
        if errors is not None and errors.size == cov_matrix.shape[0]:
            errors_quad = np.diag(cov_matrix) + errors**2
            np.fill_diagonal(cov_matrix, errors_quad)
        
        return cov_matrix