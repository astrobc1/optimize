import numpy as np
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse
from numba import jit, njit

class NoiseKernel:
    
    def __init__(self):
        pass
    
    
    @staticmethod
    @jit
    def compute_dist_matrix(x1, x2):
        n1 = len(x1)
        n2 = len(x2)
        out = np.zeros((n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                out[i, j] = np.abs(x1[i] - x2[j])
        return out
    
    
class WhiteNoiseKernel(NoiseKernel):
    
    def __init__(self):
        pass
    
    def compute_cov_matrix(self, pars, errorbars):
        return scipy.sparse.diags(errorbars**2)
    
    
class GaussianProcess(NoiseKernel):
    
    def __init__(self):
        super().__init__()
        
    def predict(self, xpred, xres, res, errorbars, stddev=False):
        
        # Get K
        K = self.kernel.compute_cov_matrix(errorbars)
        
        # Compute version of K without errorbars
        self.kernel.compute_dist_matrix(xpred, xres)
        Ks = self.kernel.compute_cov_matrix(0.0)

        L = cho_factor(K)
        alpha = cho_solve(L, res)
        mu = np.dot(Ks, alpha).flatten()

        if stddev:
            self.kernel.compute_distances(xpred, xpred)
            Kss = self.kernel.compute_covmatrix(0.0)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            return mu, stddev
        else:
            return mu
        
    