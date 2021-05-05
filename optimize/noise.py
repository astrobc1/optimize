import numpy as np
from scipy.linalg import cho_solve, cho_factor
import numba
import matplotlib.pyplot as plt

class NoiseKernel:
    """A base noise kernel class defined through a single covariance matrix.
    """
    def __init__(self, data):
        self.data = data
        
    def compute_cov_matrix(self, *args, **kwargs):
        raise NotImplementedError(f"Must implemenent the method compute_cov_matrix for class {self.__class__.__name__}.")
    
    def initialize(self, x1=None, x2=None):
        """Default wrapper to initialize the distance matrix. By default, only the distance matrix is constructed.

        Args:
            x1 (np.ndarray, optional): The x1 vector. Defaults to the Data grid (self.x).
            x2 (np.ndarray, optional): The x2 vector. Defaults to the Data grid (self.x).
        """
        if x1 is None:
            x1 = self.data.gen_vec("x")
        if x2 is None:
            x2 = x1
        self.dist_matrix = compute_stationary_dist_matrix(x1, x2)

class NoiseProcess:
    """A base noise process class defined through one or multiple covariance matrices. This class is not useful to instantiate on its own.
    """
    def __init__(self, data):
        self.data = data
        
    def compute_cov_matrix(self, *args, **kwargs):
        raise NotImplementedError(f"Must implemenent the method compute_cov_matrix for class {self.__class__.__name__}.")

    
class QuasiPeriodic(NoiseKernel):
    """A Quasiperiodic GP. The hyperparameters may be called anything, but must be in the order of amplitude, exp length scale, period, period length scale.
    """
    
    def __init__(self, data, par_names):
        super().__init__(data)
        self.par_names = par_names
    
    def compute_cov_matrix(self, pars):
        
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
        K = amp**2 * np.exp(decay_term + periodic_term)
        
        return K
    
    
class QuasiPeriodicMod(NoiseKernel):
    """A Quasiperiodic GP. The hyperparameters may be called anything, but must be in the order of amplitude, exp length scale, period, period length scale.
    """
    
    def __init__(self, data, par_names):
        super().__init__(data)
        self.par_names = par_names
    
    def compute_cov_matrix(self, pars):
        
        # Alias params
        amp = pars[self.par_names[0]].value
        ts = pars[self.par_names[1]].value
        per = pars[self.par_names[2]].value
        C = pars[self.par_names[3]].value
        
        # Construct K
        K = (amp**2 / (2 + C)) * np.exp(-self.dist_matrix / ts) * (np.cos((2 * np.pi / per) * self.dist_matrix) + (1 + C))
        
        return K


class CorrelatedNoiseProcess(NoiseProcess):
    
    def __init__(self, data):
        super().__init__(data=data)
    
    def compute_cov_matrix(self, *args, **kwargs):
        raise NotImplementedError("Must implement the method compute_cov_matrix")


class GaussianProcess(CorrelatedNoiseProcess):
    """A noise kernel defined through a single GP and diagonal error terms with an additional "jitter" parameter. Each jitter parameter must be named "jitter_label" where label is the data label.
    """
    
    def __init__(self, data, kernel):
        super().__init__(data)
        self.kernel = kernel
        self.data_inds = self.data.gen_inds_dict()
        self.initialize()
        
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
    
    def compute_data_errors(self, pars, include_gp_error=False, gp_error=None, data_with_noise=None):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.
            include_gp_error (bool, optional): Whether or not to include the gp error. Defaults to False.
            gp_error (np.ndarray, optional): The GP error. Defaults to False.
            include_gp_error (bool, optional): Whether or not to add in quadrature the gp error.
            data_with_noise (np.ndarray): The data containings noise to use in order to realize the noise process if include_uncorr_error is True.
            
        Returns:
            np.ndarray: The final data errors.
        """
    
        # Get intrinsic data errors
        errors = self.get_intrinsic_data_errors()
        
        # Square
        errors = errors**2
        
        # Compute additional per-label jitter
        for label in self.data:
            inds = self.data_inds[label]
            pname = f"jitter_{label}"
            errors[inds] += pars[pname].value**2
            
        # Compute correlated error
        if include_gp_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                if gp_error is None:
                    _, _gp_error = self.realize(pars, data_with_noise=data_with_noise, xpred=data.t, return_gp_error=True)
                    errors[inds] += _gp_error**2
                else:
                    errors[inds] += gp_error[inds]**2
                    
        # Square root
        errors **= 0.5

        return errors
    
    def get_intrinsic_data_errors(self):
        """Generates the intrinsic data errors (measured apriori).

        Returns:
            np.ndarray: The intrinsic data error bars.
        """
        errors = np.zeros(self.data.n)
        for data in self.data.values():
            inds = self.data_inds[data.label]
            errors[inds] = data.yerr
        return errors
    
    def realize(self, pars, data_with_noise, xdata=None, xpred=None, return_gp_error=False):
        """Realize the GP (sample at arbitrary points).

        Args:
            pars (Parameters): The parameters to use.
            data_with_noise (np.ndarray): The data vector containing noise.
            xpred (np.ndarray): The independent variable vector to realize the GP on.
            xdata (np.ndarray): The vector the data is on.
            return_gp_error (bool, optional): Whether or not to compute and return the uncertainty in the GP. If True, both the mean and the 1-sigma unc vectors are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: The mean or mean and uncertainty in a tuple.
        """
        
        # Resolve the grids to use.
        if xdata is None:
            xdata = self.data.gen_vec("x")
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(x1=xdata, x2=xdata)
        K = self.compute_cov_matrix(pars)
        
        # Compute version of K without errorbars
        self.initialize(x1=xpred, x2=xdata)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, data_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_gp_error:
            self.initialize(x1=xpred, x2=xpred)
            Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.initialize()
            return mu, unc
        else:
            self.initialize()
            return mu

    def initialize(self, x1=None, x2=None):
        self.kernel.initialize(x1=x1, x2=x2)


@numba.njit(nogil=True)
def compute_stationary_dist_matrix(x1, x2):
    """Computes the distance matrix, D_ij = |x_i - x_j|. This function is compiled via @numba.njit(nogil=True)

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