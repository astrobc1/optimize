import optimize.knowledge
import optimize.kernels as optnoisekernels
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import matplotlib.pyplot as plt


class ScoreFunction:
    """An base class for a general score function. Not useful to instantiate on its own.
    
    Attributes:
        data (Data): A dataset inheriting from optimize.data.Data.
        model (Model): A model inheriting from optimize.models.Model.
    """
    
    __children__ = ['data', 'model']
    
    def __init__(self, data=None, model=None):
        """Stores the basic requirements for a score function.

        Args:
            data (Data): A dataset inheriting from optimize.data.Data.
            model (Model): A model inheriting from optimize.models.Model.
        """
        self.data = data
        self.model = model

    def compute_score(self, pars):
        """Computes the score from a given set of parameters.

        Args:
            pars (Parameters): The parameters to use.

        Raises:
            NotImplementedError: Must implement this method.
        """
        raise NotImplementedError("Must implement a compute_score method.")
        
class MSE(ScoreFunction):
    """A class for the standard mean squared error (MSE) loss.
    """
    
    def compute_score(self, pars):
        """Computes the unweighted mean squared error loss.

        Args:
            pars (Parameters): The parameters object to use.

        Returns:
            float: The score.
        """
        _model = self.model.build(pars)
        _data = self.data.y
        rms = self.compute_rms(_data, _model)
        return rms
    
    @staticmethod
    def compute_rms(_data, _model):
        """Computes the RMS (Root mean squared) loss.

        Args_data 
            _data (np.ndarray): The data.
            _model (np.ndarray): The model.

        Returns:
            float: The RMS.
        """
        return np.sqrt(np.nansum((_data - _model)**2) / _data.size)
    
    @staticmethod
    def compute_chi2(res, errors):
        """Computes the (non-reduced) chi2 statistic (weighted MSE).

        Args:
            res (np.ndarray): The residuals (data - model)
            errors (np.ndarray): The effective errorbars (intrinsic and any white noise).

        Returns:
            float: The chi-squared statistic.
        """
        return np.nansum((res / errors)**2)
    
    @staticmethod
    def compute_redchi2(res, errors, ndeg=None):
        """Computes the reduced chi2 statistic (weighted MSE).

        Args:
            res (np.ndarray): The residuals (data - model)
            errors (np.ndarray): The effective errorbars (intrinsic and any white noise).
            ndeg (int): The degrees of freedom, defaults to len(res) - 1.

        Returns:
            float: The reduced chi-squared statistic.
        """
        if ndeg is None:
            ndeg = len(res) - 1
        _chi2 = np.nansum((res / errors)**2)
        return _chi2 / ndeg


class Likelihood(ScoreFunction):
    """A Bayesian likelihood score function, Priors are also considered.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    def compute_score(self, pars):
        """Computes the negative log-likelihood score.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        _neglnL = self.compute_negloglikelihood(pars)
        return _neglnL
    
    def compute_loglikelihood(self, pars, apply_priors=True):
        """Computes the log of the likelihood.

        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """
        
        # Apply priors, see if we even need to compute the model
        if apply_priors:
            _lnL = self.compute_loglikelihood_priors(pars)
            if not np.isfinite(_lnL):
                return _lnL
        else:
            _lnL = 0
        
        # Compute the model
        _model = self.model.build(pars)
        
        # Copy the data
        _data = np.copy(self.data.y)
            
        # Compute the residuals
        _res = _data - _model
            
        # Compute the error bars and cov matrix
        K = self.model.kernel.compute_cov_matrix(pars, self.compute_errorbars(pars))
            
        # Compute the determiniant and inverse of K
        try:
            
            # Reduce the cov matrix and solve for KX = _RES
            alpha = cho_solve(cho_factor(K), _res)

            # Compute the determinant of K
            _, detK = np.linalg.slogdet(K)

            # Compute the likelihood
            N = len(_data)
            _lnL = -0.5 * (np.dot(_res, alpha) + detK + N * np.log(2 * np.pi))
            
            # Return ln(L)
            return _lnL
        
        except:
            # If things fail (matrix decomp) return -inf
            return -np.inf
    
    def compute_negloglikelihood(self, pars):
        """Simple wrapper to compute -ln(L).

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The negative log likelihood, -ln(L).
        """
        return -1 * self.compute_loglikelihood(pars)
    
    def compute_ndeg(self, pars):
        """Computes the number of degrees of freedom, n_data_points - n_vary_pars.

        Returns:
            int: The degrees of freedom.
        """
        return len(self.data.x) - pars.num_varied()
    
    def compute_loglikelihood_priors(self, pars):
        _lnL = 0
        for par in pars:
            _par = pars[par]
            for prior in _par.priors:
                _lnL += prior.logprob(_par.value)
        return _lnL
    
    def compute_bic(self, pars):
        """Calculate the Bayesian information criterion (BIC).

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            float: The BIC
        """

        n = len(self.data.rv)
        k = len(pars.num_varied())
        _lnL = self.compute_loglikelihood_priors(pars)
        _bic = np.log(n) * k - 2.0 * _lnL
        return _bic

    def compute_aicc(self, pars, ):
        """Calculate the small sample Akaike information criterion (AICc).
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The AICc.
        """
        
        # Simple formula
        n = len(self.data.rv)
        k = len(pars.num_varied())
        _lnL = self.compute_loglikelihood_priors(pars)
        aic = - 2.0 * _lnL + 2.0 * k
        
        # Small sample correction
        _aicc = aic
        denom = (n - k - 1.0)
        if denom > 0:
            _aicc += (2.0 * k * (k + 1.0)) / denom
        else:
            print("Warning: The number of free parameters is greater than or equal to")
            print("         the number of data points (- 1). The AICc comparison has returned -inf.")
            _aicc = np.inf
        return _aicc
        
        
         