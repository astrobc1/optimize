import optimize.knowledge
import optimize.kernels as optnoisekernels
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import matplotlib.pyplot as plt


class ScoreFunction:
    """An base class for a general score function. Not useful to instantiate on its own.
    
    Attributes:
        data (MixedData): A combined dataset.
        model (Model): A model inheriting from optimize.models.Model. All datasets must use this model.
    """
    
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
    
    def set_pars(self, pars):
        self.model.set_pars(pars)
        
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
    """A Bayesian likelihood score function.
    """
    
    def __init__(self, label=None, data=None, model=None):
        super().__init__(data=data, model=model)
        self.label = label
        self.data_x = self.data.get_vec('t')
        self.data_y = self.data.get_vec('rv')
        self.data_yerr = self.data.get_vec('rverr')
            
    def compute_score(self, pars):
        """Computes the negative log-likelihood score.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        neglnL = self.compute_neglogL(pars)
        return neglnL
    
    def compute_logL(self, pars, apply_priors=True):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """
        
        # Apply priors, see if we even need to compute the model
        if apply_priors:
            lnL = self.compute_logL_priors(pars)
            if not np.isfinite(lnL):
                return -np.inf
        else:
            lnL = 0
        
        # Compute the model (consistent across all data sets for this likelihood).
        model_arr = self.model.build(pars)
        
        # Copy the full data set
        data_arr = np.copy(self.data.y)

        # Compute the residuals
        residuals = data_arr - model_arr
            
        # Compute the cov matrix
        K = kernel.compute_cov_matrix(pars, apply_errors=True)

        # Compute the determiniant and inverse of K
        try:
        
            # Reduce the cov matrix and solve for KX = residuals
            alpha = cho_solve(cho_factor(K), residuals)

            # Compute the determinant of K
            _, detK = np.linalg.slogdet(K)

            # Compute the likelihood
            N = len(_data)
            lnL += -0.5 * (np.dot(residuals, alpha) + detK + N * np.log(2 * np.pi))
    
        except:
            # If things fail (matrix decomp) return -inf
            return -np.inf
        
        # Return the final ln(L)
        return lnL
    
    def residuals_after_kernel(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals = self.residuals_before_kernel(pars)
        x_data = self.data.get_vec(key='x')
        if self.model.has_gp:
            gpmean = self.model.kernel.realize(pars, residuals, xpred=x_data, xres=x_data, return_unc=False)
            residuals -= gpmean
        return residuals
    
    def compute_neglogL(self, pars):
        """Simple wrapper to compute -ln(L).

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The negative log likelihood, -ln(L).
        """
        return -1 * self.compute_logL(pars)
    
    def compute_ndeg(self, pars):
        """Computes the number of degrees of freedom, n_data_points - n_vary_pars.

        Returns:
            int: The degrees of freedom.
        """
        return len(self.data.x) - pars.num_varied()
    
    def compute_logL_priors(self, pars):
        lnL = 0
        for par in pars:
            _par = pars[par]
            for prior in _par.priors:
                lnL += prior.logprob(_par.value)
                if not np.isfinite(lnL):
                    return lnL
        return lnL
    
    def compute_bic(self, pars):
        """Calculate the Bayesian information criterion (BIC).

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            float: The BIC
        """

        n = len(self.data.rv)
        k = len(pars)
        lnL = self.compute_logL_priors(pars)
        _bic = np.log(n) * k - 2.0 * lnL
        return _bic

    def compute_aicc(self, pars):
        """Calculate the small sample Akaike information criterion (AICc).
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The AICc.
        """
        
        # Simple formula
        n = len(self.data.rv)
        k = pars.num_varied()
        lnL = self.compute_logL_priors(pars)
        aic = - 2.0 * lnL + 2.0 * k
        
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
        

class MixedLikelihood(dict):
    """A class for joint likelihood functions. This should map 1-1 with the kernels map.
    """
    
    def __init__(self):
        super().__init__()
    
    def __setitem__(self, label, like):
        """Overrides the default Python dict setter.

        Args:
            label (str): How to identify this likelihood.
            like (Likelihood): The likelihood object to set.
        """
        if like.label is None:
            like.label = label
        super().__setitem__(label, like)
        
    def compute_score(self, pars):
        """Computes the negative log-likelihood score.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        neglnL = 0
        for like in self.values():
            neglnL += like.compute_neglogL(pars)
        return neglnL
    
    def compute_logL(self, pars, apply_priors=True):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """
        lnL = 0
        for like in self.values():
            lnL += like.compute_logL(pars, apply_priors=apply_priors)
        return lnL
    
    def compute_neglogL(self, pars):
        """Simple wrapper to compute -ln(L).

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The negative log likelihood, -ln(L).
        """
        return -1 * self.compute_logL(pars)
    
    def set_pars(self, pars):
        for like in self.values():
            like.set_pars(pars)
          
    @property  
    def p0(self):
        return next(iter(self.values())).p0
    
    def compute_bic(self, pars):
        """Calculate the Bayesian information criterion (BIC).

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            float: The BIC
        """
        n = 0
        for like in self.values():
            n += len(like.data_x)
        k = pars.num_varied()
        lnL = self.compute_logL(pars, apply_priors=True)
        bic = np.log(n) * k - 2.0 * lnL
        return bic

    def compute_aicc(self, pars):
        """Calculate the small sample Akaike information criterion (AICc).
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The AICc.
        """
        
        # Number of data points
        n = 0
        for like in self.values():
            n += len(like.data_x)
            
        # Number of estimated parameters
        k = pars.num_varied()
        
        # lnL
        lnL = self.compute_logL(pars, apply_priors=True)
        
        # AIC
        aic = - 2.0 * lnL + 2.0 * k

        # Small sample correction
        aicc = aic
        denom = n - k - 1
        if denom > 0:
            aicc += (2.0 * k * (k + 1.0)) / denom
        else:
            print("Warning: The number of free parameters is greater than or equal to")
            print("         the number of data points (- 1). The AICc comparison has returned -inf.")
            aicc = np.inf
        return aicc