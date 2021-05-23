import optimize.knowledge
import optimize.noise as optnoise
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import optimize.objectives as optobj
import matplotlib.pyplot as plt

class Likelihood(optobj.MaxObjectiveFunction):
    """A Bayesian likelihood objective function.
    """
    
    def __init__(self, data, model, noise, p0, label=None):
        
        # Super
        super().__init__(data=data, model=model, p0=p0)
        
        # Store the label for this likelihood.
        self.label = label
        
        # Store the noise kernel for this likelihood.
        self.noise = noise

    def compute_obj(self, pars):
        """Computes the log-likelihood.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        return self.compute_logL(pars)
    
    def compute_logL(self, pars):
        """Computes the log of the likelihood.
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The log likelihood, ln(L).
        """

        # Compute the residuals
        residuals_with_noise = self.compute_data_pre_noise_process(pars)

        # Compute the cov matrix
        K = self.noise.compute_cov_matrix(pars)
        
        # Compute the determiniant and inverse of K
        try:
        
            # Reduce the cov matrix and solve for KX = residuals
            alpha = cho_solve(cho_factor(K), residuals_with_noise)

            # Compute the log determinant of K
            _, lndetK = np.linalg.slogdet(K)

            # Compute the likelihood
            N = len(residuals_with_noise)
            lnL = -0.5 * (np.dot(residuals_with_noise, alpha) + lndetK + N * np.log(2 * np.pi))
    
        except:
            # If things fail (matrix decomp) return -inf
            lnL = -np.inf
        
        # Return the final ln(L)
        return lnL
    
    def compute_data_pre_noise_process(self, pars):
        """Computes the data containing a noise process by subtracting off the base model.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The data pre noise process.
        """
        model_arr = self.model.build(pars)
        residuals = self.data_y - model_arr
        return residuals
    
    def compute_data_post_noise_process(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        
        # Get the data containing only noise
        data_pre_noise_process = self.compute_data_pre_noise_process(pars)
        
        # Copy the data 
        data_post_noise_process = np.copy(data_pre_noise_process)
        
        # If noise is correlated, the mean may not be zero, so realize the noise process and subtract.
        if isinstance(self.noise, optnoise.CorrelatedNoise):
            noise_process_mean = self.noise.realize(pars, data_pre_noise_process)
            data_post_noise_process -= noise_process_mean
            
        return data_post_noise_process
    
    def compute_n_dof(self, pars):
        """Computes the number of degrees of freedom, n_data_points - n_vary_pars.

        Returns:
            int: The degrees of freedom.
        """
        return len(self.data_x) - pars.num_varied()
    
    def __repr__(self):
        repr(self.data)
        repr(self.model)
    
    def set_pars(self, pars):
        """Sets the current parameters attribute.

        Args:
            pars (Parameters): The parameters object
        """
        self.p0 = pars
    
    
class PureGPLikelihood(Likelihood):
    """A Bayesian likelihood objective function.
    """
    
    def __init__(self, data, model, noise, p0, label=None):
        
        # Super
        super().__init__(data=data, model=model, p0=p0, noise=noise)
        
        self.data_x = data.gen_vec("x")
        self.data_y = data.gen_vec("y")
        self.data_yerr = data.gen_vec("yerr")
    
    def compute_logL(self, pars):
        """Computes the log of the likelihood.
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The log likelihood, ln(L).
        """

        # Compute the residuals
        residuals_with_noise = self.compute_data_pre_noise_process(pars)

        # Compute the cov matrix
        K = self.noise.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute the determiniant and inverse of K
        try:
            
            # Reduce the cov matrix and solve for KX = residuals
            alpha = cho_solve(cho_factor(K), residuals_with_noise)

            # Compute the log determinant of K
            _, lndetK = np.linalg.slogdet(K)

            # Compute the likelihood
            N = len(residuals_with_noise)
            lnL = -0.5 * (np.dot(residuals_with_noise, alpha) + lndetK + N * np.log(2 * np.pi))
    
        except:
            # If things fail (matrix decomp) return -inf
            lnL = -np.inf
        
        # Return the final ln(L)
        return lnL
    
    def compute_data_pre_noise_process(self, pars):
        """Computes the data containing a noise process by subtracting off the base model.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The data pre noise process.
        """
        model_arr = self.model.build(pars)
        residuals = self.data_y - model_arr
        return residuals
    
    def compute_data_post_noise_process(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        
        # Get the data containing only noise
        data_pre_noise_process = self.compute_data_pre_noise_process(pars)
        
        # Copy the data 
        data_post_noise_process = np.copy(data_pre_noise_process)
        
        # If noise is correlated, the mean may not be zero, so realize the noise process and subtract.
        if isinstance(self.noise, optnoise.CorrelatedNoise):
            noise_process_mean = self.noise.realize(pars, data_pre_noise_process)
            data_post_noise_process -= noise_process_mean
            
        return data_post_noise_process
    
    def compute_n_dof(self, pars):
        """Computes the number of degrees of freedom, n_data_points - n_vary_pars.

        Returns:
            int: The degrees of freedom.
        """
        return len(self.data_x) - pars.num_varied()
    
    def __repr__(self):
        repr(self.data)
        repr(self.model)
    
    def set_pars(self, pars):
        """Sets the current parameters attribute.

        Args:
            pars (Parameters): The parameters object
        """
        self.p0 = pars
  
class Posterior(dict, optobj.MaxObjectiveFunction):
    """A class for joint likelihood functions. This should map 1-1 with the kernels map.
    """
    
    def __setitem__(self, label, like):
        """Overrides the default Python dict setter.

        Args:
            label (str): How to identify this likelihood.
            like (Likelihood): The likelihood object to set.
        """
        if like.label is None:
            like.label = label
        super().__setitem__(label, like)
        
    def compute_obj(self, pars):
        """Computes the log-likelihood score.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        return self.compute_logaprob(pars)
    
    def compute_prior_logL(self, pars):
        lnL = 0
        for par in pars.values():
            if par.vary:
                for prior in par.priors:
                    lnL += prior.logprob(par.value)
                    if not np.isfinite(lnL):
                        return -np.inf
        return lnL
    
    def compute_logaprob(self, pars):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """
        lnL = self.compute_prior_logL(pars)
        if not np.isfinite(lnL):
            return -np.inf
        lnL += self.compute_logL(pars)
        if not np.isfinite(lnL):
            return -np.inf
        return lnL
    
    def compute_logL(self, pars):
        lnL = 0
        for like in self.values():
            lnL += like.compute_logL(pars)
            if not np.isfinite(lnL):
                return -np.inf
        return lnL
    
    def set_pars(self, pars):
        for like in self.values():
            like.set_pars(pars)

    def compute_redchi2(self, pars):
        """Computes the reduced chi2 statistic (weighted MSE).

        Args:
            pars (Parameters): The parameters.

        Returns:
            float: The reduced chi-squared statistic.
        """
        
        chi2 = 0
        n_dof = 0
        for like in self.values():
            residuals = like.noise.compute_data_post_noise_process(pars)
            errors = like.noise.compute_data_errors(pars)
            chi2 += optscore.MSE.compute_chi2(residuals, errors)
            n_dof += len(like.data.gen_vec("x"))
        n_dof -= pars.num_varied()
        redchi2 = chi2 / n_dof
        return redchi2
      
    def compute_bic(self, pars):
        """Calculate the Bayesian information criterion (BIC).

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            float: The BIC
        """
        n = 0
        for like in self.values():
            n += len(like.data.gen_vec("x"))
        k = pars.num_varied()
        lnL = self.compute_logL(pars)
        bic = k * np.log(n) - 2.0 * lnL
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
            n += len(like.data.gen_vec("x"))
            
        # Number of estimated parameters
        k = pars.num_varied()
        
        # lnL
        lnL = self.compute_logL(pars)
        
        # AIC
        aic = 2.0 * (k - lnL)

        # Small sample correction
        aicc = aic
        d = n - k - 1
        if d > 0:
            aicc += (2 * k**2 + 2 * k) / d
        else:
            aicc = np.inf

        return aicc
      
    @property
    def p0(self):
        return self.like0.p0
    
    @property
    def like0(self):
        return next(iter(self.values()))
    
    @property
    def likes(self):
        return self.values()
