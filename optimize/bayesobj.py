
# Maths
import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Plots
import matplotlib.pyplot as plt

# Optimize deps
from optimize.noise import CorrelatedNoiseProcess
from optimize.objectives import MaxObjectiveFunction
import optimize.maths as optmath


####################
#### Likelihood ####
####################

class Likelihood(MaxObjectiveFunction):
    """A Bayesian likelihood objective function.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, label=None, model=None):
        
        # Super
        super().__init__(model=model)
        
        # Label for the likelihood.
        self.label = label
        
    #####################
    #### COMPUTE OBJ ####
    #####################

    def compute_obj(self, pars):
        """Forwards to compute_logL, the log-likelihood.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        return self.compute_logL(pars)
        
    def compute_logL(self, pars):
        raise NotImplementedError(f"Must implement the method compute_logL for class {self.__class__.__name__}")
                
    ###############
    #### MISC. ####
    ###############
    
    def compute_n_dof(self, pars):
        """Computes the number of degrees of freedom, n_data_points - n_vary_pars.

        Returns:
            int: The degrees of freedom.
        """
        return len(self.data.get_trainable()) - pars.num_varied
    
    def __repr__(self):
        return "Likelihood"


class GaussianLikelihood(Likelihood):
    """A Bayesian likelihood objective function.
    """
        
    #####################
    #### COMPUTE OBJ ####
    #####################
    
    def compute_logL(self, pars):
        """Computes the log of the likelihood.
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The log likelihood, ln(L).
        """

        # Compute the residuals
        residuals = self.model.compute_raw_residuals(pars)
        n = len(residuals)
        
        # Compute the determiniant and inverse of K
        try:

            if isinstance(self.model.noise_process, CorrelatedNoiseProcess):
                
                # Compute the cov matrix
                K = self.model.noise_process.compute_cov_matrix(pars)

                # Reduce the cov matrix
                alpha = cho_solve(cho_factor(K), residuals)

                # Compute the log determinant of K
                _, lndetK = np.linalg.slogdet(K)

                # Compute the Gaussian likelihood
                lnL = -0.5 * (np.dot(residuals, alpha) + lndetK + n * np.log(2 * np.pi))

                # Return
                return lnL
            
            else:
                
                # Errors
                errors = self.model.compute_data_errors(pars)
                
                # LogL for uncorrelated errors
                lnL = -0.5 * (np.nansum((residuals / errors)**2) + np.nansum(np.log(errors**2)) + n * np.log(2 * np.pi))
                
                # Return
                return lnL
    
        except scipy.linalg.LinAlgError:
            
            # If things fail, return -inf
            return -np.inf
    
    def __repr__(self):
        return "Gaussian Likelihood"


###################
#### POSTERIOR ####
###################

class Posterior(dict, MaxObjectiveFunction):
    """A class for joint, additive log-likelihood classes.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self):
        dict.__init__(self)
    
    #####################
    #### COMPUTE OBJ ####
    #####################

    def compute_obj(self, pars):
        """Computes the log of the a posteriori probability.
        
        Args:
            pars (Parameters): The parameters.

        Returns:
            float: ln(L).
        """
        return self.compute_logaprob(pars)
    
    def compute_prior_logprob(self, pars):
        lnL = 0
        for par in pars.values():
            if par.vary:
                for prior in par.priors:
                    lnL += prior.logprob(par.value)
                    if not np.isfinite(lnL):
                        return -np.inf
        return lnL
    
    def compute_logaprob(self, pars):
        """Computes the log of the a posteriori probability.
    
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The log likelihood, ln(L).
        """

        lnL = self.compute_prior_logprob(pars)
        if not np.isfinite(lnL):
            return -np.inf
        lnL += self.compute_logL(pars)
        if not np.isfinite(lnL):
            return -np.inf
        
        return lnL
    
    def compute_logL(self, pars):
        lnL = 0
        for like in self.likes:
            lnL += like.compute_logL(pars)
            if not np.isfinite(lnL):
                return -np.inf
        return lnL
    
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
            residuals = like.model.compute_residuals(pars)
            errors = like.model.compute_data_errors(pars)
            chi2 += optmath.compute_chi2(residuals, errors)
            n_dof += len(residuals)
        n_dof -= pars.num_varied
        redchi2 = chi2 / n_dof
        return redchi2
      
    def compute_bic(self, pars):
        """Calculate the Bayesian information criterion (BIC).

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            float: The BIC.
        """
        n = 0
        for like in self.likes:
            n += len(like.model.data.get_trainable())
        k = pars.num_varied
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
            n += len(like.model.data.get_trainable())
            
        # Number of estimated parameters
        k = pars.num_varied
        
        # lnL
        lnL = self.compute_logL(pars)
        
        # AIC
        aic = 2.0 * (k - lnL)

        # Small sample correction
        d = n - k - 1
        if d > 0:
            aicc = aic + (2 * k**2 + 2 * k) / d
        else:
            aicc = np.inf

        return aicc
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0):
        self.p0 = p0
        for like in self.values():
            like.initialize(self.p0)

    ###############
    #### MISC. ####
    ###############
    
    def __setitem__(self, label, like):
        """Overrides the default Python dict setter.

        Args:
            label (str): How to identify this likelihood.
            like (Likelihood): The likelihood object to set.
        """
        if like.label is None:
            like.label = label
        super().__setitem__(label, like)
    
    @property
    def like0(self):
        return next(iter(self.values()))
    
    @property
    def likes(self):
        return self.values()

    def __repr__(self):
        s = ""
        for like in self.values():
            s += repr(like) + "\n"
        return s
