
# Maths
import numpy as np
import scipy.linalg
from scipy.linalg import cho_factor, cho_solve

# Optimize deps
from optimize.objectives import ObjectiveFunction
from optimize.noise import UnCorrelatedNoiseProcess

TWO_PI = 2 * np.pi
LOG_2PI = np.log(TWO_PI)


####################
#### Likelihood ####
####################

class Likelihood(ObjectiveFunction):
    """A Bayesian likelihood objective function.
    """

    def __init__(self, noise_process=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_process = noise_process

    def compute_obj(self, *args, **kwargs):
        return self.compute_logL(*args, **kwargs)

    def compute_logL(self, *args, **kwargs):
        raise NotImplementedError(f"Must implement method compute_logL for class {self.__class__.__name__}")

    def compute_data_errors(self, *args, **kwargs):
        raise NotImplementedError(f"Must implement method compute_data_errors for class {self.__class__.__name__}")

    def compute_dof(self, residuals, pars):
        n_good = np.where(np.isfinite(residuals) & (residuals != 0))[0].size
        n_dof = n_good - pars.num_varied
        return n_dof

    def compute_cov_matrix(self, *args, **kwargs):
        return self.noise_process.compute_cov_matrix(*args, **kwargs)

    @staticmethod
    def redchi2loss(residuals, errors, n_dof):
        return np.nansum((residuals / errors)**2) / n_dof
        
    def compute_logL(self, pars):
        raise NotImplementedError(f"Must implement the method compute_logL for class {self.__class__.__name__}")

    @property
    def datax(self):
        raise NotImplementedError(f"Must implement the property datax for class {self.__class__.__name__}")

    @property
    def datay(self):
        raise NotImplementedError(f"Must implement the property datay for class {self.__class__.__name__}")
    
    @property
    def datayerr(self):
        raise NotImplementedError(f"Must implement the property datay for class {self.__class__.__name__}")
                
    def __repr__(self):
        return "Generic Likelihood"


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
        residuals = self.compute_residuals(pars)
        errors = self.compute_data_errors(pars)
        n = len(residuals)

        # Check if noise is correlated
        if isinstance(self.noise_process, UnCorrelatedNoiseProcess):
            lnL = -0.5 * (np.sum((residuals / errors)**2) + np.sum(np.log(errors**2)) + n * LOG_2PI)
            return lnL
        
        else:
        
            # Compute the determiniant and inverse of K
            try:
                    
                # Compute the cov matrix
                K = self.compute_cov_matrix(pars, self.datax, self.datax, include_uncorrelated_error=True)

                # Reduce the cov matrix
                alpha = cho_solve(cho_factor(K), residuals)

                # Compute the log determinant of K
                _, lndetK = np.linalg.slogdet(K)

                # Compute the Gaussian likelihood
                lnL = -0.5 * (np.dot(residuals, alpha) + lndetK + n * LOG_2PI)

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

class Posterior(ObjectiveFunction):
    """A class for joint, additive log-likelihood classes.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, likes=None):
        self.likes = likes
    
    #####################
    #### COMPUTE OBJ ####
    #####################

    def compute_obj(self, *args, **kwargs):
        return self.compute_logaprob(*args, **kwargs)
    
    def compute_prior_logprob(self, pars):
        return pars.compute_prior_logprob()
    
    def compute_logaprob(self, pars):
        lnL = self.compute_prior_logprob(pars)
        if not np.isfinite(lnL):
            return -np.inf
        lnL += self.compute_logL(pars)
        if not np.isfinite(lnL):
            return -np.inf
        
        return lnL
    
    def compute_logL(self, pars):
        lnL = 0
        for like in self.likes.values():
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
        for like in self.likes.values():
            residuals = like.compute_residuals(pars)
            errors = like.compute_data_errors(pars)
            chi2 += np.nansum((residuals / errors)**2)
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
        for like in self.likes.values():
            n += len(like.model.data.t)
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
        for like in self.likes.values():
            n += len(like.model.data.t)
            
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
    
    ###############
    #### MISC. ####
    ###############
    
    @property
    def like0(self):
        return next(iter(self.likes.values()))

    def __repr__(self):
        s = "Posterior with Likelihoods:\n"
        for like in self.likes.values():
            s += f"{like}\n"
        return s
