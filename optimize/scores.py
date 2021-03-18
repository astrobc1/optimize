import optimize.knowledge
import optimize.kernels as optnoisekernels
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import matplotlib.pyplot as plt


class ScoreFunction:
    """An base class for a general score function. Not useful to instantiate on its own.
    
    Attributes:
        data (CompositeData): A combined dataset.
        model (Model): A model inheriting from optimize.models.Model. All datasets must use this model.
    """
    
    def __init__(self, data=None, model=None):
        """Stores the basic requirements for a score function.

        Args:
            data (CompositeData): A composite dataset inheriting from optimize.data.CompositeData.
            model (Model): A model inheriting from optimize.models.Model.
        """
        self.data = data
        self.model = model

    def compute_score(self, pars):
        """Computes the score from a given set of parameters. This method must be implemented for each score function.

        Args:
            pars (Parameters): The parameters to use.

        Raises:
            NotImplementedError: Must implement this method.
        """
        raise NotImplementedError("Must implement a compute_score method.")
    
    def set_pars(self, pars):
        """Propogates calls to set_pars for the initial parameters, p0.

        Args:
            pars (p0): The parameters to set.
        """
        self.model.set_pars(pars)
        
        
class MinScoreFunction(ScoreFunction):
    pass

class MaxScoreFunction(ScoreFunction):
    pass
    
    
        
class MSE(MinScoreFunction):
    """A class for the standard mean squared error (MSE) loss and a namespace for commonly used routines. The loss function used here is just the RMS.
    """
    
    def compute_score(self, pars):
        """Computes the unweighted mean squared error loss.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The RMS.
        """
        model_arr = self.model.build(pars)
        data_arr = self.data.y
        rms = self.compute_rms(data_arr, model_arr)
        return rms
    
    @staticmethod
    def compute_rms(data_arr, model_arr):
        """Computes the RMS (Root mean squared) loss.

        Args_data 
            data_arr (np.ndarray): The data array.
            model_arr (np.ndarray): The model array.

        Returns:
            float: The RMS.
        """
        return np.sqrt(np.nansum((data_arr - model_arr)**2) / data_arr.size)
    
    @staticmethod
    def compute_chi2(residuals, errors):
        """Computes the (non-reduced) chi2 statistic (weighted MSE).

        Args:
            residuals (np.ndarray): The residuals = data - model.
            errors (np.ndarray): The effective errorbars.

        Returns:
            float: The chi-squared statistic.
        """
        return np.nansum((residuals / errors)**2)
    
    @staticmethod
    def compute_redchi2(residuals, errors, n_dof=None):
        """Computes the reduced chi2 statistic (weighted MSE).

        Args:
            residuals (np.ndarray): The residuals = data - model
            errors (np.ndarray): The effective errorbars (intrinsic and any white noise).
            n_dof (int): The degrees of freedom, defaults to len(res) - 1.

        Returns:
            float: The reduced chi-squared statistic.
        """
        if n_dof is None:
            n_dof = len(residuals) - 1
        chi2 = np.nansum((residuals / errors)**2)
        redchi2 = chi2 / n_dof
        return redchi2

from optimize.bayes import Likelihood, Posterior