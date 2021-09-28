# Maths
import numpy as np

# optimize deps
import optimize.maths as optmath


####################
#### BASE TYPES ####
####################

class ObjectiveFunction:
    """An base class for a general objective function. Not useful to instantiate on its own.
    """
    
    def __init__(self, model):
        self.model = model

    def compute_obj(self, pars):
        """Computes the score from a given set of parameters. This method must be implemented for each score function.

        Args:
            pars (Parameters): The parameters to use.

        Raises:
            NotImplementedError: Must implement this method.
        """
        raise NotImplementedError(f"Must implement a compute_obj method for class {self.__class__.__name__}.")
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0):
        self.p0 = p0
        self.model.initialize(self.p0)
    
    ###############
    #### MISC. ####
    ###############

    def __repr__(self):
        return f"Objective function: {self.__class__.__name__}"

class MinObjectiveFunction(ObjectiveFunction):
    def __repr__(self):
        return "Minimum Objective function"

class MaxObjectiveFunction(ObjectiveFunction):
    def __repr__(self):
        return "Max Objective function"


#################################
#### MEAN SQUARE ERROR (RMS) ####
#################################

class MSE(MinObjectiveFunction):
    """A class for the standard mean squared error (MSE=RMS) loss.
    """
    
    #####################
    #### COMPUTE OBJ ####
    #####################
    
    def compute_obj(self, pars):
        """Computes the unweighted mean squared error loss.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The RMS.
        """
        residuals = self.model.compute_residuals(pars)
        rms = self.compute_rms(residuals)
        return rms
    
    @staticmethod
    def compute_rms(residuals):
        """Computes the RMS (Root mean squared) loss. This method does not account for 

        Args:
            data_arr (np.ndarray): The data array.
            model_arr (np.ndarray): The model array.

        Returns:
            (float): The RMS.
        """
        return optmath.compute_rms(residuals)
    
    def __repr__(self):
        return "Objective: Mean Squared Error"

class Chi2(MinObjectiveFunction):
    """A class for a simple reduced chi square loss.
    """
    
    def compute_obj(self, pars):
        """Computes the reduced chi2 statistic.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The reduced chi2.
        """
        residuals = self.model.compute_residuals(pars)
        errors = self.model.compute_data_errors(pars)
        n_dof = len(residuals) - pars.num_varied
        redchi2 = self.compute_redchi2(residuals, errors, n_dof=n_dof)
        return redchi2
    
    @staticmethod
    def compute_chi2(residuals, errors):
        """Computes the (non-reduced) chi2 statistic (weighted MSE).

        Args:
            residuals (np.ndarray): The residuals array.
            errors (np.ndarray): The effective errorbars.

        Returns:
            float: The chi-squared statistic.
        """
        return optmath.compute_chi2(residuals, errors)
    
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
        return optmath.compute_redchi2(residuals, errors, n_dof)

    def __repr__(self):
        return "Objective: Chi 2"


##############################
#### BAYESIAN OOBJECTIVES ####
##############################

from .bayesobj import *