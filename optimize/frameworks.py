import optimize.knowledge as optknow
import optimize.models as optmodels
import optimize.optimizers as optimizers
import optimize.score as optscores
import optimize.data as optdatasets
import optimize.frameworks as optframeworks

import numpy as np

class OptProblem:
    """A base class for optimization problems.
    
    Attributes:
        data (Data): A dataset inheriting from optimize.data.Data.
        model (Model): A model inheriting from optimize.models.Model.
        p0 (Parameters, optional): The initial parameters to use. Defaults to None.
        optimizer (Optimizer, optional): The optimizer to use. Defaults to NelderMead (not SciPy).
        scorer (ScoreFunction, optional): The score function to use. Defaults to MLE/reduced chi-squared for data with errorbars, and a simple MSE loss function for all other cases.
    """

    def __init__(self, data=None, model=None, p0=None, optimizer=None):
        """A base class for optimization problems.
    
        Args:
            data (Data): A dataset inheriting from optimize.data.Data.
            model (Model): A model inheriting from optimize.models.Model.
            p0 (Parameters, optional): The initial parameters to use. Defaults to None.
            optimizer (Optimizer, optional): The optimizer to use. Defaults to NelderMead (not SciPy).
            scorer (ScoreFunction, optional): The score function to use. Defaults to MLE/reduced chi-squared for data with errorbars, and a simple MSE loss function for all other cases.
        """
        
        # Store the data, model, and starting parameters
        self.data = data
        self.model = model
        self.p0 = p0
            
        # Store the Optimizer
        self.optimizer = optimizer
            
    def optimize(self):
        pass

class OptProblem(OptProblem):
    
    def optimize(self):
        return self.optimizer.optimize()

class BayesianProblem(OptProblem):
    
    def optimize(self):
        return self.optimizer.optimize()
    
    def residuals_after_kernel(self, pars):
        
        # Residuals before kernel
        _res = residuals_beforekernel(pars)
        mu = self.model.kernel.predict(pars)
    
    def residuals_before_kernel(self, pars):
        _model = self.build(self, pars)
        _res = self.data.y - _model
        return _res