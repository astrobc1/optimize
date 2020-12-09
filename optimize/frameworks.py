import optimize.knowledge as optknow
import optimize.models as optmodels
import optimize.optimizers as optimizers
import optimize.scores as optscores
import optimize.data as optdatasets
import optimize.frameworks as optframeworks

import matplotlib.pyplot as plt

import numpy as np

class OptProblem:
    """A class for most Bayesian optimization problems.
    
    Attributes:
        data (Data): A dataset inheriting from optimize.data.Data.
        model (Model): A model inheriting from optimize.models.Model.
        p0 (Parameters): The initial parameters to use. Defaults to None.
        optimizer (Optimizer, optional): The optimizer to use. Defaults to NelderMead (not SciPy).
        sampler (Sampler, optional): The sampler to use to MCMC analysis.
    """
    
    __children__ = ['data', 'model', 'p0', 'optimizer', 'sampler']

    def __init__(self, data=None, model=None, p0=None, optimizer=None, sampler=None):
        """A base class for optimization problems.
    
        Args:
            data (Data, optional): A dataset inheriting from optimize.data.Data.
            model (Model, optional): A model inheriting from optimize.models.Model.
            p0 (Parameters, optional): The initial parameters to use. Defaults to None.
            optimizer (Optimizer, optional): The optimizer to use.
            sampler (Sampler, optional): The sampler to use to MCMC analysis.
        """
        
        # Store the data, model, and starting parameters
        self.data = data
        self.model = model
        self.p0 = p0
        self.optimizer = optimizer
        self.sampler = sampler
        
    def optimize(self):
        """Generic optimize method, calls self.optimizer.optimize().

        Returns:
            opt_result (dict): The optimization result.
        """
        return self.optimizer.optimize()
    
    def print_summary(self, opt_result):
        """A nice generic print method for the Bayesian framework.

        Args:
            opt_result (dict, optional): The optimization result to print. Defaults to None, and thus prints the initial parameters.
        """
        
        # Print the data and model
        print(self.data, flush=True)
        print(self.model, flush=True)
        
        # Print the optimizer and sampler
        if hasattr(self, 'optimizer'):
            print(self.optimizer, flush=True)
        if hasattr(self, 'sampler'):
            print(self.sampler, flush=True)
            
        # Print the best fit parameters or initial parameters.
        print("Parameters:", flush=True)
        if opt_result is not None:
            opt_result['pbest'].pretty_print()
        else:
            self.p0.pretty_print()
            
    def set_pars(self, pars):
        """Simple setter method for the parameters that may be extended.

        Args:
            pars (Parameters): The new starting parameters to use.
        """
        self.p0 = pars
        self.model.set_pars(pars)
        
    def residuals_after_kernel(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        _res = self.residuals_before_kernel(pars)
        _errors = self.sampler.scorer.compute_errorbars(pars)
        mu = self.model.kernel.realize(pars, xpred=self.data.x, xres=self.data.x, res=_res, errors=_errors, stddev=False)
        return _res - mu
    
    def residuals_before_kernel(self, pars):
        """Computes the residuals without subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        _model = self.build(self, pars)
        _res = self.data.y - _model
        return _res
        
            
        
            