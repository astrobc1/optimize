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
        p0 (Parameters): The initial parameters to use.
        scores (MixedScores): The score functions.
        optimizer (Optimizer, optional): The optimizer to use.
        sampler (Sampler, optional): The sampler to use for an MCMC analysis.
    """

    def __init__(self, data=None, p0=None, optimizer=None, sampler=None, scorer=None):
        """A base class for optimization problems.
    
        Args:
            p0 (Parameters, optional): The initial parameters to use. Can be set later.
            optimizer (Optimizer, optional): The optimizer to use. Can be set later.
            sampler (Sampler, optional): The sampler to use for an MCMC analysis. Can be set later.
            scorer (Scorer, optional): The score function to use. Can be set later.
        """
        
        # Store the data, model, and starting parameters
        self.data = data
        self.p0 = p0
        self.optimizer = optimizer
        self.sampler = sampler
        self.scorer = scorer
        
    def optimize(self, *args, **kwargs):
        """Generic optimize method, calls self.optimizer.optimize().
        
        Args:
            args: Any arguments to pass to optimize()
            kwargs: Any keyword arguments to pass to optimize()

        Returns:
            dict: The optimization result.
        """
        return self.optimizer.optimize(*args, **kwargs)
    
    def sample(self, *args, **kwargs):
        """Generic sample method, calls self.sampler.sample().
        
        Args:
            args: Any arguments to pass to sample()
            kwargs: Any keyword arguments to pass to sample()

        Returns:
            dict: The sampler result.
        """
        return self.sampler.sample(*args, **kwargs)
    
    def print_summary(self, opt_result):
        """A nice generic print method for the problem.

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
        """Setter method for the parameters.

        Args:
            pars (Parameters): The parameters to set.
        """
        
        # Set self
        self.p0 = pars
        
        # Set in remaining components
        if self.optimizer is not None:
            self.optimizer.set_pars(pars)
        if self.scorer is not None:
           self.scorer.set_pars(pars)
        if self.sampler is not None:
            self.sampler.set_pars(pars)
            
    def set_optimizer(self, optimizer):
        """Setter method for the optimizer.

        Args:
            optimizer (Optimizer): The optimizer to set.
        """
        self.optimizer = optimizer
        
    def set_sampler(self, sampler):
        """Setter method for the sampler.

        Args:
            sampler (Sampler): The sampler to set.
        """
        self.sampler = sampler
        
    def corner_plot(self, sampler_result=None, **kwargs):
        """Calls the corner plot method in the sampler class.

        Args:
            sampler_result (dict, optional): The sampler result.

        Returns:
            Matplotlib.Figure: A matplotlib figure containing the corner plot.
        """
        return self.sampler.corner_plot(*args, sampler_result=sampler_result, **kwargs)