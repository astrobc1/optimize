# Maths
import numpy as np

class OptProblem:
    """A base class for optimization problems.
    
    Attributes:
        obj (ObjectiveFunction): The score functions.
        optimizer (Optimizer): The optimizer to use.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################

    def __init__(self, p0=None, obj=None, optimizer=None):
        """A base class for optimization problems.
    
        Args:
            p0 (The initial parameter, optional): The initial parameters to use.
            obj (ObjectiveFunction, optional): The objective function.
            optimizer (Optimizer, optional): The optimizer to use. May be set later.
        """
        
        # Store the objective and optimizer
        self.p0 = p0
        self.obj = obj
        self.optimizer = optimizer
        
    ##################
    #### OPTIMIZE ####
    ##################
        
    def optimize(self):
        """Forward method for optimizing. Calls self.optimizer.optimize(*args, **kwargs)
        
        Args:
            args: Any arguments to pass to optimize()
            kwargs: Any keyword arguments to pass to optimize()

        Returns:
            dict: The optimization result.
        """
        self.initialize()
        return self.optimizer.optimize()
            
    #################
    #### SETTERS ####
    #################
    
    def set_pars(self, pars):
        """Setter method for the parameters.

        Args:
            pars (Parameters): The parameters to set.
        """
        self.pars = pars
    
    def set_obj(self, obj):
        """Setter method for the objective function.

        Args:
            obj (ObjectiveFunction): The objective function to set.
        """
        self.obj = obj
            
    def set_optimizer(self, optimizer):
        """Setter method for the optimizer.

        Args:
            optimizer (Optimizer): The optimizer to set.
        """
        self.optimizer = optimizer
    
    #####################
    #### INITIALIZER ####
    #####################
    
    def initialize(self):
        self.obj.initialize(self.p0)
        self.optimizer.initialize(self.obj)
            
            
    ###############
    #### MISC. ####
    ###############
    
    def __repr__(self):
        
        # Header
        s = "Optimization Problem\n"
        
        # Print the objective function details
        s += f" Objective: {self.obj}\n"
        
        # Print the optimizer
        s += f" Optimizer: {self.optimizer}\n"
            
        # Print the current parameters.
        s += " Parameters:\n"
        s += f"  {self.p0}"
        
        return s


class BayesianProblem(OptProblem):
    """A base class for optimization problems.
    
    Attributes:
        p0 (Parameters): The initial parameters to use.
        post (Posterior): The posterior objective function.
        optimizer (Optimizer): The optimizer to use.
        sampler (Sampler): The sampler to use for an MCMC analysis.
    """

    #####################
    #### CONSTRUCTOR ####
    #####################

    def __init__(self, p0=None, post=None, optimizer=None, sampler=None):
        """A base class for optimization problems.
    
        Args:
            p0 (Parameters, optional): The initial parameters to use.
            post (Posterior, optional): The score function to use.
            optimizer (Optimizer, optional): The optimizer to use.
            sampler (Sampler, optional): The sampler to use for an MCMC analysis.
        """
        
        # Super
        super().__init__(p0=p0, obj=post, optimizer=optimizer)
        
        # Store sampler
        self.sampler = sampler
    
    ##################
    #### OPTIMIZE ####
    ##################
        
    def optimize(self):
        """Forward method for optimizing. Calls self.optimizer.optimize(*args, **kwargs)
        
        Args:
            args: Any arguments to pass to optimize()
            kwargs: Any keyword arguments to pass to optimize()

        Returns:
            dict: The optimization result.
        """
        self.initialize()
        return self.optimizer.optimize()
            
    def run_mapfit(self, *args, **kwargs):
        """Alias for optimize.

        Args:
            *args: Any args.
            **kwargs: Any keyword args.
            
        Returns:
            dict: A dictionary with the optimize results.
        """
        return self.optimize(*args, **kwargs)
            
    def run_mcmc(self, *args, **kwargs):
        """Forward method for MCMC sampling.
        
        Args:
            args: Any arguments to pass to sample()
            kwargs: Any keyword arguments to pass to sample()

        Returns:
            dict: The sampler result.
        """
        self.initialize()
        return self.sampler.run_mcmc(*args, **kwargs)
     
    #################
    #### SETTERS ####
    #################
    
    def set_post(self, post):
        """Setter method for the posterior function.

        Args:
            post (Posterior): The posterior to set.
        """
        self.obj = post
        
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
 
    #####################
    #### INITIALIZER ####
    #####################
    
    def initialize(self):
        super().initialize()
        if self.sampler is not None:
            self.sampler.initialize(self.obj)
        
    #####################
    #### CORNER PLOT ####
    #####################
        
    def corner_plot(self, mcmc_result, **kwargs):
        """Calls the corner plot method in the sampler class.

        Args:
            mcmc_result (dict, optional): The sampler result.

        Returns:
            Matplotlib.Figure: A matplotlib figure containing the corner plot.
        """
        return self.sampler.corner_plot(mcmc_result, **kwargs)
    
    ###############
    #### MISC. ####
    ###############
    
    @property
    def likes(self):
        return self.post
    
    @property
    def post(self):
        return self.obj
    
    @property
    def like0(self):
        return self.post.like0
    
    def __repr__(self):
        
        # Header
        s = "Bayesian Optimization Problem"
        
        # Loop over likes and print
        for like in self.post.values():
            s += f"  {like}\n"
        
        # Print the optimizer
        if hasattr(self, 'optimizer'):
            s += f"  {self.optimizer}\n"
            
        # Print the current parameters.
        s += "  Parameters:"
        s += f"{self.p0}"
        
        
        return s