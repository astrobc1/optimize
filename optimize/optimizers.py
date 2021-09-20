import optimize.parameters as optpars

####################
#### BASE TYPES ####
####################

class Optimizer:
    """An base optimizer class.
    """
    
    def initialize(self, obj):
        self.obj = obj
    
    #####################
    #### COMPUTE OBJ ####
    #####################
    
    def compute_obj(self, pars):
        """A wrapper to computes the objective function. This method may further take in any number of args or kwargs, unlike the obj.compute_obj method.
        """
        return self.obj.compute_obj(pars)
    
    ##################
    #### OPTIMIZE ####
    ##################
    
    def optimize(self):
        raise NotImplementedError("Need to implement an optimize method")
        
    ###############
    #### MISC. ####
    ###############
        
    def __repr__(self):
        return "Optimizer"
        
class Minimizer(Optimizer):
    """Trait.
    """
    def __repr__(self):
        return "Minimizer"

    def penalize(self, pars, f):
        """Penalize the objective function for bounded parameters.
        """
        if type(pars) is optpars.BoundedParameters:
            f += pars.num_out_of_bounds * self.penalty
        return f

class Maximizer(Optimizer):
    """Trait.
    """
    def __repr__(self):
        return "Maximizer"

    def penalize(self, pars, f):
        """Penalize the objective function for bounded parameters.
        """
        if type(pars) is optpars.BoundedParameters:
            f += pars.num_out_of_bounds * self.penalty
        return f

class Sampler(Optimizer):
    """Base class for MCMC samplers.
    """
    def __repr__(self):
        return "Sampler"


# Import into namespace
from .neldermead import IterativeNelderMead
from .scipy_optimizers import SciPyMinimizer
from .samplers import emceeSampler, ZeusSampler