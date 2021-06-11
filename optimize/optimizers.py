####################
#### BASE TYPES ####
####################

class Optimizer:
    """An base optimizer class.
    
    Attributes:
        obj (ObjectiveFunction, optional): The objective function object.
        options (dict): The options dictionary, with keys specific to each optimizer.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, *args, **kwargs):
        """Construct for the base optimization class.
        """
        pass
    
    def initialize(self, obj):
        self.obj = obj
    
    #####################
    #### COMPUTE OBJ ####
    #####################
    
    def compute_obj(self, pars):
        """A wrapper to computes the objective function. This method may further take in any number of args or kwargs, unlike the compute_obj method.
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
    """Right now, just a node in the type tree that offers no additional functionality.
    """
    def __repr__(self):
        return "Minimizer"

class Maximizer(Optimizer):
    """Trait.
    """
    def __repr__(self):
        return "Maximizer"

class Sampler(Optimizer):
    """Base class for mcmc samplers.
    """
    def __repr__(self):
        return "Sampler"


# Import into namespace
from .neldermead import *
from .scipy_optimizers import *
from .samplers import *