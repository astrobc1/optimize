import optimize.knowledge as optknowledge
import optimize.objectives as optobj
import matplotlib.pyplot as plt

class Optimizer:
    """An base optimizer class.
    
    Attributes:
        obj (ObjectiveFunction, optional): . Defaults to MSE.
        data (Data, optional): The data.
        options (dict): The options dictionary, with keys specific to each optimizer.
    """
    
    def __init__(self, obj=None, options=None):
        """Construct for the base optimization class.

        Args:
            obj (ObjectiveFunction, optional): . Defaults to MSE.
            p0 (Parameters, optional): [description]. Defaults to None.
            options (dict, optional): [description]. Defaults to None.
        """
        
        # Store the objective function
        self.obj = obj
        
        # Store the current options dictionary and resolve
        self.options = options
        self.resolve_options()
    
    def compute_obj(self, pars):
        """A wrapper to computes the objective function.
        """
        return self.obj.compute_obj(pars, *self.obj.args_to_pass, **self.obj.kwargs_to_pass)
    
    def resolve_options(self):
        pass
    
    def optimize(self, *args, **kwargs):
        raise NotImplementedError("Need to implement an optimize method")
    
    def resolve_option(self, key, default_value):
        """Given an option key and default value, this will set the corresponding item in the options dictionary if not already set.

        Args:
            key (str): The key to set or check.
            default_value (object): The default value to use if not set by the user.
        """
        if key not in self.options:
            self.options[key] = default_value
            
    def set_pars(self, pars):
        self.obj.set_pars(pars)
        
class Minimizer(Optimizer):
    """Right now, just a node in the type tree that offers no additional functionality.
    """
    pass

class Maximizer(Optimizer):
    """Right now, just a node in the type tree that offers no additional functionality.
    """
    pass


class Sampler(Optimizer):
    """Right now, just a node in the type tree that offers no additional functionality.
    """
    pass
        
    

# Import into namespace
from .neldermead import *
from .scipy_optimizers import *
from .samplers import *