import optimize.knowledge as optknowledge
import optimize.scores as optscores
import matplotlib.pyplot as plt

class Optimizer:
    """An base optimizer class.
    
    Attributes:
        scorer (ScoreFunction, optional): . Defaults to MSEScore.
        data (Data, optional): The data.
        options (dict): The options dictionary, with keys specific to each optimizer.
    """
    
    def __init__(self, scorer=None, options=None):
        """Construct for the base optimization class.

        Args:
            scorer (ScoreFunction, optional): . Defaults to MSEScore.
            p0 (Parameters, optional): [description]. Defaults to None.
            options (dict, optional): [description]. Defaults to None.
        """
        
        # Store scorer
        self.scorer = scorer
        
        # Store the current options dictionary and resolve
        self.options = options
        self.resolve_options()
    
    def compute_score(self, pars):
        """A wrapper to computes the score from self.scorer. 
        """
        return self.scorer.compute_score(pars, *self.scorer.args_to_pass, **self.scorer.kwargs_to_pass)
    
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
        self.scorer.set_pars(pars)
        
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