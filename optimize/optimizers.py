import optimize.knowledge as optknowledge
import optimize.score as optscore


class Optimizer:
    """An base optimizer class.
    
    Attributes:
        scorer (ScoreFunction, optional): . Defaults to MSEScore.
        p0 (Parameters, optional): [description]. Defaults to None.
        options (dict, optional): [description]. Defaults to None.
    """
    
    def __init__(self, scorer=None, p0=None, options=None):
        """Construct for the base optimization class.

        Args:
            scorer (ScoreFunction, optional): . Defaults to MSEScore.
            p0 (Parameters, optional): [description]. Defaults to None.
            options (dict, optional): [description]. Defaults to None.
        """
        
        # Store scorer
        self.scorer = scorer
        
        # Store init params
        self.p0 = p0
        
        # Store the current options dictionary and resolve
        self.options = options
        self.resolve_options()
    
    def compute_score(self, *args, **kwargs):
        return self.scorer.compute_score()
    
    def resolve_options(self):
        pass
    
    def optimize(self, *args, **kwargs):
        return NotImplementedError("Need to implement an optimize method")
    
    def resolve_option(self, key, default_value):
        if key not in self.options:
            self.options[key] = default_value
        
        
class Minimizer(Optimizer):
    
    def __init__(self, scorer=None, p0=None, options=None):
        super().__init__(scorer=scorer, p0=p0, options=options)


class Sampler(Optimizer):
    pass
    
class AffineInvSampler(Sampler):
    pass
    
class MultiNestSampler(Sampler):
    pass

# Import into namespace
from .neldermead import *
from .scipy_optimizers import *