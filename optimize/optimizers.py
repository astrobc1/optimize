import optimize.knowledge as optknowledge
import optimize.score as optscore


class AbstractOptimizer:
    
    def __init__(self, scorer=None, p0=None, options=None):
        
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
        
        
class AbstractMinimizer(AbstractOptimizer):
    
    def __init__(self, scorer=None, p0=None, options=None):
        super().__init__(scorer=scorer, p0=p0, options=options)


class AbstractSampler(AbstractOptimizer):
    pass
    
class AffineInvSampler(AbstractSampler):
    pass
    
class MultiNestSampler(AbstractSampler):
    pass

# Import into namespace
from .neldermead import *
from .scipy_optimizers import *