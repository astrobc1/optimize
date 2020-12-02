import numpy as np
import optimize.kernels as optnoisekernels

class Model:
    
    def __init__(self):
        pass

class SimpleModel(Model):
    """A base class for a model to optimize.
    """
    
    def __init__(self, builder, args_to_pass=None, kwargs_to_pass=None):
        """Constructs a base model for optimization.

        Args:
            builder (callable): Defines the model to use, must be callable via self.builder(*self.args_to_pass, **kwargs_to_pass)
            args_to_pass (tuple, optional): The arguments to pass to the build method. Defaults to ().
            kwargs_to_pass (dict, optional): The keyword arguments to pass to the build method. Defaults to {}.
        """
        self.builder = builder
        self.args_to_pass = () if args_to_pass is None else args_to_pass
        self.kwargs_to_pass = {} if kwargs_to_pass is None else kwargs_to_pass
        
    def build(self, pars, *args, **kwargs):
        """Builds the model.

        Args:
            pars (Parameters): The parameters to use to build the model.

        Returns:
            np.ndarray: The constructed model.
        """
        _model = self.builder(pars, *self.args_to_pass, **self.kwargs_to_pass)
        return _model
    
class BayesianModel(Model):
    """A general class for a Bayesian model to optimize.
    
    Attributes:
        kernel (NoiseKernel): The noise kernel to use.
    """
    
    def __init__(self, builder, kernel=None, args_to_pass=None, kwargs_to_pass=None):
        """Constructs a Bayesian model for optimization.

        Args:
            builder (callable): Defines the model to use, must be callable via self.builder(*self.args_to_pass, **kwargs_to_pass)
            args_to_pass (tuple, optional): The arguments to pass to the build method. Defaults to an empty tuple ().
            kwargs_to_pass (dict, optional): The keyword arguments to pass to the build method. Defaults to an empty dict {}.
            kernel (NoiseKernel, optional): The noise kernel to use. Defaults to WhiteNoiseKernel.
        """
        super().__init__(builder, args_to_pass=args_to_pass, kwargs_to_pass=kwargs_to_pass)
        if kernel is None:
            self.kernel = optnoisekernels.WhiteNoiseKernel()
        else:
            self.kernel = kernel
    
    