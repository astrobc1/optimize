import numpy as np
import optimize.noise as optnoise
import matplotlib.pyplot as plt

class Model:
    """Constructs a base model for optimization. This class may be instantiated for simple optimization problems but extending it is preferred.

    Attributes:
        builder (callable, optional): Funtion that constructs the model with signature fun(*args_to_pass, **kwargs_to_pass). Models extending this class may define their build method elsewhere. Defaults to None. If None, the class must override the build method.
        args_to_pass (tuple, optional): The arguments to pass to the build method. Defaults to ().
        kwargs_to_pass (dict, optional): The keyword arguments to pass to the build method. Defaults to {}.
    """
    
    def __init__(self, builder=None, args_to_pass=None, kwargs_to_pass=None):
        """Constructs a base model for optimization.

        Args:
            builder (callable, optional): Funtion that constructs the model with signature fun(pars, *args_to_pass, **kwargs_to_pass).
            args_to_pass (tuple, optional): The arguments to pass to the build method. Defaults to ().
            kwargs_to_pass (dict, optional): The keyword arguments to pass to the build method. Defaults to {}.
            kernel (NoiseKernel, optional): The noise kernel to use, defaults to None (no noise).
        """
        self.builder = builder
        self.args_to_pass = () if args_to_pass is None else args_to_pass
        self.kwargs_to_pass = {} if kwargs_to_pass is None else kwargs_to_pass
    
    def build(self, pars):
        """Constructs the model.

        Args:
            pars (Parameters): The parameters to use to build the model.

        Returns:
            object: The constructed model, probably as a numpy array but is ultimately managed by the objective function.
        """
        _model = self.builder(pars, *self.args_to_pass, **self.kwargs_to_pass)
        return _model


class PureGP(Model):

    def build(self, pars):
        return self.data_zeros

# class PyMC3Model(Model, pm.model.Model):
    
#     def __init__(self, p0=None, data=None, kernel=None):
#         super().__init__(p0=p0, data=data, kernel=kernel)
#         self.pars_to_pmd()