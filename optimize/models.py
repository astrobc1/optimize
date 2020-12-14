import numpy as np
import optimize.kernels as optnoisekernels
import matplotlib.pyplot as plt

class Model:
    """Constructs a Bayesian base model for optimization. This class is useful to instantiate for simple Bayesian optimization problems.

        Attributes:
            p0 (Parameters, optional): The initial parameters to use. Defaults to None.
            data (MixedData, optional): The dataset.
            builder (callable): Defines the model to use. Any methods that construct the model will start at the build method, which 1. must be called as build(pars) and 2. by default calls builder(*args_to_pass, **kwargs_to_pass). A second option is to extend the Model class and implement one's own build method.
            args_to_pass (tuple, optional): The arguments to pass to the build method. Defaults to ().
            kwargs_to_pass (dict, optional): The keyword arguments to pass to the build method. Defaults to {}.
            kernel (NoiseKernel): The noise kernel to use.
    """
    
    def __init__(self, p0=None, data=None, builder=None, args_to_pass=None, kwargs_to_pass=None, kernel=None):
        """Constructs a base model for optimization.

        Args:
            p0 (Parameters, optional): The initial parameters to use. Defaults to None.
            data (MixedData, optional): The dataset, must be identical to kernel.data.
            builder (callable): Defines the model to use. Any methods that construct the model will start at the build method, which 1. must be called as build(pars) and 2. by default calls builder(*args_to_pass, **kwargs_to_pass). A second option is to extend the Model class and implement one's own build method.
            args_to_pass (tuple, optional): The arguments to pass to the build method. Defaults to ().
            kwargs_to_pass (dict, optional): The keyword arguments to pass to the build method. Defaults to {}.
            kernel (NoiseKernel): The noise kernel to use.
        """
        self.p0 = p0
        self.data = data
        self.builder = builder
        self.args_to_pass = () if args_to_pass is None else args_to_pass
        self.kwargs_to_pass = {} if kwargs_to_pass is None else kwargs_to_pass
        self.kernel = kernel
        self.has_gp = False
        if isinstance(self.kernel, optnoisekernels.GaussianProcess):
            self.has_gp = True
        self.data_inds = {}
        for data in self.data.values():
            self.data_inds[data.label] = self.data.get_inds(data.label)
    
    def build(self, pars):
        """Builds the model.

        Args:
            pars (Parameters): The parameters to use to build the model.

        Returns:
            np.ndarray: The constructed model.
        """
        _model = self.builder(pars, *self.args_to_pass, **self.kwargs_to_pass)
        return _model
    
    def set_pars(self, pars):
        """Simple setter method for the parameters that may be extended.

        Args:
            pars (Parameters): The new starting parameters to use.
        """
        self.p0 = pars

class MixedModel(dict):
    pass