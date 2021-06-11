# Contains the custom Nelder-Mead algorithm
import numpy as np
import copy
import optimize.knowledge
import inspect
import scipy.optimize

from optimize.objectives import MaxObjectiveFunction
from optimize.optimizers import Minimizer


class SciPyMinimizer(Minimizer):
    """A class that interfaces to scipy.optimize.minimize.
    """
    
    def __init__(self, method="Nelder-Mead", options=None):
        super().__init__()
        self.method = method
        self.options = {} if options is None else options
    
    ##################
    #### OPTIMIZE ####
    ##################
    
    def optimize(self):
        """Calls the scipy.optimize.minimize routine.

        Returns:
            dict: The optimization result.
        """
        
        p0 = self.obj.p0
        p0_varied = p0.get_varied()
        self.p0_numpy_varied = p0_varied.unpack()
        p0_dict = p0.unpack()
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        p0_vals_vary = p0_dict["value"][self.p0_vary_inds]
        self.test_pars = copy.deepcopy(p0)
        self.test_pars_vec = self.test_pars.unpack(keys="value")["value"]
        res = scipy.optimize.minimize(self.compute_obj, p0_vals_vary, options=self.options)
        opt_result = {}
        opt_result["pbest"] = copy.deepcopy(p0)
        opt_result["pbest"].set_vec(res.x, "value", varied=True)
        opt_result.update(inspect.getmembers(res, lambda a:not(inspect.isroutine(a))))
        opt_result["fbest"] = opt_result["fun"]
        opt_result["fcalls"] = opt_result["nfev"]
        del opt_result["x"], opt_result["fun"], opt_result["nfev"]
        return opt_result
    

    #####################
    #### COMPUTE OBJ ####
    #####################

    def compute_obj(self, pars):
        """Computes the objective.

        Args:
            pars (np.ndarray): The parameters to use, as a numpy array to interface with scipy.
            
        Returns:
            float: The objective.
        """
        
        self.test_pars.set_vec(pars, "value", varied=True)
        
        f = self.obj.compute_obj(self.test_pars)
        if isinstance(self.obj, MaxObjectiveFunction):
            f *= -1
        return f