# Contains the custom Nelder-Mead algorithm
import numpy as np
import copy
import optimize.knowledge
import inspect
import scipy.optimize
from optimize.optimizers import Minimizer
import matplotlib.pyplot as plt


class SciPyMinimizer(Minimizer):
    """A class that interfaces to scipy.optimize.minimize.
    """
        
    def compute_score(self, pars):
        """Computes the score.

        Args:
            pars (np.ndarray): The parameters to use, as a numpy array to interface with scipy.
            
        Returns:
            float: The score.
        """
        self.test_pars_vec[self.p0_vary_inds] = pars
        self.test_pars.setv(value=self.test_pars_vec)
        return self.scorer.compute_score(self.test_pars)
    
    def optimize(self, **kwargs):
        """Calls the scipy.optimize.minimize routine.

        Returns:
            dict: The optimization result.
        """
        p0 = self.scorer.model.p0
        p0_dict = p0.unpack()
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        p0_vals_vary = p0_dict["value"][self.p0_vary_inds]
        lower_bounds, upper_bounds = p0.get_hard_bounds_vary()
        p0_bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(upper_bounds))]
        self.test_pars = copy.deepcopy(p0)
        self.test_pars_vec = self.test_pars.unpack(keys="value")["value"]
        res = scipy.optimize.minimize(self.compute_score, p0_vals_vary, bounds=p0_bounds, options=self.options, **kwargs)
        opt_result = {}
        opt_result["pbest"] = copy.deepcopy(p0)
        par_vec = np.copy(self.test_pars_vec)
        par_vec[self.p0_vary_inds] = res.x
        opt_result["pbest"].setv(value=par_vec)
        opt_result.update(inspect.getmembers(res, lambda a:not(inspect.isroutine(a))))
        opt_result["fbest"] = opt_result["fun"]
        del opt_result["x"], opt_result["fun"]
        return opt_result