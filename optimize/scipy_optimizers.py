# Contains the custom Nelder-Mead algorithm
import numpy as np
import copy
import optimize.knowledge
import inspect
import scipy.optimize

import optimize.scores as optscores
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
        if isinstance(self.scorer, optscores.Likelihood) or isinstance(self.scorer, optscores.MixedLikelihood):
            return -1 * self.scorer.compute_score(self.test_pars)
        else:
            return self.scorer.compute_score(self.test_pars)
    
    def optimize(self, **kwargs):
        """Calls the scipy.optimize.minimize routine.

        Returns:
            dict: The optimization result.
        """
        p0 = self.scorer.p0
        p0_dict = p0.unpack()
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        p0_vals_vary = p0_dict["value"][self.p0_vary_inds]
        self.test_pars = copy.deepcopy(p0)
        self.test_pars_vec = self.test_pars.unpack(keys="value")["value"]
        res = scipy.optimize.minimize(self.compute_score, p0_vals_vary, options=self.options, **kwargs)
        opt_result = {}
        opt_result["pbest"] = copy.deepcopy(p0)
        par_vec = np.copy(self.test_pars_vec)
        par_vec[self.p0_vary_inds] = res.x
        opt_result["pbest"].setv(value=par_vec)
        opt_result.update(inspect.getmembers(res, lambda a:not(inspect.isroutine(a))))
        opt_result["fbest"] = opt_result["fun"]
        opt_result["fcalls"] = opt_result["nfev"]
        del opt_result["x"], opt_result["fun"], opt_result["nfev"]
        return opt_result