# Contains the custom Nelder-Mead algorithm
import numpy as np
import copy
import optimize.knowledge
import inspect
import scipy.optimize
from optimize.optimizers import AbstractMinimizer


class SciPyMinimizer(AbstractMinimizer):
    
    def __init__(self, scorer, p0, scipy_kwargs=None, options=None):
        super().__init__(scorer, p0, options=options)
        if scipy_kwargs is None:
            self.scipy_kwargs = {}
        else:
            self.scipy_kwargs = scipy_kwargs
        
    def compute_score(self, pars):
        self.test_pars_vec[self.p0_vary_inds] = pars
        self.test_pars.setv(value=self.test_pars_vec)
        return self.scorer.compute_score(self.test_pars)
    
    def optimize(self):
        self.p0_numpy = self.p0.unpack()
        self.p0_vary_inds = np.where(self.p0_numpy["vary"])[0]
        p0_vals_vary = self.p0_numpy["value"][self.p0_vary_inds]
        lower_bounds, upper_bounds = self.p0.get_hard_bounds_vary()
        p0_bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(upper_bounds))]
        self.test_pars = copy.deepcopy(self.p0)
        self.test_pars_vec = self.test_pars.unpack(keys="value")["value"]
        res = scipy.optimize.minimize(self.compute_score, p0_vals_vary, bounds=p0_bounds, options=self.options, **self.scipy_kwargs)
        opt_result = {}
        opt_result["pbest"] = copy.deepcopy(self.p0)
        par_vec = np.copy(self.test_pars_vec)
        par_vec[self.p0_vary_inds] = res.x
        opt_result["pbest"].setv(value=par_vec)
        opt_result.update(inspect.getmembers(res, lambda a:not(inspect.isroutine(a))))
        opt_result["fbest"] = opt_result["fun"]
        del opt_result["x"], opt_result["fun"]
        return opt_result