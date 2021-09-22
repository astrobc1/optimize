# Contains the custom Nelder-Mead algorithm
import numpy as np
import sys
eps = sys.float_info.epsilon # For Amoeba xtol and tfol
import time
import pdb
import copy
import numba
from numba import jit, njit, prange

import optimize.parameters as optpars
import optimize.objectives as optobj
import optimize.optimizers as optimizers
import matplotlib.pyplot as plt

class IterativeNelderMead(optimizers.Minimizer):
    """A class to interact with the iterative Nelder Mead optimizer.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, sigma=0.5, delta=0.5, xtol=1E-8, ftol=1E-6, n_iterations=None, no_improve_break=3, penalty=1E6, max_f_evals=None, initial_scale_factor=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.delta = delta
        self.ftol = ftol
        self.xtol = xtol
        self.max_f_evals = max_f_evals
        self.n_iterations = n_iterations
        self.no_improve_break = no_improve_break
        self.penalty = penalty
        self.initial_scale_factor = initial_scale_factor
    
    #############################
    #### CONSTRUCTOR HELPERS ####
    #############################
    
    def init_params(self):
        """Initialize the parameters

        Args:
        """
        
        # The number of parameters
        self.n_pars = len(self.obj.p0)
        self.n_pars_vary = self.obj.p0.num_varied

        # Remap pointers
        self.p0_numpy = self.obj.p0.unpack()
        self.p0_numpy_vary = self.obj.p0.unpack(vary_only=True)
        self.p0_vary_inds = np.where(self.p0_numpy['vary'])[0]
        
        # Initialize a simplex
        self.current_full_simplex = np.zeros(shape=(self.n_pars_vary, self.n_pars_vary + 1), dtype=float)

        # Fill each column with the initial parameters
        self.current_full_simplex[:, :] = np.tile(self.p0_numpy_vary['value'].reshape(self.n_pars_vary, 1), (1, self.n_pars_vary + 1))
        
        # For each column, offset a uniqe parameter according to p=initial_scale_factor*p
        self.current_full_simplex[:, :-1] += np.diag(self.initial_scale_factor * self.p0_numpy_vary['value'])

    ####################
    #### INITIALIZE ####
    ####################

    def initialize(self, obj):
        
        # Store objective function
        self.obj = obj
        
        # Alias p0
        p0 = self.obj.p0
        
        # Resolve the number of fevals
        if self.max_f_evals is None:
            self.max_f_evals = p0.num_varied * 500
        
        # Number of ameoba iterations
        if self.n_iterations is None:
            self.n_iterations = p0.num_varied
        
    def init_subspaces(self):
        
        # Subspaces
        self.subspaces = []
        pars_varied = self.obj.p0.get_varied()
        for i in range(len(pars_varied) - 1):
            self.subspaces.append([pars_varied[i].name, pars_varied[i + 1].name])
        self.subspaces.append([pars_varied[-1].name, pars_varied[0].name])
            
        self.subspace_inds = []
        self.subspace_inds_vary = []
        for s in self.subspaces:
            self.subspace_inds.append([])
            self.subspace_inds_vary.append([])
            for pname in s:
                self.subspace_inds[-1].append(self.obj.p0.index_from_par(pname))
                self.subspace_inds_vary[-1].append(self.obj.p0.index_from_par(pname, rel_vary=True))

    def init_space(self, subspace_index=None):
        
        if subspace_index is not None:
            n = len(self.subspaces[subspace_index])
            inds = [self.obj.p0.index_from_par(pname) for pname in self.subspaces[subspace_index]]
            self.current_simplex = np.zeros((n, n+1))
            pbest = self.pmin.unpack(keys='value')['value'][inds]
            pinit = self.p0_numpy['value'][inds]
            self.current_simplex[:, 0] = np.copy(pbest)
            self.current_simplex[:, 1] = np.copy(pinit)
            for i in range(2, n + 1):
                self.current_simplex[:, i] = np.copy(pbest)
                j = i - 2
                self.current_simplex[j, i] = np.copy(pinit[j])
        else:
            self.current_simplex = np.copy(self.current_full_simplex)
            
        self.test_pars = copy.deepcopy(self.pmin)

    ##################
    #### OPTIMIZE ####
    ##################

    def optimize_space(self, subspace_index=None):
        
        # Generate a simplex for this subspace
        self.init_space(subspace_index=subspace_index)
        
        # Alias the simplex
        simplex = self.current_simplex
        
        # Alias the hyperparams
        alpha, gamma, sigma, delta = self.alpha, self.gamma, self.sigma, self.delta
        
        # Define these as they are used often
        nx, nxp1 = simplex.shape

        # Initiate storage arrays
        fvals = np.empty(nxp1, dtype=float)
        xr = np.empty(nx, dtype=float)
        xbar = np.empty(nx, dtype=float)
        xc = np.empty(nx, dtype=float)
        xe = np.empty(nx, dtype=float)
        xcc = np.empty(nx, dtype=float)
        
        # Generate the fvals for the initial simplex
        for i in range(nxp1):
            fvals[i] = self.compute_obj(simplex[:, i], subspace_index=subspace_index)

        # Sort the fvals and then simplex
        ind = np.argsort(fvals)
        simplex = simplex[:, ind]
        fvals = fvals[ind]
        fmin = fvals[0]
        
        # Best fit parameter is now the first column
        pmin = simplex[:, 0]
        
        # Keeps track of the number of times the solver thinks it has converged in a row.
        n_converged = 0
        
        # Force convergence with break
        while True:

            # Sort the vertices according from best to worst
            # Define the worst and best vertex, and f(best vertex)
            xnp1 = simplex[:, -1]
            fnp1 = fvals[-1]
            x1 = simplex[:, 0]
            f1 = fvals[0]
            xn = simplex[:, -2]
            fn = fvals[-2]
                
            # Checks whether or not to shrink if all other checks "fail"
            shrink = False

            # break after max number function calls is reached.
            if self.fcalls >= self.max_f_evals:
                break
                
            # Break if f tolerance has been met
            if self.compute_df(fmin, fnp1) > self.ftol:
                n_converged = 0
            else:
                n_converged += 1
            if n_converged >= self.no_improve_break:
                break

            # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
            # We want to iteratively replace the worst point with a better point.
            
            # The "average" vector, ignoring the worst point
            # We first anchor points off this average Vector
            xbar[:] = np.average(simplex[:, :-1], axis=1)
            
            # The reflection point
            xr[:] = xbar + alpha * (xbar - xnp1)
            
            # Update the current testing parameter with xr
            fr = self.compute_obj(xr, subspace_index=subspace_index)

            if fr < f1:
                xe[:] = xbar + gamma * (xbar - xnp1)
                fe = self.compute_obj(xe, subspace_index=subspace_index)
                if fe < fr:
                    simplex[:, -1] = np.copy(xe)
                    fvals[-1] = fe
                else:
                    simplex[:, -1] = np.copy(xr)
                    fvals[-1] = fr
            elif fr < fn:
                simplex[:, -1] = xr
                fvals[-1] = fr
            else:
                if fr < fnp1:
                    xc[:] = xbar + sigma * (xbar - xnp1)
                    fc = self.compute_obj(xc, subspace_index=subspace_index)
                    if fc <= fr:
                        simplex[:, -1] = np.copy(xc)
                        fvals[-1] = fc
                    else:
                        shrink = True
                else:
                    xcc[:] = xbar + sigma * (xnp1 - xbar)
                    fcc = self.compute_obj(xcc, subspace_index=subspace_index)
                    if fcc < fvals[-1]:
                        simplex[:, -1] = np.copy(xcc)
                        fvals[-1] = fcc
                    else:
                        shrink = True
            if shrink:
                for j in range(1, nxp1):
                    simplex[:, j] = x1 + delta * (simplex[:, j] - x1)
                    fvals[j] = self.compute_obj(simplex[:, j], subspace_index=subspace_index)

            ind = np.argsort(fvals)
            fvals = fvals[ind]
            simplex = simplex[:, ind]
            fmin = fvals[0]
            pmin = simplex[:, 0]
            
        # Update current simplex
        self.current_simplex = np.copy(simplex)
        
        # Update full simplex
        if subspace_index is not None:
            self.current_full_simplex[self.subspace_inds_vary[subspace_index], self.subspace_inds_vary[subspace_index][0]] = np.tile(pmin.reshape(pmin.size, 1), (len(self.subspace_inds_vary[subspace_index]) - 1)).flatten()
        else:
            self.current_full_simplex = np.copy(self.current_simplex)
        
        if subspace_index is None:
            self.current_full_simplex = np.copy(simplex)
            for i, p in enumerate(self.p0_numpy_vary['name']):
                self.pmin[p].value = pmin[i]
        else:
            for i, p in enumerate(self.subspaces[subspace_index]):
                self.pmin[p].value = pmin[i]
        
        # Update the current function minimum
        self.fmin = fmin
               
    def optimize(self):
        
        # Init params
        self.init_params()
        
        # Init subspaces
        self.init_subspaces()
        
        # test_pars is constantly updated and passed to the target function wrapper
        self.test_pars = copy.deepcopy(self.obj.p0)
        
        # Copy the original parameters to the current best
        self.pmin = copy.deepcopy(self.obj.p0)
        
        # f calls
        self.fcalls = 0
        
        # The current fmin = inf
        self.fmin = np.inf

        # The current status
        self.status = "failed"
        
        for iteration in range(self.n_iterations):
            
            dx = self.compute_dx(self.current_full_simplex)
            if dx < self.xtol:
                self.status = "success"
                break

            # Perform Ameoba call for all parameters
            self.optimize_space(None)
            
            # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
            if self.n_pars_vary <= 2:
                break
            
            # Perform Ameoba call for subspaces
            for subspace_index in range(len(self.subspaces)):
                self.optimize_space(subspace_index)
        
        # Output variable
        out = {}
        
        out['status'] = self.status
        out['fbest'] = self.fmin
        out['fcalls'] = self.fcalls
            
        # Recreate new parameter obejcts
        out['pbest'] = self.pmin

        return out
        
    ###############
    #### MISC. ####
    ###############
    
    @staticmethod
    def compute_dx(simplex):
        a = np.nanmin(simplex, axis=1)
        b = np.nanmax(simplex, axis=1)
        c = (np.abs(b) + np.abs(a)) / 2
        c = np.atleast_1d(c)
        ind = np.where(c < eps)[0]
        if ind.size > 0:
            c[ind] = 1
        r = np.abs(b - a) / c
        return np.nanmax(r)

    @staticmethod
    @njit(numba.types.float64(numba.types.float64, numba.types.float64))
    def compute_df(a, b):
        return np.abs(a - b)
    
    def compute_obj(self, x, subspace_index=None):
        
        if subspace_index is None:
            for i, p in enumerate(self.p0_numpy_vary['name']):
                self.test_pars[p].value = x[i]
        else:
            for i, p in enumerate(self.subspaces[subspace_index]):
                self.test_pars[p].value = x[i]
        
        # Call the target function
        f = self.obj.compute_obj(self.test_pars)
        
        # Update fcalls
        self.fcalls += 1

        # Penalize
        self.penalize(self.test_pars, f)
            
        # Return max or min of obj
        if isinstance(self.obj, optobj.MaxObjectiveFunction):
            f *= -1

        # If f is not finite, don't return inf or nan, return a large number
        if not np.isfinite(f):
            f = self.penalty
        
        return f
    
    def __repr__(self):
        return "Optimizer: Iterative Nelder Mead"