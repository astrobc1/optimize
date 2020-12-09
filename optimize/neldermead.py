# Contains the custom Nelder-Mead algorithm
import numpy as np
import sys
eps = sys.float_info.epsilon # For Amoeba xtol and tfol
import time
import pdb
import copy
import numba
from numba import jit, njit, prange

import optimize.knowledge as optknow
import optimize.scores as optscores
import optimize.optimizers as optimizers
import matplotlib.pyplot as plt

class NelderMead(optimizers.Minimizer):
    """A class to interact with the Nelder Mead optimizer.
    
    """
        
    def __init__(self, scorer=None, options=None):
        """Initiate a Nelder Mead solver.
        
        Args:
        """
        
        super().__init__(scorer=scorer, options=options)
        
        # Init simplex
        self.init_params()
        
        # test_pars is constantly updated and passed to the target function wrapper
        self.test_pars = copy.deepcopy(self.scorer.model.p0)
        
        # Copy the original parameters to the current best
        self.pmin = copy.deepcopy(self.scorer.model.p0)
        
        # f calls
        self.fcalls = 0
        
        # The current fmin = inf
        self.fmin = np.inf
            
    def init_params(self):
        """Initialize the parameters

        Args:
        """
        
        # The number of parameters
        self.n_pars = len(self.scorer.model.p0)
        self.n_pars_vary = self.scorer.model.p0.num_varied()

        # Remap pointers
        self.scorer.model.p0_numpy = self.scorer.model.p0.unpack()
        self.scorer.model.p0_numpy_vary = self.scorer.model.p0.unpack(vary_only=True)
        self.scorer.model.p0_vary_inds = np.where(self.scorer.model.p0_numpy['vary'])[0]
        
        # Initialize a simplex
        self.current_full_simplex = np.zeros(shape=(self.n_pars_vary, self.n_pars_vary + 1), dtype=float)
        
        # Fill each column with the initial parameters
        self.current_full_simplex[:, :] = np.tile(self.scorer.model.p0_numpy_vary['value'].reshape(self.n_pars_vary, 1), (1, self.n_pars_vary + 1))
        
        # For each column, offset a uniqe parameter according to p=1.5*p
        self.current_full_simplex[:, :-1] += np.diag(0.5 * self.scorer.model.p0_numpy_vary['value'])


    def resolve_options(self):
        
        # Define the dictionary if not properly defined
        if type(self.options) is not dict:
            self.options = {}
        
        # Resolve the usual Hyperparams for NM
        self.resolve_option('alpha', 1.0)
        self.resolve_option('gamma', 2.0)
        self.resolve_option('sigma', 0.5)
        self.resolve_option('delta', 0.5)
            
        # Resolve the number of fevals
        self.resolve_option('max_f_evals', int(self.scorer.model.p0.num_varied() * 500))
        
        # Resolve number of times solver has effecively converged to declare true convergence
        self.resolve_option('no_improve_break', 3)
        
        # Number of ameoba iterations
        self.resolve_option('n_iterations', self.scorer.model.p0.num_varied())
        
        # Resolve xtol and ftol
        self.resolve_option('xtol', 1E-6)
        self.resolve_option('ftol', 1E-6)
        
        # Subspaces
        if 'subspaces' not in self.options:
            self.subspaces = []
            pars_varied = self.scorer.model.p0.get_varied()
            for i in range(len(pars_varied) - 1):
                self.subspaces.append([pars_varied[i].name, pars_varied[i + 1].name])
            self.subspaces.append([pars_varied[-1].name, pars_varied[0].name])
            
        else:
            self.subspaces = self.options["subspaces"]
            
        self.subspace_inds = []
        self.subspace_inds_vary = []
        for s in self.subspaces:
            self.subspace_inds.append([])
            self.subspace_inds_vary.append([])
            for pname in s:
                self.subspace_inds[-1].append(self.scorer.model.p0.index_from_par(pname))
                self.subspace_inds_vary[-1].append(self.scorer.model.p0.index_from_par(pname, rel_vary=True))

    def init_space(self, subspace_index=None):
        
        if subspace_index is not None:
            n = len(self.subspaces[subspace_index])
            inds = [self.scorer.model.p0.index_from_par(pname) for pname in self.subspaces[subspace_index]]
            self.current_simplex = np.zeros((n, n+1))
            pbest = self.pmin.unpack(keys='value')['value'][inds]
            pinit = self.scorer.model.p0_numpy['value'][inds]
            self.current_simplex[:, 0] = np.copy(pbest)
            self.current_simplex[:, 1] = np.copy(pinit)
            for i in range(2, n + 1):
                self.current_simplex[:, i] = np.copy(pbest)
                j = i - 2
                self.current_simplex[j, i] = np.copy(pinit[j])
        else:
            self.current_simplex = np.copy(self.current_full_simplex)
            
        self.test_pars = copy.deepcopy(self.pmin)

    def optimize_space(self, subspace_index=None):
        
        # Generate a simplex for this subspace
        self.init_space(subspace_index=subspace_index)
        
        # Alias the simplex
        simplex = self.current_simplex
        
        # Alias the hyperparams
        alpha, gamma, sigma, delta = self.options["alpha"], self.options["gamma"], self.options["sigma"], self.options["delta"]
        
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
            fvals[i] = self.compute_score(simplex[:, i], subspace_index=subspace_index)

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
            if self.fcalls >= self.options["max_f_evals"]:
                break
                
            # Break if f tolerance has been met
            if self.compute_ftol(fmin, fnp1) > self.options["ftol"]:
                n_converged = 0
            else:
                n_converged += 1
            if n_converged >= self.options["no_improve_break"]:
                break

            # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
            # We want to iteratively replace the worst point with a better point.
            
            # The "average" vector, ignoring the worst point
            # We first anchor points off this average Vector
            xbar[:] = np.average(simplex[:, :-1], axis=1)
            
            # The reflection point
            xr[:] = xbar + alpha * (xbar - xnp1)
            
            # Update the current testing parameter with xr
            fr = self.compute_score(xr, subspace_index=subspace_index)

            if fr < f1:
                xe[:] = xbar + gamma * (xbar - xnp1)
                fe = self.compute_score(xe, subspace_index=subspace_index)
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
                    fc = self.compute_score(xc, subspace_index=subspace_index)
                    if fc <= fr:
                        simplex[:, -1] = np.copy(xc)
                        fvals[-1] = fc
                    else:
                        shrink = True
                else:
                    xcc[:] = xbar + sigma * (xnp1 - xbar)
                    fcc = self.compute_score(xcc, subspace_index=subspace_index)
                    if fcc < fvals[-1]:
                        simplex[:, -1] = np.copy(xcc)
                        fvals[-1] = fcc
                    else:
                        shrink = True
            if shrink:
                for j in range(1, nxp1):
                    simplex[:, j] = x1 + delta * (simplex[:, j] - x1)
                    fvals[j] = self.compute_score(simplex[:, j], subspace_index=subspace_index)

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
            for i, p in enumerate(self.scorer.model.p0_numpy_vary['name']):
                self.pmin[p].setv(value=pmin[i])
        else:
            for i, p in enumerate(self.subspaces[subspace_index]):
                self.pmin[p].setv(value=pmin[i])
        
        # Update the current function minimum
        self.fmin = fmin
        
        
    def optimize(self):
        
        for iteration in range(self.options["n_iterations"]):
            
            dx = self.compute_xtol(self.current_full_simplex)
            if dx < self.options["xtol"]:
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
        out['status'] = "converged"
        out['fbest'] = self.fmin
        out['fcalls'] = self.fcalls
            
        # Recreate new parameter obejcts
        out['pbest'] = self.pmin
        out['fbest'] = self.fmin
        out['fcalls'] = self.fcalls

        return out
    
    def compute_uncertainties(self, sim):
        
        # fit quadratic coefficients
        sim = sim.T
        n = len(sim) - 1
        fsim = np.zeros(n+1)
        for i in range(n+1):
            fsim[i] = self.compute_score(sim[i])

        ymin = fsim[0]

        sim = np.copy(sim)
        fsim = np.copy(fsim)

        centroid = np.mean(sim, axis=0)
        fcentroid = self.compute_score(centroid)

        # enlarge distance of simplex vertices from centroid until all have at
        # least an absolute function value distance of 0.1
        for i in range(n + 1):
            while np.abs(fsim[i] - fcentroid) < 0.01:
                sim[i] += sim[i] - centroid
                fsim[i] = self.compute_score(sim[i])

        # the vertices and the midpoints x_ij
        x = 0.5 * (
            sim[np.mgrid[0:n + 1, 0:n + 1]][1] +
            sim[np.mgrid[0:n + 1, 0:n + 1]][0]
        )

        y = np.nan * np.ones(shape=(n + 1, n + 1))
        for i in range(n + 1):
            y[i, i] = fsim[i]
            for j in range(i + 1, n + 1):
                y[i, j] = y[j, i] = self.compute_score(x[i, j])

        y0i = y[np.mgrid[0:n + 1, 0:n + 1]][0][1:, 1:, 0]

        y0j = y[np.mgrid[0:n + 1, 0:n + 1]][0][0, 1:, 1:]

        b = 2 * (y[1:, 1:] + y[0, 0] - y0i - y0j)

        q = (sim - sim[0])[1:].T

        varco = ymin * np.dot(q, np.dot(np.linalg.inv(b), q.T))
        errors = np.sqrt(np.diag(varco))
        
        return errors
    
        
    @staticmethod
    def compute_xtol(simplex):
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
    def compute_ftol(a, b):
        return np.abs(a - b)
            
    def compute_score(self, x, subspace_index=None):
        
        if subspace_index is None:
            for i, p in enumerate(self.scorer.model.p0_numpy_vary['name']):
                self.test_pars[p].setv(value=x[i])
        else:
            for i, p in enumerate(self.subspaces[subspace_index]):
                self.test_pars[p].setv(value=x[i])
        
        # Call the target function
        f = self.scorer.compute_score(self.test_pars)
        
        # Update fcalls
        self.fcalls += 1
        
        # Return the score
        if np.isinf(f):
            f = 1E6
            
        return f