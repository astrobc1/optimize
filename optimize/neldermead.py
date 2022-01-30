# Default Python
import sys
EPSILON = sys.float_info.epsilon # For Amoeba xtol and tfol
import copy

# Maths
import numpy as np

# LLVM
import numba
from numba import jit, njit, prange

# Optimize
import optimize.parameters

class IterativeNelderMead:
    """An iterative Nelder-Mead optimizer for high-dimensional parameter spaces.
    """

    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, obj=None, alpha=1.0, gamma=2.0, sigma=0.5, delta=0.5, xtol_rel=1E-6, ftol_rel=1E-6, n_iterations=None, no_improve_break=3, apply_penalty=None, penalty=1E6, max_f_evals=None, initial_scale_factor=0.5, obj_args=None, obj_kwargs=None, maximize=False):
        """Construct the iterative Nelder-Mead optimizer.

        Args:
            obj (callable, optional): The objective to call. Defaults to None and must be set later.
            alpha (float, optional): Nelder-Mead hyper-parameter. Defaults to 1.0.
            gamma (float, optional): Nelder-Mead hyper-parameter. Defaults to 2.0.
            sigma (float, optional): Nelder-Mead hyper-parameter. Defaults to 0.5.
            delta (float, optional): Nelder-Mead hyper-parameter. Defaults to 0.5.
            xtol (float, optional): The relative x tolderance for convergence. Defaults to 1E-6.
            ftol (float, optional): The relative f tolerance for convergence. Defaults to 1E-6.
            n_iterations (int, optional): The number of iterations. Defaults to len(p0).
            no_improve_break (int, optional): The number of times in a row the solver must converge before actually breaking. Defaults to 3.
            penalty (float, optional): The penalty to add to the objective for each BoundedParameter. Defaults to 1E6.
            max_f_evals (float, optional): The maximum number of f evaluations. Defaults to 500 * len(p0).
            initial_scale_factor (float or str, optional): A scale factor to initiate the initial simplex. Defaults to 0.5.
            obj_args (tuple, optional): Additional argument
        """
        self.obj = obj
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.delta = delta
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.max_f_evals = max_f_evals
        self.n_iterations = n_iterations
        self.no_improve_break = no_improve_break
        self.apply_penalty = apply_penalty
        self.penalty = penalty
        self.initial_scale_factor = initial_scale_factor
        self.obj_args = () if obj_args is None else obj_args
        self.obj_kwargs = {} if obj_kwargs is None else obj_kwargs
        self._numpy = None
        self.maximize = maximize

    ####################
    #### INITIALIZE ####
    ####################

    def initialize(self, p0, obj=None, lower_bounds=None, upper_bounds=None, parameter_names=None, obj_args=None, obj_kwargs=None):

        # Objective
        if self.obj is None and obj is not None:
            self.obj = obj

        # Objective args
        if obj_args is not None:
            self.obj_args = obj_args
        elif self.obj_args is None and obj_args is None:
            self.obj_args = ()

        # Objective kwargs
        if obj_kwargs is not None:
            self.obj_kwargs = obj_kwargs
        elif self.obj_kwargs is None and obj_kwargs is None:
            self.obj_kwargs = {}

        # Init params
        self.initialize_parameters(p0, lower_bounds, upper_bounds, parameter_names)
        
        # Init the full simplex
        self.initialize_simplex()

        # Initialize the subspaces
        self.initialize_subspaces()

        # Copy the original parameters to the current best
        self.pmin = copy.deepcopy(self.p0)
        
        # Number of objective calls
        self.fcalls = 0
        
        # The current fmin is infinity
        self.fmin = np.inf

        # The current status
        self.status = "failed"

    def initialize_parameters(self, p0, lower_bounds=None, upper_bounds=None, parameter_names=None, vary=None):
        """Initialize the remaining solver options with the initial parameters.

        Args:
            p0 (Parameters or np.ndarray): The initial parameters.
            lower_bounds (np.ndarray, optional): The lower bounds if using numpy vectors.
            upper_bounds (np.ndarray, optional): The upper bounds if using numpy vectors.
            parameter_names (np.ndarray, optional): The parameter names if using numpy vectors.
            vary (np.ndarray, optional): An array of ones and zeros of whether or not to vary that parameter if using numpy vectors.
        """

        # Parameters
        if isinstance(p0, optimize.parameters.Parameters):
            self._numpy = False
            self.p0 = p0
            self.n_pars = len(p0)
            self.n_pars_vary = p0.num_varied
        else:
            self._numpy = True
            self.n_pars = len(p0)
            if parameter_names is None:
                parameter_names = np.array([f"param{i+1}" for i in range(self.n_pars)])
            if lower_bounds is None:
                lower_bounds = np.full(len(p0), -np.inf)
            if upper_bounds is None:
                upper_bounds = np.full(len(p0), np.inf)
            if vary is None:
                vary = np.full(len(p0), True)
            self.p0 = optimize.parameters.BoundedParameters()
            for i in range(self.n_pars):
                self.p0[parameter_names[i]] = optimize.parameters.BoundedParameter(value=p0[i], lower_bound=lower_bounds[i], upper_bound=upper_bounds[i], vary=vary[i])
            self.n_pars_vary = self.p0.num_varied

        
        # Test pars
        self.test_pars = copy.deepcopy(self.p0)

        # Now unpack to numpy
        self.p0_numpy = self.p0.unpack()
        self.p0_numpy_vary = self.p0.unpack(vary_only=True)
        self.p0_vary_inds = np.where(self.p0_numpy['vary'])[0]
        
        # Resolve the number of fevals
        if self.max_f_evals is None:
            self.max_f_evals = self.p0.num_varied * 10_000
        
        # Number of iterations
        if self.n_iterations is None:
            self.n_iterations = self.p0.num_varied
   
    def initialize_simplex(self):
        """Initialize the initial full simplex.
        """
        self.current_full_simplex = np.tile(self.p0_numpy_vary['value'].reshape(self.n_pars_vary, 1), (1, self.n_pars_vary + 1))
        self.current_full_simplex[:, :-1] += np.diag(self.initial_scale_factor * self.p0_numpy_vary['value'])

    def initialize_subspaces(self):
        
        # Store each subspace in a list
        self.subspaces = []

        # Get varied pars
        p0_varied = self.p0.get_varied()

        # Compute each consecutive pair of params
        for i in range(len(p0_varied) - 1):
            self.subspaces.append([p0_varied[i].name, p0_varied[i + 1].name])

        # Circle back around (Last, First)
        self.subspaces.append([p0_varied[-1].name, p0_varied[0].name])
        
        # If only more than three varied params, also add [Second, Second to last]
        if len(p0_varied) > 3:
            self.subspaces.append([p0_varied[1].name, p0_varied[-2].name])
            
        # Compute additional helpful items
        self.subspace_inds = []
        self.subspace_inds_vary = []
        for s in self.subspaces:
            self.subspace_inds.append([])
            self.subspace_inds_vary.append([])
            for pname in s:
                self.subspace_inds[-1].append(self.p0.index_from_par(pname))
                self.subspace_inds_vary[-1].append(self.p0.index_from_par(pname, rel_vary=True))


    ##################
    #### OPTIMIZE ####
    ##################

    def optimize(self, p0, obj=None, lower_bounds=None, upper_bounds=None, parameter_names=None, obj_args=None, obj_kwargs=None):

        # Initialize
        self.initialize(p0, obj=obj, lower_bounds=lower_bounds, upper_bounds=upper_bounds, parameter_names=parameter_names, obj_args=obj_args, obj_kwargs=obj_kwargs)
        
        # Loop over iterations
        for iteration in range(self.n_iterations):
            
            if iteration > 0:
                dx_rel = self.compute_dx_rel(self.current_full_simplex)
                if dx_rel < self.xtol_rel:
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
        
        # Store status, fbest, fcalls
        out['status'] = self.status
        out['fbest'] = self.fmin
        out['fcalls'] = self.fcalls
            
        # Store best fit parameters
        if self._numpy:
            out['pbest'] = self.pmin.unpack(keys='value')['value']
        else:
            out['pbest'] = self.pmin
            

        return out

    def optimize_space(self, subspace_index=None):
        
        # Generate a simplex for this subspace
        simplex = self.initialize_subspace(subspace_index=subspace_index)

        # Current status
        self.status = "failed"
        
        # Alias the hyperparams
        # 1, 2, 0.5, 0.5
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
            if self.compute_df_rel(fmin, fnp1) > self.ftol_rel:
                n_converged = 0
            else:
                n_converged += 1
            if n_converged >= self.no_improve_break:
                self.status = "success"
                break

            # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
            # We want to iteratively replace the worst vector with a better vector.
            
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

        ind = np.argsort(fvals)
        fvals = fvals[ind]
        simplex = simplex[:, ind]
        fmin = fvals[0]
        pmin = simplex[:, 0]
        
        # Update the full simplex
        if subspace_index is None:
            for i, p in enumerate(self.p0_numpy_vary['name']):
                self.pmin[p].value = pmin[i]
            #self.current_full_simplex = np.copy(simplex)
        else:
            for i, p in enumerate(self.subspaces[subspace_index]):
                self.pmin[p].value = pmin[i]
            self.current_full_simplex[:, subspace_index] = self.pmin.unpack(vary_only=True)["value"]
        
        # Update the current function minimum
        self.fmin = fmin
               
    def initialize_subspace(self, subspace_index=None):
        
        if subspace_index is not None:
            n = len(self.subspaces[subspace_index])
            inds = self.subspace_inds[subspace_index]
            simplex = np.zeros((n, n+1))
            pmin = self.pmin.unpack(keys='value')['value'][inds]
            pinit = self.p0_numpy['value'][inds]
            simplex[:, 0] = np.copy(pinit)
            simplex[:, 1] = np.copy(pmin)
            for i in range(2, n + 1):
                simplex[:, i] = np.copy(pmin)
                j = i - 2
                simplex[j, i] = np.copy(pinit[j])
        else:
            simplex = np.copy(self.current_full_simplex)
            
        self.test_pars = copy.deepcopy(self.pmin)

        return simplex
    
    ###########################
    #### OBJECTIVE WRAPPER ####
    ###########################

    def compute_obj(self, x, subspace_index=None):
        
        if subspace_index is None:
            for i, p in enumerate(self.p0_numpy_vary['name']):
                self.test_pars[p].value = x[i]
        else:
            for i, p in enumerate(self.subspaces[subspace_index]):
                self.test_pars[p].value = x[i]
        
        # Call the objective
        if self._numpy:
            test_pars_vec = self.test_pars.unpack()['value']
            f = self.obj(test_pars_vec, *self.obj_args, **self.obj_kwargs)
        else:
            f = self.obj(self.test_pars, *self.obj_args, **self.obj_kwargs)

        # Max or min
        if self.maximize:
            f *= -1
        
        # Update fcalls
        self.fcalls += 1

        # Penalize
        if self.apply_penalty:
            f = self.penalize(self.test_pars, f)

        # Last resort, can't return inf or nan at this stage
        if not np.isfinite(f):
            f = 1E6
        
        # Return
        return f

    def penalize(self, pars, f):
        """Penalize the objective function for bounded parameters.
        """
        return f + pars.num_out_of_bounds * self.penalty


    ###################
    #### TOLERANCE ####
    ###################
    
    @staticmethod
    def compute_dx_rel(simplex):
        a = np.nanmin(simplex, axis=1)
        b = np.nanmax(simplex, axis=1)
        c = (np.abs(b) + np.abs(a)) / 2
        c = np.atleast_1d(c)
        ind = np.where(c < EPSILON)[0]
        if ind.size > 0:
            c[ind] = 1
        r = np.abs(b - a) / c
        return np.nanmax(r)

    @staticmethod
    @njit(numba.types.float64(numba.types.float64, numba.types.float64))
    def compute_df_rel(a, b):
        avg = (np.abs(a) + np.abs(b)) / 2
        return np.abs(a - b) / avg
    

    ###############
    #### MISC. ####
    ###############

    def __repr__(self):
        s = "Iterative Nelder Mead Optimizer:\n"
        s += f"  Number of iterations: {self.n_iterations}"
        return s