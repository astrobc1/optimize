import numpy as np
import optimize.knowledge as optknow
import copy

##########################
#### SINGLE PARAMETER ####
##########################

class Parameter:
    """A base class for a scalar model parameter.

    Attributes:
        name (str): The name of the parameter.
        value (str): The current value of the parameter.
        vary (bool): Whether or not to vary this parameter.
        latex_str (str): A string for plot formatting, most likely using latex math-mode formatting $$.
    """
    
    __slots__ = ['name', 'value', 'vary', 'latex_str']
    
    #####################
    #### CONSTRUCTOR ####
    #####################

    def __init__(self, name=None, value=None, vary=True, latex_str=None):
        """Creates a Parameter object.

        Args:
            name (str): The name of the parameter.
            value (float): The starting value of the parameter.
            vary (bool): Whether or not to vary (optimize) this parameter.
            latex_str (str): A string for latex formatting in math mode.
        """
        
        # Set all attributes
        self.name = name
        self.value = value
        self.vary = vary
        self.latex_str = self.name if latex_str is None else latex_str
    
    ###############
    #### MISC. ####
    ###############
    
    def set_name(self, name):
        """Sets the name for the parameter. Also sets the latex string of not already set.

        Args:
            name (str): The parameter name.
        """
        self.name = name
        if self.latex_str is None:
            self.latex_str = name
        
    def __repr__(self):
        s = f"Name: {self.name} | Value: {self.value}"
        if not self.vary:
            s += ' (Locked)'
        return s
     
    @property
    def value_str(self):
        """The current value of the parameter as a string

        Returns:
            str: The value as a string
        """
        return f"{self.value}"

    def gen_nan_pars(self):
        pars = copy.deepcopy(self)
        for par in pars:
            par.value = np.nan
        return pars

class BoundedParameter(Parameter):
    """A class for a bounded model parameter.

    Attributes:
        name (str): The name of the parameter.
        value (str): The current value of the parameter.
        vary (bool): Whether or not to vary (optimize) this parameter.
        lower_bound (float): The lower bound.
        upper_bound (float): The upper bound.
        latex_str (str): A string for plot formatting, most likely using latex formatting.
    """
    
    __slots__ = ['name', 'value', 'vary', 'lower_bound', 'upper_bound', 'latex_str']
    
    #####################
    #### CONSTRUCTOR ####
    #####################

    def __init__(self, name=None, value=None, vary=True, lower_bound=-np.inf, upper_bound=np.inf, latex_str=None):
        """Creates a Parameter object.

        Args:
            name (str): The name of the parameter.
            value (float): The starting value of the parameter.
            vary (bool): Whether or not to vary (optimize) this parameter.
            lower_bound (float): The lower bound.
            upper_bound (float): The upper bound.
            latex_str (str): A string for plot formatting, most likely using latex formatting.
        """
        
        # Set attributes
        super().__init__(name=name, value=value, vary=vary, latex_str=latex_str)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.latex_str = self.name if latex_str is None else latex_str

    ###############
    #### MISC. ####
    ###############
    
    def sanity_lock(self):
        if self.lower_bound == self.upper_bound and self.vary:
            self.vary = False

    def __repr__(self):
        s = f"Name: {self.name} | Value: {self.value}"
        if not self.vary:
            s += ' (Locked)'
        s += f" | Bounds: {self.lower_bound}, {self.upper_bound}"
        return s
        
    @property
    def hard_bounds(self):
        """Gets the current hard bounds.

        Returns:
            float: The lower bound.
            float: The upper bound.
        """
        return self.lower_bound, self.upper_bound
    
    @property
    def in_bounds(self):
        """Determines whether or not the current parameter value is in bounds.

        Returns:
            bool: True if the parameter is in bounds, False if not.
        """
        return self.value > self.lower_bound and self.value < self.upper_bound
    
    @property
    def out_of_bounds(self):
        """Determines whether or not the current parameter value out of bounds.

        Returns:
            bool: True if the parameter is out of bounds, False if not.
        """
        return not self.in_bounds

class BayesianParameter(Parameter):
    
    """A class for a Bayesian model parameter.

    Attributes:
        name (str): The name of the parameter.
        value (str): The current value of the parameter.
        vary (bool): Whether or not to vary (optimize) this parameter.
        priors (list): A list of priors to apply.
        scale (float): A search scale to initiate mcmc walkers.
        latex_str (str): A string for plot formatting, most likely using latex formatting.
    """
    
    __slots__ = ['name', 'value', 'vary', 'priors', '_scale', 'latex_str', 'unc']
    
    #####################
    #### CONSTRUCTOR ####
    #####################

    def __init__(self, name=None, value=None, vary=True, priors=None, scale=None, latex_str=None, unc=None):
        """Creates a Bayesian Parameter object.

        Args:
            name (str): The name of the parameter.
            value (float): The starting value of the parameter.
            vary (bool): Whether or not to vary (optimize) this parameter.
            priors (list): A list of priors to apply.
            latex_str (str): A string for plot formatting, most likely using latex formatting.
            scale (float): A search scale to initiate mcmc walkers.
            unc (tuple): A tuple containing the uncertainty in this parameter (-, +).
        """
        
        # Set all attributes
        self.name = name
        self.value = value
        self.vary = vary
        if priors is None:
            self.priors = []
        elif type(priors) is optknow.Prior:
            self.priors = [self.priors]
        self._scale = scale
        self.latex_str = self.name if latex_str is None else latex_str
        self.unc = None
    
    ################
    #### PRIORS ####
    ################
    
    def has_prior(self, prior_type):
        """Checks whether or not the parameter contains a certain type of prior

        Args:
            prior_type (type): The type of the prior.

        Returns:
            bool: True if the parameter has the prior, False otherwise.
        """
        for prior in self.priors:
            if type(prior) is prior_type:
                return True
        return False
    
    def add_prior(self, prior):
        """Adds a new prior. Equivalent to self.priors.append(prior)

        Args:
            prior (Prior): The prior to add.
        """
        self.priors.append(prior)
            
    ###############
    #### MISC. ####
    ###############
        
    @property    
    def scale(self):
        """The scale for the parameter to initiate MCMC walkers.

        Returns:
            float: The parameter scale.
        """
        if self._scale is not None:
            return self._scale
        if len(self.priors) == 0:
            scale = np.abs(self.value) / 100
            if scale == 0:
                return 1
            else:
                return scale
        for prior in self.priors:
            if isinstance(prior, optknow.priors.Gaussian):
                return prior.sigma / 10
        for prior in self.priors:
            if isinstance(prior, optknow.priors.Uniform):
                dx1 = np.abs(prior.upper_bound - self.value)
                dx2 = np.abs(self.value - prior.lower_bound)
                scale = np.min([dx1, dx2]) / 100
                if scale == 0:
                    return 1
                else:
                    return scale
        scale = np.abs(self.value) / 100
        if scale == 0:
            return 1
        else:
            return scale
        
    @property
    def hard_bounds(self):
        """Gets the hard bounds from uniform priors or Jeffreys prior if present, otherwise assumes +/- inf.

        Returns:
            float: The lower bound.
            float: The upper bound.
        """
        vlb, vub = -np.inf, np.inf
        if len(self.priors) > 0:
            for prior in self.priors:
                if isinstance(prior, Uniform):
                    vlb, vub = prior.lower_bound, prior.upper_bound
                    return vlb, vub
                if isinstance(prior, JeffreysG):
                    vlb, vub = prior.lower_bound, prior.upper_bound
                    return vlb, vub
        return vlb, vub
    
    @property
    def upper_bound(self):
        """The upper bound.

        Returns:
            float: The upper bound.
        """
        return self.hard_bounds[1]
    
    @property
    def lower_bound(self):
        """The lower bound.

        Returns:
            float: The lower bound.
        """
        return self.hard_bounds[0]
        
    def __repr__(self):
        s = f"Name: {self.name} | Value: {self.value}"
        if not self.vary:
            s += ' (Locked)'
        if self.unc is not None:
            s += f" | Unc: -{self.unc[0]}, +{self.unc[1]}"
        if len(self.priors) > 0:
            s += '\n  Priors:\n'
            for prior in self.priors:
                s += "   " + prior.__repr__() + "\n"
        return s
    

#############################
#### MULTIPLE PARAMETERS ####
#############################

class Parameters(dict):
    """A container for a set of model parameters which extends the Python 3 dictionary, which is ordered by default.
    """
    
    default_keys = Parameter.__slots__
    
    ##################
    #### TO NUMPY ####
    ##################
    
    def unpack(self, keys=None, vary_only=False):
        """Unpacks values to a dict of numpy arrays.

        Args:
            keys (iterable or string): A tuple of strings containing the keys to unpack, defaults to None for all keys.
            
        Returns:
            dict: A dictionary containing the numpy arrays.
        """
        if keys is None:
            keys = self.default_keys
        else:
            t = type(keys)
            if t is str:
                keys = [keys]
        out = {}
        if vary_only:
            for key in keys:
                t = type(getattr(self[0], key))
                if t is int:
                    t = float
                out[key] = np.array([getattr(self[pname], key) for pname in self if self[pname].vary], dtype=t)
        else:
            for key in keys:
                t = type(getattr(self[0], key))
                if t is int:
                    t = float
                out[key] = np.array([getattr(self[pname], key) for pname in self], dtype=t)

        return out

    ###########################
    #### ADD NEW PARAMETER ####
    ###########################

    def add_par(self, par):
        """Adds a parameter to the Parameters dictionary with par.name as a key.

        Args:
            par (Parameter): The parameter to add.
        """
        self[par.name] = par
                
    ###############
    #### MISC. ####
    ###############
                
    def get_view(self, par_names):
        """Gets a subspace of parameter objects.
        
        Args:
            pars (iterable of str): An iterable of string objects (the names of the parameters to fetch).

        Returns:
            type(self): A parameters object containing pointers to the desired parameters. The class will be the same as the current instance.
        """
        sub_pars = self.__class__()
        for pname in par_names:
            sub_pars.add_par(self[pname])
        return sub_pars
    
    def par_from_index(self, k, rel_vary=False):
        """Gets the parameter at a given numerical index.

        Args:
            k (int): The numerical index.
            rel_vary (bool, optional): Whether or not this index is relative to all parameters or only varied parameters. Defaults to False.

        Returns:
            Parameter: The parameter at the given index.
        """
        if rel_vary:
            return self[list(self.get_varied().keys())[k]]
        else:
            return self[list(self.keys())[k]]
    
    def index_from_par(self, name, rel_vary=False):
        """Gets the index of a given parameter name.

        Args:
            name (str): The name of the parameter.
            rel_vary (bool, optional): Whether or not to return an index which is relative to all parameters or only varied parameters. Defaults to False.

        Returns:
            int: The numerical index of the parameter.
        """
        if rel_vary:
            return list(self.get_varied().keys()).index(name)
        else:
            return list(self.keys()).index(name)
    
    def get_varied(self):
        """Gets the varied parameters in a new parameters object

        Returns:
            Parameters: A parameters object containing pointers to only the varied parameters.
        """
        varied_pars = self.__class__()
        for pname in self:
            if self[pname].vary:
                varied_pars.add_par(self[pname])
        return varied_pars
    
    def get_locked(self):
        """Gets the locked parameters in a new parameters object

        Returns:
            Parameters: A parameters object containing pointers to only the locked parameters.
        """
        locked_pars = self.__class__()
        for pname in self:
            if not self[pname].vary:
                locked_pars.add_par(self[pname])
        return locked_pars
    
    def set_vec(self, x, attr, varied=False):
        """Sets an attribute for all parameters given a vector.

        Args:
            x (np.ndarray): The vector to set.
            attr (str): The attribute to set for each parameter.
            varied (bool, optional): Whether or not to only set the varied parameters. Defaults to False.
        """
        if varied:
            i = 0
            for par in self.values():
                if par.vary:
                    setattr(par, attr, x[i])
                    i += 1
        else:
            for i, par in enumerate(self.values()):
                setattr(par, attr, x[i])
    
    def __setitem__(self, key, par):
        if par.name is None:
            par.set_name(key)
        super().__setitem__(key, par)
    
    def __getitem__(self, key):
        t = type(key)
        if t is str or t is np.str or t is np.str_:
            return super().__getitem__(key)
        elif t is int:
            return self[list(self.keys())[key]]
        
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            return super().__getattr__(attr)
    
    def __repr__(self):
        s = "Parameters:\n"
        for par in self.values():
            s += f"  {repr(par)}\n"
        return s
    
    @property
    def varied_inds(self):
        """The indices of the varied parameters.

        Returns:
            np.dnarray: The indices of the varied parameters.
        """
        return np.array([i for i, par in enumerate(self.values()) if par.vary], dtype=int)
    
    @property
    def num_varied(self):
        """The number of varied parameters.

        Returns:
            int: The number of varied parameters.
        """
        n = 0
        for par in self.values():
            if par.vary:
                n += 1
        return n
    
    @property
    def num_locked(self):
        """The number of locked parameters.

        Returns:
            int: The number of locked parameters.
        """
        n = 0
        for par in self.values():
            if par.vary:
                n += 1
        return n

class BoundedParameters(Parameters):
    
    default_keys = BoundedParameter.__slots__
    
    ###############
    #### MISC. ####
    ###############
    
    def sanity_lock(self):
        for par in self.values():
            par.sanity_lock()
    
    @property
    def num_in_bounds(self):
        """The number of parameters in bounds.
        
        Returns:
            int: The number of parameters in bounds.
        """
        n = 0
        for par in self.values():
            if par.in_bounds and par.vary:
                n += 1
        return n
    
    @property
    def num_out_of_bounds(self):
        """The number of parameters out of bounds.
        
        Returns:
            int: The number of parameters out of bounds.
        """
        n = 0
        for par in self.values():
            if par.out_of_bounds and par.vary:
                n += 1
        return n
    
    @property
    def hard_bounds(self):
        """The hard bounds as a tuple.

        Returns:
            np.ndarray: The lower bounds.
            np.ndarray: The upper bounds.
        """
        return self.lower_bounds, self.upper_bounds
    
    @property
    def upper_bounds(self):
        """The upper bounds.

        Returns:
            np.ndarray: The upper bounds.
        """
        return np.array([par.upper_bound for par in self.values()], dtype=float)
    
    @property
    def lower_bounds(self):
        """The lower bounds.

        Returns:
            np.ndarray: The lower bounds.
        """
        return np.array([par.lower_bound for par in self.values()], dtype=float)
    
    @property
    def hard_bounds_varied(self):
        """Gets the hard bounds, but only the varied params.

        Returns:
            np.ndarray: The lower bounds.
            np.ndarray: The upper bounds.
        """
        return self.lower_bounds[self.varied_inds], self.upper_bounds[self.varied_inds]
    
    @property
    def all_in_bounds(self):
        """Checks whether or not all parameters are in bounds (or not.)

        Returns:
            bool: True if the parameters are all in bounds, False otherwise.
        """
        for par in par.values():
            if par.out_of_bounds:
                return False
        return True
    
    @property
    def any_out_of_bounds(self):
        """Checks whether or not any parameters are out of bounds.

        Returns:
            bool: True if any parameters are out of bounds, False otherwise.
        """
        return not self.all_in_bounds

    def __repr__(self):
        s = "Bounded Parameters:\n"
        for par in self.values():
            s += f"  {repr(par)}\n"
        return s

class BayesianParameters(Parameters):
    
    default_keys = BayesianParameter.__slots__
    
    @property
    def scales(self):
        """The scales for each parameter.

        Returns:
            np.ndarray: The scales for each parameter.
        """
        return np.array([par.scale for par in self.values()], dtype=float)
    
    def __repr__(self):
        s = "Bayesian Parameters:\n"
        for par in self.values():
            s += f"  {repr(par)}\n"
        return s