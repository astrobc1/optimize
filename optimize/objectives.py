# Base Python
import types
import inspect

# Maths
import numpy as np

####################
#### BASE TYPES ####
####################

class ObjectiveFunction:
    """An base class for a general objective function. Not useful to instantiate on its own.
    """

    def __init__(self, **kwargs):
        # Set all kwargs by default
        for key in kwargs:
            setattr(self, key, kwargs[key])

    ###########################
    #### COMPUTE OBJECTIVE ####
    ###########################

    def __call__(self, *args, **kwargs):
        return self.compute_obj(*args, **kwargs)

    def compute_obj(self, *args, **kwargs):
        raise NotImplementedError(f"Must implement method compute_obj for class {self.__class__.__name__}")

    ###################
    #### RESIDUALS ####
    ###################

    def compute_residuals(self, *args, **kwargs):
        raise NotImplementedError(f"Must implement method compute_residuals for class {self.__class__.__name__}")


    ###############
    #### MISC. ####
    ###############

    def __repr__(self):
        return f"Objective Function: {self.__class__.__name__}"

    def __setattr__(self, key, attr):
        if callable(attr) and not inspect.isclass(attr):
            self.register_method(attr, key)
        else:
            super().__setattr__(key, attr)

    def register_method(self, method, method_name):
        if '.' not in method.__qualname__ and method.__code__.co_varnames[0] == "self":
            super().__setattr__(method_name, types.MethodType(method, self))
        else:
            super().__setattr__(method_name, method)


class RMSLoss(ObjectiveFunction):

    def __init__(self, flag_worst=0, remove_edges=0, **kwargs):
        super().__init__(**kwargs)
        self.flag_worst = flag_worst
        self.remove_edges = remove_edges

    def compute_obj(self, pars, *args, **kwargs):
        residuals = self.compute_residuals(pars)
        rms = self.rmsloss(residuals, flag_worst=self.flag_worst, remove_edges=self.remove_edges)
        return rms
        
    @staticmethod
    def rmsloss(residuals, weights=None, flag_worst=0, remove_edges=0):
        """Convenient method to compute the weighted RMS between two vectors x and y.

        Args:
            residuals (np.ndarray): The residuals array, where residuals = data - model.
            weights (np.ndarray, optional): The weights. Defaults to uniform weights.
            flag_worst (int, optional): Flag the largest outliers (with weights applied). Defaults to 0.
            remove_edges (int, optional): Ignore this number of edges on each side. Note this is only relevant for 1d data. Defaults to 0.

        Returns:
            float: The weighted RMS.
        """
        
        # Compute diffs2
        if weights is not None:
            good = np.where(np.isfinite(residuals) & np.isfinite(weights) & (weights > 0))[0]
            res2, ww = residuals[good]**2, weights[good]
        else:
            good = np.where(np.isfinite(residuals))[0]
            res2 = residuals[good]**2
        
        # Ignore worst N pixels
        if flag_worst > 0:
            ss = np.argsort(res2)
            res2[ss[-1*flag_worst:]] = np.nan
            if weights is not None:
                ww[ss[-1*flag_worst:]] = 0
                    
        # Remove edges
        if remove_edges > 0:
            xi, xf = good.min(), good.max()
            res2[xi:xi+remove_edges] = np.nan
            res2[xf-remove_edges:] = np.nan
            if weights is not None:
                ww[xi:xi+remove_edges] = 0
                ww[xf-remove_edges:] = 0
            
        # Compute rms
        if weights is not None:
            rms = np.sqrt(np.nansum(res2) / np.nansum(ww))
        else:
            n_good = np.where(np.isfinite(res2) & (res2 > 0))[0].size
            rms = np.sqrt(np.nansum(res2) / n_good)

        return rms


class Chi2Loss(ObjectiveFunction):

    def __init__(self, flag_worst=0, remove_edges=0, **kwargs):
        super().__init__(**kwargs)
        self.flag_worst = flag_worst
        self.remove_edges = remove_edges

    def compute_obj(self, pars):
        residuals = self.compute_residuals(pars)
        errors = self.compute_data_errors(pars)
        n_dof = self.compute_dof(residuals, pars)
        redchi2 = self.redchi2loss(residuals, errors, n_dof)
        return redchi2

    def compute_data_errors(self, *args, **kwargs):
        raise NotImplementedError(f"Must implement method compute_data_errors for class {self.__class__.__name__}")

    def compute_dof(self, residuals, pars):
        n_good = np.where(np.isfinite(residuals) & (residuals != 0))[0].size
        n_dof = n_good - pars.num_varied
        return n_dof

    @staticmethod
    def redchi2loss(residuals, errors, n_dof):
        return np.nansum((residuals / errors)**2) / n_dof