import numpy as np
from functools import lru_cache

class Data:
    """An base class for general datasets.
    """
    
    def __init__(self, label=None):
        self.label = label
    
class SimpleData(Data):
    """A base class for simple datasets.

    Attributes:
        x (np.ndarray): The effective independent variable.
        y (np.ndarray): The effective dependent variable.
        yerr (np.ndarray): The intrinsic errorbars for y.
        mask (np.ndarray): An array defining good (=1) and bad (=0) data points, must have the same shape as y. Defaults to None (all good data).
    """
    
    __slots__ = ['label', 'x', 'y', 'yerr', 'mask']
    
    def __init__(self, x, y, yerr=None, mask=None, label=None):
        """Constructs a general dataset.

        Args:
            x (np.ndarray): The effective independent variable.
            y (np.ndarray): The effective dependent variable.
            mask (np.ndarray): An array defining good (=1) and bad (=0) data points, must have the same shape as y. Defaults to None (all good data).
        """
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.yerr = yerr
        self.mask = mask
        
    def compute_errorbars(self, pars):
        """Computes the effective error bars after including additional white noise ("jitter") terms. Errors are added in quadrature.

        Args:
            pars (Parameters): The parameters object containing the "jitter" parameter. If not present, the errorbars are returned.

        Returns:
            np.ndarray: The computed errorbars.
        """
        if self.label + "_jitter" in pars:
            return np.sqrt(self.yerr**2 + pars["jitter"].value**2)
        else:
            return self.yerr
        
class MixedData(dict):
    
    def __init__(self):
        super().__init__() 

    #@lru_cache
    def unpack(self, keys, subkey="x"):
        if type(keys) is str:
            keys = [keys]
        out = np.zeros(len(self[keys[0]]))
        for key in keys:
            _data = getattr(self[key], subkey)
            out = np.concatenate((out, _data))
        return out