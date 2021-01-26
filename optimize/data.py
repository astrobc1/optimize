import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
    
class Data:
    """A base class for simple datasets. Additional datasets may ignore the slots and define their own attributes, but the memory usage will resort to the typical Python dict implementation. A __dict__ will be created unless a new __slots__ class attribute is used.
 
    Attributes:
        x (np.ndarray): The effective independent variable.
        y (np.ndarray): The effective dependent variable.
        yerr (np.ndarray): The intrinsic errorbars for y.
        mask (np.ndarray): An array defining good (=1) and bad (=0) data points, must have the same shape as y. Defaults to None (all good data).
        label (str): The label for this dataset.
    """
    
    __slots__ = ['x', 'y', 'yerr', 'mask', 'label']
    
    def __init__(self, x, y, yerr=None, mask=None, label=None):
        """Constructs a general dataset.

        Args:
            x (np.ndarray): The effective independent variable.
            y (np.ndarray): The effective dependent variable.
            yerr (np.ndarray): The intrinsic errorbars for y.
            mask (np.ndarray): An array defining good (=1) and bad (=0) data points, must have the same shape as y. Defaults to None (all good data).
            label (str): The label for this dataset.
        """
        self.x = x
        self.y = y
        self.yerr = yerr
        self.mask = mask
        self.label = label
        
    def __repr__(self):
        return 'A Simple Data Set'
        
    def compute_errorbars(self, pars):
        """Computes the effective error bars after including additional white noise ("jitter") terms. Errors are added in quadrature. Jitter params must be names label_jitter.

        Args:
            pars (Parameters): The parameters object containing the "jitter" parameter. If not present, the errorbars are returned.

        Returns:
            np.ndarray: The computed errorbars for this dataset.
        """
        if self.label is not None and self.label + "_jitter" in pars:
            return np.sqrt(self.yerr**2 + pars["jitter"].value**2)
        else:
            return self.yerr
        
class MixedData(dict):
    """A useful class to extend for composite data sets. Data sets of the same physical measurement, or different measurements of the same object may be utilized here. The labels of each dataset correspond the the keys of the dictionary.
    """
    
    def __init__(self):
        super().__init__()
    
    def __setitem__(self, label, data):
        if data.label is None:
            data.label = label
        super().__setitem__(label, data)
        
    def get(self, labels):
        """Returns a view into sub data objects.

        Args:
            labels (list): A list of labels (str).

        Returns:
            MixedData: A view into the original data object.
        """
        data_view = self.__class__()
        for label in labels:
            data_view[label] = self[label]
        return data_view
        