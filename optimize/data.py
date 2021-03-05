# Numpy
import numpy as np

class Data:
    """A base class for datasets. Additional datasets may ignore the slots and define their own attributes, but the memory usage will resort to the typical Python dict implementation. A __dict__ will be created unless a new __slots__ class attribute is used.
 
    Attributes:
        x (np.ndarray): The effective independent variable.
        y (np.ndarray): The effective dependent variable.
        yerr (np.ndarray): The intrinsic errorbars for y.
        mask (np.ndarray): An array defining good (=1) and bad (=0) data points, must have the same shape as y. Defaults to None (all good data).
        label (str): The label for this dataset. Defaults to None.
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
        return 'Data: ' + self.label
    
class Data1d:
    """A base class for 1d datasets.
 
    Attributes:
        x (np.ndarray): The effective independent variable.
        y (np.ndarray): The effective dependent variable.
        yerr (np.ndarray): The intrinsic errorbars for y.
        mask (np.ndarray): An array defining good (=1) and bad (=0) data points, must have the same shape as y. Defaults to None (all good data).
        label (str): The label for this dataset. Defaults to None.
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
        return 'Data 1d: ' + self.label
        
class CompositeData(dict):
    """A useful class to extend for composite data sets. Data sets of the same physical measurement, or different measurements of the same object may be utilized here. The labels of each dataset correspond the the keys of the dictionary.
    """
    
    def __init__(self):
        super().__init__()
        self.label_vec = self.make_label_vec()
    
    def __setitem__(self, label, data):
        if data.label is None:
            data.label = label
        super().__setitem__(label, data)
        self.label_vec = self.make_label_vec()
        
    def get(self, labels):
        """Returns a view into sub data objects.

        Args:
            labels (list): A list of labels (str).

        Returns:
            CompositeData: A view into the original data object.
        """
        data_view = self.__class__()
        for label in labels:
            data_view[label] = self[label]
        return data_view
    
    def get_inds(self, label):
        inds = np.where(self.label_vec == label)[0]
        return inds
    
    def get_vec(self, key, labels=None, sort=True):
        """Combines a certain vector from all labels into one array, and can then sort it according to x.

        Args:
            key (str): The key to get (x, y, yerr, mask)
            labels (list): A list of labels (dict keys)

        Returns:
            np.ndarray: The vector, sorted according to x.
        """
        if labels is None:
            labels = list(self.keys())
        out = np.array([], dtype=float)
        if sort:
            x = np.array([], dtype=float)
        for label in labels:
            assert isinstance(self[label], Data1d)
            out = np.concatenate((out, getattr(self[label], key)))
            if sort:
                x = np.concatenate((x, self[label].x))
        # Sort
        if sort:
            ss = np.argsort(x)
            out = out[ss]

        return out
        
    def make_label_vec(self):
        label_vec = np.array([], dtype='<U50')
        x = self.get_vec('x', sort=False)
        for label in self:
            label_vec = np.concatenate((label_vec, np.full(len(self[label].x), fill_value=label, dtype='<U50')))
        ss = np.argsort(x)
        label_vec = label_vec[ss]
        return label_vec
        
        