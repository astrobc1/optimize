# Science / maths
import numpy as np

class Data:
    """A base class for datasets. Additional datasets may ignore the slots and define their own attributes, but the memory usage will resort to the Python dict implementation for classes. Defining a new __slots__ class attribute for the new class will avoid this.
 
    Attributes:
        label (str): The label for this dataset. Defaults to "User Data".
    """
    
    def __init__(self, label=None):
        """Constructs a general dataset, simply stores a label for now.

        Args:
            label (str): The label for this dataset.
        """
        
        self.label = label
        
    def __repr__(self):
        return 'Data: ' + self.label

 
class DataS1d(Data):
    """A base class for 1d->1d datasets.
 
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
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.yerr = yerr
        self.mask = mask
        
    def __repr__(self):
        return f"Data S1d {self.label}"


class CompositeData(dict):
    """A useful class to extend for composite 1d data sets. Data sets of the same physical measurement, or different measurements of the same object may be utilized here. The labels of each dataset correspond the the keys of the dictionary. The independent variable may be can be composed of a value or array of observations.
    """

    def get(self, labels):
        """Returns a view into sub data objects.

        Args:
            labels (str or list of strings): A 

        Returns:
            CompositeData: A view into the original data object.
        """
        data_view = self.__class__()
        labels = np.atleast_1d(labels)
        for label in labels:
            data_view[label] = self[label]
        return data_view
    
    def __setitem__(self, label, data):
        if data.label is None:
            data.label = label
        super().__setitem__(label, data)
        

class CompositeDataS1d(CompositeData):
    """A useful class to extend for composite 1d data sets where there is a bijection between measurements AND each measurement is represented by a an independent value x (float), measurement y (float), and uncertainy yerr (float, identical upper and lower values).
    """
    
    def __init__(self):
        super().__init__()
        self.indices = {}
    
    def gen_label_vec(self):
        """Generates a vector where each index corresponds to the label of measurement x, sorted by x as well.

        Returns:
            np.ndarray: The label vector in order of what makes sense for x according to np.argsort(x)
        """
        label_vec = np.array([], dtype='<U50')
        x = self.gen_vec('x', sort=False)
        for data in self.values():
            label_vec = np.concatenate((label_vec, np.full(len(data.x), fill_value=data.label, dtype='<U50')))
        ss = np.argsort(x)
        label_vec = label_vec[ss]
        return label_vec
    
    def gen_vec(self, key, labels=None, sort=True):
        """Combines a certain vector from all labels into one vector, and can then sort it according to x if sort=True.

        Args:
            key (str): The key to get, must be an attribute of each data object.
            labels (list): A list of labels.
            sort (bool): Whether or not to sort the returned vector.

        Returns:
            np.ndarray: The vector, sorted according to x if sort=True is set.
        """
        if labels is None:
            labels = list(self.keys())
        else:
            labels = np.atleast_1d(labels)
        out = np.array([], dtype=float)
        if sort:
            x = np.array([], dtype=float)
        for label in labels:
            out = np.concatenate((out, getattr(self[label], key)))
            if sort:
                x = np.concatenate((x, self[label].x))
        # Sort
        if sort:
            ss = np.argsort(x)
            out = out[ss]

        return out
    
    def __setitem__(self, label, data):
        super().__setitem__(label, data)
        self.label_vec = self.gen_label_vec()
        for data in self.values():
            inds = np.where(self.label_vec == data.label)[0]
            self.indices[data.label] = inds
        self.n = len(self.label_vec)
        
    def __delitem__(self, key):
        super().__delitem__(key)
        del self.indices[key]
        self.label_vec = self.gen_label_vec()
        for data in self.values():
            inds = np.where(self.label_vec == data.label)[0]
            self.indices[data.label] = inds
        self.n = len(self.label_vec)
            