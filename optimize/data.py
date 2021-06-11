# Science / maths
import numpy as np

####################
#### BASE TYPES ####
####################

class Dataset:
    """A base class for datasets. This class is not useful to instantiate on its own.
 
    Attributes:
        label (str): The label for this dataset. Defaults to None.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, label=None):
        """Constructs a general dataset, simply stores a label for now.

        Args:
            label (str): The label for this dataset.
        """
        
        self.label = label
        
    #######################
    #### GET TRAINABLE ####
    #######################
        
    def get_trainable(self, *args, **kwargs):
        """Gets the trainable dataset as a numpy array.

        Raises:
            NotImplementedError: Must implement this method for a given subclass.
        """
        raise NotImplementedError(f"Must implement a method get_trainable for class {self.__class__.__name__}")
        
    #########################
    #### GET DATA ERRORS ####
    #########################
    
    def get_apriori_errors(self):
        """Gets the aprior errors (likely known beforehand and provided by the user).

        Raises:
            NotImplementedError Must implement this method.
        """
        raise NotImplementedError(f"Must implement a method get_apriori_errors for class {self.__class__.__name__}")
        
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, pars):
        """Called before optimization routines.

        Args:
            pars (Parameters): The parameters.
        """
        pass
        
    def __repr__(self):
        return f"Data: {self.label}"


class CompositeDataset(Dataset, dict):
    """A useful class to extend for composite data sets.
    """
    
    #######################
    #### GET TRAINABLE ####
    #######################
    
    def get_trainable(self, labels=None):
        raise NotImplementedError(f"Must implement a method get_trainable for class {self.__class__.__name__}")
    
    
    ###############
    #### MISC. ####
    ###############
    
    def get_view(self, labels):
        """Returns a view into sub data objects.

        Args:
            labels (str or list of strings): The labels to get.

        Returns:
            type(self): A view into the original data object.
        """
        out = self.__class__()
        labels = np.atleast_1d(labels)
        for label in labels:
            out[label] = self[label]
        return out
    
    def __setitem__(self, label, data):
        if data.label is None:
            data.label = label
        dict.__setitem__(self, label, data)
        
    def __repr__(self):
        s = "Composite Data:\n"
        for data in self.values():
            s += f"  {repr(data)}\n"
        return s

        

#####################
#### BASIC TYPES ####
#####################

class SimpleSeries(Dataset):
    """A class for a 1d->1d homogeneous dataset, sorted according to x. This class may be used directly or extended.
 
    Attributes:
        x (np.ndarray): The effective independent variable.
        y (np.ndarray): The effective dependent variable.
        yerr (np.ndarray, optional): The intrinsic errorbars for y. Defaults to None.
        label (str): The label for this dataset. Defaults to None.
    """
    
    __slots__ = ['x', 'y', 'yerr', 'label']
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, x, y, yerr=None, label=None):
        """Constructs a series dataset.

        Args:
            x (np.ndarray): The effective independent variable.
            y (np.ndarray): The effective dependent variable.
            yerr (np.ndarray): The intrinsic errorbars for y. Defaults to None. If not provided, one cannot use noise-based modeling.
            label (str): The label for this dataset. Defaults to None.
        """
        super().__init__(label)
        self.x = x
        self.y = y
        self.yerr = yerr
    
    #######################
    #### GET TRAINABLE ####
    #######################
    
    def get_trainable(self):
        return self.y
    
    #########################
    #### GET DATA ERRORS ####
    #########################
    
    def get_apriori_errors(self):
        return self.yerr
    
    def get_trainable_errors(self, *args, **kwargs):
        return self.yerr
    
    
    ###############
    #### MISC. ####
    ###############
    
    def __repr__(self):
        return f"Homogeneous Series: {self.label}"

class HomogeneousCompositeSimpleSeries(CompositeDataset):
    """A useful class to extend for composite 1d datasets which are all instances of SimpleSeries on the same data set.
    
    Attributes:
        indices (dict): The indices for each dataset when sorted according to x.
    
    Properties:
        x (np.ndarray): The x vector.
        y (np.ndarray): The y vector.
        yerr (np.ndarray): The yerr vector.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self):
        super().__init__()
        self.indices = {}
    
    #######################
    #### GET TRAINABLE ####
    #######################
        
    def get_trainable(self):
        """Gets the trainable dataset as a numpy array.

        Returns:
            np.ndarray: Returns self.y.
        """
        return self.y
        
    ##########################
    #### COLLECT VECTORS #####
    ##########################
    
    @property
    def n(self):
        """The number of data points.

        Returns:
            int: The number of data points.
        """
        n = 0
        for data in self.values():
            n += len(data.x)
        return n
    
    @property
    def x(self):
        """The x vector.

        Returns:
            np.ndarray: The sorted x vector.
        """
        return self.get_vec("x")
    
    @property
    def y(self):
        """The y vector.

        Returns:
            np.ndarray: The y vector sorted according to x.
        """
        return self.get_vec("y")
    
    @property
    def yerr(self):
        """The yerr vector.

        Returns:
            np.ndarray: The yerr vector sorted according to x.
        """
        return self.get_vec("yerr")
    
    def get_vec(self, attr, sort=True, labels=None):
        """Gets a vector for certain labels and possibly sorts it.

        Args:
            attr (str): The attribute to get.
            sort (bool): Whether or not to sort the returned vector.
            labels (list of strings, optional): The labels to get. Defaults to all.

        Returns:
            np.ndarray: The vector, sorted according to x if sort=True.
        """
        if labels is None:
            labels = list(self.keys())
        if sort:
            out = np.zeros(self.n)
            for label in labels:
                out[self.indices[label]] = getattr(self[label], attr)
        else:
            out = np.array([], dtype=float)
            for label in labels:
                out = np.concatenate((out, getattr(self[label], attr)))
        return out
    
    #########################
    #### GET DATA ERRORS ####
    #########################
    
    def get_apriori_errors(self):
        return self.yerr
    
    ###############
    #### MISC. ####
    ###############
    
    def gen_label_vec(self):
        """Generates a vector where each index corresponds to the label of measurement x, sorted by x as well.

        Returns:
            np.ndarray: The label vector sorted according to self.indices.
        """
        label_vec = np.empty(self.n, dtype='<U50')
        for data in self.values():
            label_vec[self.indices[data.label]] = data.label
        return label_vec
    
    def gen_indices(self):
        """Utility function to generate the indices of each dataset (when sorted according to x).

        Returns:
            dict: A dictionary with keys = data labels, values = numpy array of indices (ints).
        """
        indices = {}
        label_vec = np.array([], dtype="<U50")
        x = np.array([], dtype=float)
        for data in self.values():
            x = np.concatenate((x, data.x))
            label_vec = np.concatenate((label_vec, np.full(len(data.x), fill_value=data.label, dtype="<U50")))
        ss = np.argsort(x)
        label_vec = label_vec[ss]
        for data in self.values():
            inds = np.where(label_vec == data.label)[0]
            indices[data.label] = inds
        return indices
    
    def __setitem__(self, label, data):
        super().__setitem__(label, data)
        self.indices = self.gen_indices()
    
    def __delitem__(self, key):
        super().__delitem__(key)
        self.indices = self.gen_indices()