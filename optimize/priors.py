# Maths
import numpy as np

###################
#### BASE TYPE ####
###################

class Prior:
    """An interface for a general Bayesian prior.
    """

    def logprob(self, x):
        """Computes the logprobability given the current parameter value, x.

        Args:
            x (float): The current parameter value.
        Returns:
            float: The logarithm of the probability distribution for this value.
        """
        raise NotImplementedError(f"Must implement the method logprob for class {self.__class__.__name__}.")
    
    def __repr__(self):
        return f"Prior: {self.__class__.__name__}"


#########################
#### CONCRETE PRIORS ####
#########################

class Gaussian(Prior):
    """A prior defined by a normal distribution.

    Attributes:
        mu (float): The center of the distribution.
        sigma (float): The stddev. of the distribution.
    """
    
    __slots__ = ['mu', 'sigma']
    name = 'Gaussian'
    
    def __init__(self, mu, sigma):
        """Constructor for a Gaussian prior.

        Args:
            mu (float): The center of the distribution.
            sigma (float): The stddev. of the distribution.
        """
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma
        
    def logprob(self, x):
        """Computes the log probability given the current parameter value, x.

        Args:
            x (float): The current parameter value.
        Returns:
            float: The logarithm of the probability distribution for this value.
        """
        return -0.5 * ((x - self.mu) / self.sigma)**2 - 0.5 * np.log((self.sigma**2) * 2 * np.pi)
    
    def __repr__(self):
        return f"{self.name}: [{self.mu}, {self.sigma}]"
    
class Uniform(Prior):
    """A prior defined by hard bounds.

        Attributes:
            lower_bound (float): The lower bound.
            upper_bound (float): The upper bound.
        """
    
    __slots__ = ['lower_bound', 'upper_bound']
    
    name = "Uniform"
    
    def __init__(self, lower_bound, upper_bound):
        """Constructor for a Uniform prior.

        Args:
            lower_bound (float): The lower bound.
            upper_bound (float): The upper bound.
        """
        assert lower_bound < upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def logprob(self, x):
        """Computes the log probability given the current parameter value, x.

        Args:
            x (float): The current parameter value.
        Returns:
            float: The logarithm of the probability distribution for this value.
        """
        if self.lower_bound < x < self.upper_bound:
            return -1 * np.log(self.upper_bound - self.lower_bound)
        else:
           return -np.inf
        
    def __repr__(self):
        return f"{self.name}: [{self.lower_bound}, {self.upper_bound}]"

class Positive(Prior):
    """A prior to force x > 0.
    """
    
    __slots__ = []
    
    name = "Positive"
    
    def __init__(self):
        """Constructs a positive prior.
        """
        pass
        
    def logprob(self, x):
        """Computes the log probability given the current parameter value, x.

        Args:
            x (float): The current parameter value.
        Returns:
            float: The logarithm of the probability distribution for this value.
        """
        return 0 if x > 0 else -np.inf
        
    def __repr__(self):
        return self.name
    
class Negative(Prior):
    """A prior to force x < 0.
    """
    
    __slots__ = []
    
    name = "Negative"
    
    def __init__(self):
        """Constructs a negative prior.
        """
        pass
        
    def logprob(self, x):
        """Computes the log probability given the current parameter value, x.

        Args:
            x (float): The current parameter value.
        Returns:
            float: The logarithm of the probability distribution for this value.
        """
        return 0 if x < 0 else -np.inf
        
    def __repr__(self):
        return self.name
      
class JeffreysG(Prior):
    """A Jeffrey's prior for a Gaussian likelihood which is proportional to 1 / x.

        Attributes:
            lower_bound (float): The lower bound.
            upper_bound (float): The upper bound.
        """
    
    __slots__ = ['lower_bound', 'upper_bound', 'lognorm', 'knee']
    name = "Jeffrey's"
    
    def __init__(self, lower_bound, upper_bound, knee=0):
        """Constructs a Jeffrey's prior for a Gaussian likelihood.

        Args:
            lower_bound (float): The lower bound of the prior.
            upper_bound (float): The upper bound of the prior.
            knee (int, optional): The knee of the distribution (x0) The distribution is proportional to 1 / (x - x0). Defaults to 0.
        """
        assert lower_bound <= upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.knee = knee
        self.lognorm = np.log(1.0 / np.log((self.upper_bound - self.knee) / (self.lower_bound - self.knee)))
        
    def logprob(self, x):
        """Computes the log probability given the current parameter value, x.

        Args:
            x (float): The current parameter value.
        Returns:
            float: The logarithm of the probability distribution for this value.
        """
        if self.lower_bound < x < self.upper_bound:
            return self.lognorm - np.log(x - self.knee)
        else:
            return -np.inf
        
    def __repr__(self):
        return f"{self.name}: [{self.lower_bound}, {self.knee}, {self.upper_bound}]"

