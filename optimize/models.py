
# Maths
import numpy as np

# Optimize deps
from optimize.noise import CorrelatedNoiseProcess


####################
#### BASE TYPES ####
####################

class Model:
    
    def __init__(self, data=None, name=None):
        self.data = data
        self.name = name
    
    def __repr__(self):
        return f"Model: {self.name}"
    
    def build(self, pars):
        raise NotImplementedError(f"Must implement a build method for the class {self.__class__.__name__}")
    
    def initialize(self, p0):
        self.p0 = p0
        
    def compute_residuals(self, pars):
        data_arr = self.data.get_trainable()
        model_arr = self.build(pars)
        return data_arr - model_arr
    
    def compute_data_errors(self, pars):
        return self.data.get_apriori_errors()

class DeterministicModel(Model):
    
    def __call__(self, pars):
        return self.build(pars)
    
    def __repr__(self):
        return f"Deterministic model: {self.name}"
 
class NoiseBasedModel(Model):
    
    def __init__(self, det_model=None, noise_process=None, data=None, name=None):
        super().__init__(data=data, name=name)
        self.det_model = det_model
        self.noise_process = noise_process
    
    def initialize(self, p0):
        super().initialize(p0)
        self.det_model.initialize(self.p0)
        self.noise_process.initialize(self.p0)
    
    def build(self, pars):
        if self.det_model is not None:
            return self.det_model.build(pars)
        else:
            return 0
    
    def compute_raw_residuals(self, pars):
        if self.det_model is not None:
            return self.data.get_trainable() - self.build(pars)
        else:
            return self.data.get_trainable()
        
    def compute_residuals(self, pars):
        residuals = self.compute_raw_residuals(pars)
        if isinstance(self.noise_process, CorrelatedNoiseProcess):
            residuals -= self.realize(pars, linpred=residuals)
        else:
            return residuals
    
    def compute_data_errors(self, pars, include_corr_error=False):
        if isinstance(self.noise_process, CorrelatedNoiseProcess):
            if include_corr_error:
                linpred = self.compute_raw_residuals(pars)
            else:
                linpred = None
            return self.noise_process.compute_data_errors(pars, include_corr_error=include_corr_error, linpred=linpred)
        else:
            return self.noise_process.compute_data_errors(pars)
        
    def __repr__(self):
        return f"Noise model: {self.name}"

class GPBasedModel(NoiseBasedModel):

    def __repr__(self):
        return f"GP Based Model: {self.name}"