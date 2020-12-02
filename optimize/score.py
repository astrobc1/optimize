import optimize.knowledge
import optimize.kernels as optnoisekernels
import numpy as np


class ScoreFunction:
    """An abstract class for a general score function.
    """
    
    def __init__(self, data, model):
        """Stores the basic requirements for a score function.

        Args:
            data (Data): A dataset inheriting from optimize.data.Data.
            model (Model): A model inheriting from optimize.models.Model.
        """
        self.data = data
        self.model = model

    def compute_score(self, pars):
        """Computes the score from a given set of parameters.

        Args:
            pars (Parameters): The parameters to use.

        Raises:
            NotImplementedError: Must implement this method.
        """
        raise NotImplementedError("Must implement a compute_score method.")
        
class MSEScore(ScoreFunction):
    """A class for the standard mean squared error (MSE) loss.
    """
    
    def __init__(self, data, model):
        super().__init__(data, model)
    
    def compute_score(self, pars):
        """Computes the mean squared error loss

        Args:
            pars (Parameters): The parameters object to use.

        Returns:
            float: The score.
        """
        _model = self.model.build(pars)
        _data = self.data.y
        good = np.where(np.isfinite(_data) & np.isfinite(_model))[0]
        rms = self.compute_rms(_data[good], _model[good])
        return rms
    
    @staticmethod
    def compute_rms(_data, _model):
        """Computes the RMS (Root mean squared) loss

        Args_data 
            _data (np.ndarray): The data.
            _model (np.ndarray): The model.

        Returns:
            float: The RMS.
        """
        return np.sqrt(np.nansum((_data - _model)**2) / _data.size)
    
    @staticmethod
    def compute_chi2(res, errorbars):
        """Computes the (non-reduced) chi2 statistic (weighted MSE).

        Args:
            res (np.ndarray): The residuals (data - model)
            errorbars ([type]): The effective errorbars (intrinsic and any white noise).

        Returns:
            float: The chi-squared statistic.
        """
        return np.nansum((res / errorbars)**2)
    
    @staticmethod
    def compute_redchi2(res, errorbars, ndeg=None):
        """Computes the reduced chi2 statistic (weighted MSE).

        Args:
            res (np.ndarray): The residuals (data - model)
            errorbars ([type]): The effective errorbars (intrinsic and any white noise).
            ndeg (int): The degrees of freedom, defaults to len(res) - 1.

        Returns:
            float: The reduced chi-squared statistic.
        """
        if ndeg is None:
            ndeg = res.size - 1
        _chi2 = np.nansum((res / errorbars)**2)
        return _chi2 / ndeg

class MLEScore(ScoreFunction):
    
    def __init__(self, model, data):
        super().__init__(model, data)
    
    def compute_score(self, pars):
        """Computes the negative likelihood score.

        Args_data 
            _data (np.ndarray): The data.
            _model (np.ndarray): The model.

        Returns:
            float: -ln(Likelihood).
        """
        neglnL = self.compute_negloglikelihood(pars)
        return neglnL
    
    def compute_loglikelihood(self, pars):
        _model = self.model.build(pars)
        _data = self.data.y
        good = np.where(np.isfinite(_data) & np.isfinite(_model))[0]
        N = good.size
        _res = _data[good] - _model[good]
        if isinstance(self.model.kernel, optnoisekernels.GaussianProcess):
            K = self.model.kernel.compute_cov_matrix(pars, self.errorbars()[good])
            _lnL = np.dot(np.dot(_res, np.linalg.inv(K)), _res.reshape(1, N)) - 0.5 * np.log(np.linalg.det(K)) - 0.5 * N * np.log(2 * np.pi)
        else:
            _redchi2 = self.compute_redchi2(_res, self.data.compute_errorbars(pars))
            _lnL = -1 * _redchi2 / 2
        return _lnL
    
    
    def compute_negloglikelihood(self, pars):
         return -1 * self.compute_loglikelihood(pars)