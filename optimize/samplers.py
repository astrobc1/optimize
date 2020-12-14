# Contains the custom Nelder-Mead algorithm
import numpy as np
import copy
import optimize.knowledge
from optimize.optimizers import Sampler
from multiprocessing import Pool
import emcee
import time
import corner
import matplotlib.pyplot as plt

class AffInv(Sampler):
    
    def __init__(self, scorer=None, options=None):
        super().__init__(scorer=scorer, options=options)
        self.init_sampler()
        p0_dict = self.scorer.p0.unpack()
        self.test_pars = copy.deepcopy(self.scorer.p0)
        self.test_pars_vec = np.copy(p0_dict['value'])
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        
    def init_walkers(self, pars=None):
        
        if pars is None:
            pars = self.scorer.p0
        pars_vary_dict = pars.unpack(vary_only=True)
        n_pars = len(pars)
        n_pars_vary = pars.num_varied()
        search_scales = np.array([par.compute_crude_scale() for par in pars.values() if par.vary])
        walkers = pars_vary_dict['value'] + search_scales * np.random.randn(self.n_walkers, n_pars_vary)
        return walkers
    
    def set_pars(self, pars):
        self.p0 = pars
    
    @property
    def n_walkers(self):
        return self.sampler.nwalkers
    
    def init_sampler(self):
        n_pars_vary = self.scorer.p0.num_varied()
        n_walkers = 2 * n_pars_vary
        self.sampler = emcee.EnsembleSampler(n_walkers, n_pars_vary, self.compute_score) #, backend=emcee.backends.HDFBackend('chains.h5'))
        
    def sample(self, pars=None, walkers=None, do_burn=True, n_burn_steps=None, n_steps=None):
        
        if pars is None and walkers is None:
            walkers = self.init_walkers()
            pars = self.scorer.p0
        elif pars is not None:
            walkers = self.init_walkers(pars)
            
        if n_steps is None:
            n_steps = 10_000 * pars.num_varied()
        
        # Burn in
        if do_burn:
            
            # Number of burn in steps
            if n_burn_steps is None:
                n_burn_steps = 100 * pars.num_varied()
            
            # Run burn in
            walkers = self.sampler.run_mcmc(walkers, n_burn_steps, progress=True)
            
            print("Burn in complete ...")
            print("Current Parameters ...")
            
            flat_chains = self.sampler.flatchain
            pars_mcmc = flat_chains[np.nanargmax(self.sampler.flatlnprobability)]
            _pbest = copy.deepcopy(self.scorer.p0)
            _par_vec = np.copy(self.test_pars_vec)
            _par_vec[self.p0_vary_inds] = pars_mcmc
            _pbest.setv(value=_par_vec)
            _pbest.pretty_print()
        
        # Reset
        self.sampler.reset()
        
        # Run full search
        
        # Convergence testing
        k = 0
        autocorrs = np.full(n_steps, fill_value=np.nan)
        old_tau = np.inf

        # Sample up to n_steps
        converged = False
        for sample in self.sampler.sample(walkers, iterations=n_steps, progress=True):
            
            # Only check convergence every 500 steps
            if self.sampler.iteration % 500:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = self.sampler.get_autocorr_time(tol=0)
            autocorrs[k] = np.mean(tau)
            k += 1

            # Check convergence
            #converged = np.all(tau * 500 < self.sampler.iteration)
            converged = np.all(np.abs(old_tau - tau) / tau < 0.02)
            if converged:
                break
                
            old_tau = tau
            
        # Outputs
        sampler_result = {}
        if converged:
            print("Success!")
        sampler_result["flat_chains"] = self.sampler.flatchain
        pars_mcmc = sampler_result["flat_chains"][np.nanargmax(self.sampler.flatlnprobability)]
        sampler_result["pbest"] = copy.deepcopy(self.scorer.p0)
        par_vec = np.copy(self.test_pars_vec)
        par_vec[self.p0_vary_inds] = pars_mcmc
        sampler_result["pbest"].setv(value=par_vec)
        errors = self.compute_errors(sampler_result)
        sampler_result["errors"] = errors
        return sampler_result
    
    def corner_plot(self, sampler_result, show=True):
        plt.clf()
        pbest_vary_dict = sampler_result["pbest"].unpack(vary_only=True)
        truths = pbest_vary_dict["value"]
        labels = pbest_vary_dict["name"]
        fig = corner.corner(sampler_result["flat_chains"], labels=labels, truths=truths, show_titles=True)
        if show:
            plt.show()
        else:
            return fig
        
    def compute_errors(self, sampler_result, percentiles=None):
        errors = {}
        if percentiles is None:
            percentiles = [15.9, 50, 84.1]
        pnames_vary = sampler_result["pbest"].unpack(keys='name', vary_only=True)['name']
        for i in range(len(pnames_vary)):
            vals = np.percentile(sampler_result["flat_chains"][:, i], percentiles)
            errors[pnames_vary[i]] = np.diff(vals)
        return errors
        
    def compute_score(self, pars):
        ti = time.time()
        self.test_pars_vec[self.p0_vary_inds] = pars
        self.test_pars.setv(value=self.test_pars_vec)
        _score = self.scorer.compute_score(self.test_pars)
        tf = time.time()
        return -1 * _score
    
    
class MultiNest(Sampler):
    pass

class HNUT(Sampler):
    pass