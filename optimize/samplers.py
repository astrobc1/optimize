# Contains the custom Nelder-Mead algorithm
import numpy as np
import copy
import optimize.knowledge
from optimize.optimizers import Sampler
import tqdm
from joblib import Parallel, delayed
import emcee
import time
import corner
import matplotlib.pyplot as plt
import zeus

class emceeLikeSampler(Sampler):
    
    def __init__(self, obj=None, options=None):
        """Default constructor for emcee like samplers emcee and Zeus.

        Args:
            obj (CompositeLikelihood): The composite likelihood object.
            options (dict): A dictionary containing any emcee.EnsembleSampler kwargs or Zeus kwargs. Defaults to None.
        """
        super().__init__(obj=obj, options=options)
        p0_dict = self.obj.p0.unpack()
        self.test_pars = copy.deepcopy(self.obj.p0)
        self.test_pars_vec = np.copy(p0_dict['value'])
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        self.init_sampler()
        
    def init_walkers(self, pars=None):
        """Initializes a set of walkers.

        Args:
            pars (Parameters, optional): The starting parameters. Defaults to p0.

        Returns:
            np.ndarray: A set of walkers as a numpy array with shape=(n_walkers, n_parameters_vary)
        """
        
        if pars is None:
            pars = self.obj.p0
        pars_vary_dict = pars.unpack(vary_only=True)
        n_pars = len(pars)
        n_pars_vary = pars.num_varied()
        search_scales = np.array([par.compute_crude_scale() for par in pars.values() if par.vary])
        walkers = pars_vary_dict['value'] + search_scales * np.random.randn(self.n_walkers, n_pars_vary)
        return walkers

    def compute_obj(self, pars):
        """Wrapper to compute the objective.

        Args:
            pars (np.ndarray): The parameter values for only the varied parameters.

        Returns:
            float: The log(likelihood)
        """
        self.test_pars_vec[self.p0_vary_inds] = pars
        self.test_pars.setv(value=self.test_pars_vec)
        lnL = self.obj.compute_logaprob(self.test_pars)
        return lnL

    @property
    def n_walkers(self):
        """Alias for the number of walkers.

        Returns:
            int: The number of walkers.
        """
        return self.sampler.nwalkers

    @staticmethod
    def chain_uncertainty(flat_chain, percentiles=[15.9, 50, 84.1]):
        
        # Compute percentiles
        par_quantiles = np.percentile(flat_chain, percentiles)
        pmed = par_quantiles[1]
        unc = np.diff(par_quantiles)
        out = (pmed, unc[0], unc[1])
        
        return out

    @staticmethod
    def corner_plot(mcmc_result):
        """Constructs a corner plot.

        Args:
            mcmc_result (dict): The mcmc result.
            
        Returns:
            fig: A matplotlib figure.
        """
        pbest_vary_dict = mcmc_result["pmed"].unpack(vary_only=True)
        truths = pbest_vary_dict["value"]
        labels = [par.latex_str for par in mcmc_result["pbest"].values() if par.vary]
        corner_plot = corner.corner(mcmc_result["chains"], labels=labels, truths=truths, show_titles=True)
        return corner_plot

    
class emceeSampler(emceeLikeSampler):
    """An class to interface to the emcee affine invariant Ensemble Sampler.
    """
    
    def init_sampler(self):
        """Initializes the emcee.Ensemble sampler.
        """
        p0_dict = self.obj.p0.unpack()
        self.test_pars = copy.deepcopy(self.obj.p0)
        self.test_pars_vec = np.copy(p0_dict['value'])
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        n_pars_vary = self.obj.p0.num_varied()
        n_walkers = 2 * n_pars_vary
        self.sampler = emcee.EnsembleSampler(n_walkers, n_pars_vary, self.compute_obj)
        
    def sample(self, pars=None, walkers=None, n_burn_steps=500, check_every=200, n_steps=75_000, rel_tau_thresh=0.01, n_min_steps=1000, n_cores=1, n_taus_thresh=40, progress=True):
        """Wrapper to perform a burn-in + full MCMC exploration.

        Args:
            pars (Parameters, optional): The starting parameters. Defaults to p0.
            walkers (np.ndarray, optional): The starting walkers. Defaults to calling self.init_walkers(pars).
            n_burn_steps (int, optional): The number of burn in steps. Defaults to 100 * pars.num_varied().
            n_steps (int, optional): The number of mcmc steps to perform in the full phase (post burn-in). Defaults to 50_000 * pars.num_varied().
            rel_tau_thresh (float, optional): The relative change in the auto-correlation time for convergence. This criterion must be met for all walkers. Defaults to 0.01.
            n_min_steps (int, optional): The minimum number of steps to run. Defaults to 1000
            n_threads (int, optional): The number of threads to use. Defaults to 1.

        Returns:
            dict: The sampler result, with keys: flat_chains, autocorrs, steps, pbest, errors, lnL.
        """
        
        self.n_cores = n_cores
        
        # Init pars
        if pars is None and walkers is None:
            walkers = self.init_walkers()
            pars = self.obj.p0
        elif pars is not None:
            walkers = self.init_walkers(pars)

        # Burn in
        if n_burn_steps > 0:
            
            # Run burn in
            print("Running Burn-in MCMC Phase [" + str(n_burn_steps) + "]")
            walkers = self.sampler.run_mcmc(walkers, n_burn_steps, progress=True)
            
            print("Burn in complete ...")
            print("Current Parameters ...")
            
            flat_chains = self.sampler.flatchain
            pars_best = flat_chains[np.nanargmax(self.sampler.flatlnprobability)]
            _pbest = copy.deepcopy(self.obj.p0)
            _par_vec = np.copy(self.test_pars_vec)
            _par_vec[self.p0_vary_inds] = pars_best
            _pbest.setv(value=_par_vec)
            _pbest.pretty_print()
    
            # Reset
            self.sampler.reset()
    
        # Run full search
    
        # Convergence testing
        autocorrs = []
        old_tau = np.inf

        # Sample up to n_steps
        converged = False
        print("Running Full MCMC Phase ...")
        _trange = tqdm.trange(n_steps, desc="Running Min Steps = " + str(n_min_steps), leave=True)
        for _, sample in zip(_trange, self.sampler.sample(walkers, iterations=n_steps, progress=False)):
            
            # Only check convergence every 200 steps and run at least a minimum number of steps
            if self.sampler.iteration % check_every:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            taus = self.sampler.get_autocorr_time(tol=0)
            med_tau = np.nanmedian(taus)
            autocorrs.append(med_tau)

            # Check convergence:
            # 1. Ensure we've run a sufficient number of autocorrelation time scales
            # 2. Ensure the estimations of the autorr times themselves are settling.
            converged = med_tau * n_taus_thresh < self.sampler.iteration
            rel_tau = np.abs(old_tau - med_tau) / med_tau
            converged &= rel_tau < rel_tau_thresh
            converged &= self.sampler.iteration > n_min_steps
            _trange.set_description("\u03C4 = " + str(round(med_tau, 5)) + " / x" + str(n_taus_thresh) + ", rel change = " + str(round(rel_tau, 5)) + " / " + str(round(rel_tau_thresh, 5)))
            if converged:
                print("Success!")
                break
            old_tau = med_tau
            
        # Output
        mcmc_result = {}
        
        # Add steps and autocorrs
        mcmc_result["n_steps"] = self.sampler.iteration
        mcmc_result["autocorrs"] = autocorrs
        
        # Get the flat chains and lnLs
        mcmc_result["chains"] = self.sampler.flatchain
        mcmc_result["lnLs"] = self.sampler.lnprobability
        mcmc_result["acc"] = self.sampler.acceptance_fraction
        
        # Best parameters from best sampled like (sampling must be dense enough, probably is)
        pars_best = self.sampler.chain[1, np.argmax(self.sampler.lnprobability[1, :]), :]
        pbest = copy.deepcopy(self.obj.p0)
        par_vec = np.copy(self.test_pars_vec)
        par_vec[self.p0_vary_inds] = pars_best
        pbest.setv(value=par_vec)
        mcmc_result["pbest"] = pbest
        mcmc_result["lnL"] = self.obj.compute_logaprob(mcmc_result["pbest"])
        
        # Parameter uncertainties
        pnames_vary = mcmc_result["pbest"].unpack(keys='name', vary_only=True)['name']
        pmed = copy.deepcopy(self.obj.p0)
        for i, pname in enumerate(pnames_vary):
            _pmed, unc_lower, unc_upper = self.chain_uncertainty(mcmc_result["chains"][:, i])
            pmed[pname].value = _pmed
            pmed[pname].unc = (unc_lower, unc_upper)
        
        # Add pmed
        mcmc_result["pmed"] = pmed
        
        return mcmc_result
    
    
class ZeusSampler(emceeLikeSampler):
    """A class to interface with the zeus sliced Ensemble Sampler.
    """
    
    def init_sampler(self):
        """Initializes the zues Ensemble sampler.
        """
        p0_dict = self.obj.p0.unpack()
        self.test_pars = copy.deepcopy(self.obj.p0)
        self.test_pars_vec = np.copy(p0_dict['value'])
        self.p0_vary_inds = np.where(p0_dict["vary"])[0]
        n_pars_vary = self.obj.p0.num_varied()
        n_walkers = 2 * n_pars_vary
        self.sampler = zeus.EnsembleSampler(n_walkers, n_pars_vary, self.compute_obj)
        
    def compute_obj(self, pars):
        """Wrapper to compute the objective.

        Args:
            pars (np.ndarray): The parameter values for only the varied parameters.

        Returns:
            float: The log(likelihood)
        """
        self.test_pars_vec[self.p0_vary_inds] = pars
        self.test_pars.setv(value=self.test_pars_vec)
        lnL = self.obj.compute_logaprob(self.test_pars)
        return lnL
        
    def sample(self, pars=None, walkers=None, n_burn_steps=500, check_every=200, n_steps=75_000, rel_tau_thresh=0.01, n_min_steps=1000, n_cores=1, n_taus_thresh=40, progress=True):
        """Wrapper to perform a burn-in + full MCMC exploration.

        Args:
            pars (Parameters, optional): The starting parameters. Defaults to p0.
            walkers (np.ndarray, optional): The starting walkers. Defaults to calling self.init_walkers(pars).
            n_burn_steps (int, optional): The number of burn in steps. Defaults to 100 * pars.num_varied().
            n_steps (int, optional): The number of mcmc steps to perform in the full phase (post burn-in). Defaults to 50_000 * pars.num_varied().
            rel_tau_thresh (float, optional): The relative change in the auto-correlation time for convergence. This criterion must be met for all walkers. Defaults to 0.01.
            n_min_steps (int, optional): The minimum number of steps to run. Defaults to 1000
            n_threads (int, optional): The number of threads to use. Defaults to 1.

        Returns:
            dict: The sampler result, with keys: flat_chains, autocorrs, steps, pbest, errors, lnL.
        """
        
        self.n_cores = n_cores
        
        # Init pars
        if pars is None and walkers is None:
            walkers = self.init_walkers()
            pars = self.obj.p0
        elif pars is not None:
            walkers = self.init_walkers(pars)

        # Burn in
        if n_burn_steps > 0:
            
            # Run burn in
            print("Running Burn-in MCMC Phase [" + str(n_burn_steps) + "]")
            walkers = self.sampler.run_mcmc(walkers, n_burn_steps, progress=True)
            
            print("Burn in complete ...")
            print("Current Parameters ...")
            
            flat_chains = self.sampler.get_chain(flat=True)
            flat_lnprob = self.sampler.get_log_prob(flat=True)
            pars_best = flat_chains[np.nanargmax(flat_lnprob)]
            _pbest = copy.deepcopy(self.obj.p0)
            _par_vec = np.copy(self.test_pars_vec)
            _par_vec[self.p0_vary_inds] = pars_best
            _pbest.setv(value=_par_vec)
            _pbest.pretty_print()
    
            # Reset
            self.sampler.reset()
    
        # Run full search
    
        # Convergence testing
        autocorrs = []
        old_tau = np.inf

        # Sample up to n_steps
        converged = False
        print("Running Full MCMC Phase ...")
        _trange = tqdm.trange(n_steps, desc="Running Min Steps = " + str(n_min_steps), leave=True)
        for _, sample in zip(_trange, self.sampler.sample(walkers, iterations=n_steps, progress=False)):
            
            # Only check convergence every 200 steps and run at least a minimum number of steps
            if self.sampler.iteration % check_every:
                continue
            
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            taus = self.get_auto_corr_times()
            med_tau = np.nanmedian(taus)
            autocorrs.append(med_tau)

            # Check convergence:
            # 1. Ensure we've run a sufficient number of autocorrelation time scales
            # 2. Ensure the estimations of the autorr times themselves are settling.
            converged = med_tau * n_taus_thresh < self.sampler.iteration
            rel_tau = np.abs(old_tau - med_tau) / med_tau
            converged &= rel_tau < rel_tau_thresh
            converged &= self.sampler.iteration > n_min_steps
            _trange.set_description("\u03C4 = " + str(round(med_tau, 5)) + " / x" + str(n_taus_thresh) + ", rel change = " + str(round(rel_tau, 5)) + " / " + str(round(rel_tau_thresh, 5)))
            if converged:
                print("Success!")
                break
            old_tau = med_tau
            
        # Output
        mcmc_result = {}
        
        # Add steps and autocorrs
        mcmc_result["n_steps"] = self.sampler.iteration
        mcmc_result["autocorrs"] = autocorrs
        
        # Get the flat chains and lnLs
        mcmc_result["chains"] = self.get_parameter_chains(flat=True)
        mcmc_result["lnLs"] = self.get_logprob_chains(flat=True)
        mcmc_result["acc"] = self.sampler.efficiency
        
        # Best parameters from best sampled like (sampling must be dense enough, probably is)
        flat_chains = self.get_parameter_chains(flat=True)
        pars_best = flat_chains[np.nanargmax(mcmc_result["lnLs"]), :]
        pbest = copy.deepcopy(self.obj.p0)
        par_vec = np.copy(self.test_pars_vec)
        par_vec[self.p0_vary_inds] = pars_best
        pbest.setv(value=par_vec)
        mcmc_result["pbest"] = pbest
        mcmc_result["lnL"] = self.obj.compute_logaprob(mcmc_result["pbest"])
        
        # Parameter uncertainties
        pnames_vary = mcmc_result["pbest"].unpack(keys='name', vary_only=True)['name']
        pmed = copy.deepcopy(self.obj.p0)
        for i, pname in enumerate(pnames_vary):
            _pmed, unc_lower, unc_upper = self.chain_uncertainty(mcmc_result["chains"][:, i])
            pmed[pname].value = _pmed
            pmed[pname].unc = (unc_lower, unc_upper)
        
        # Add pmed
        mcmc_result["pmed"] = pmed
        
        return mcmc_result
        
    def get_auto_corr_times(self):
        chain = self.sampler.get_chain(flat=False)
        ii = self.sampler.iteration
        ac_all = zeus.AutoCorrTime(chain[0:ii, :, :])
        return ac_all
    
    def get_parameter_chains(self, flat=False):
        n_steps = self.sampler.iteration
        chains_all = self.sampler.get_chain(flat=False)
        n_pars = chains_all.shape[2]
        n_walkers = self.n_walkers
        chains = chains_all[0:n_steps, :, :]
        if flat:
            chains = chains.reshape((n_steps * n_walkers, n_pars))
        return chains
    
    def get_logprob_chains(self, flat=False):
        n_steps = self.sampler.iteration
        logprob_chains_all = self.sampler.get_log_prob(flat=False)
        n_walkers = self.n_walkers
        logprob_chains = logprob_chains_all[0:n_steps, :]
        if flat:
            logprob_chains = logprob_chains.reshape((n_steps * n_walkers))
        return logprob_chains
        
    
    
# NOT IMPLEMENTED YET
# Pymultinest
class MultiNest(Sampler):
    pass

# NOT IMPLEMENTED YET
# (Hamiltonian No U-Turn)
class HNUT(Sampler):
    pass