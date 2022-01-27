
import corner

class OptProblem:
    """A base class for optimization problems.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################

    def __init__(self, name=None, obj=None, obj_args=None, obj_kwargs=None, p0=None):
        """Base constructor.
        """
        self.name = "Optimization Problem" if name is None else name
        self.obj = obj
        self.obj_args = obj_args
        self.obj_kwargs = obj_kwargs
        self.p0 = p0

    ##################
    #### OPTIMIZE ####
    ##################

    def optimize(self, optimizer, p0=None, obj=None, obj_args=None, obj_kwargs=None, *args, **kwargs):
        if p0 is not None:
            self.p0 = p0
        self.resolve_obj(obj, obj_args, obj_kwargs)
        opt_result = optimizer.optimize(self.p0, obj=self.obj, obj_args=self.obj_args, obj_kwargs=self.obj_kwargs, *args, **kwargs)
        self.print_opt_result(opt_result)
        return opt_result

    def run_mcmc(self, sampler, p0=None, obj=None, obj_args=None, obj_kwargs=None, *args, **kwargs):
        if p0 is not None:
            self.p0 = p0
        self.resolve_obj(obj, obj_args, obj_kwargs)
        mcmc_result = sampler.run_mcmc(self.p0, obj=self.obj, obj_args=self.obj_args, obj_kwargs=self.obj_kwargs, *args, **kwargs)
        self.print_mcmc_result(mcmc_result)
        return mcmc_result
            
    ###############
    #### MISC. ####
    ###############

    def resolve_obj(self, obj=None, obj_args=None, obj_kwargs=None):
        
        # Objective
        if self.obj is None and obj is not None:
            self.obj = obj

        # Objective args
        if obj_args is not None:
            self.obj_args = obj_args
        elif self.obj_args is None and obj_args is None:
            self.obj_args = ()

        # Objective kwargs
        if obj_kwargs is not None:
            self.obj_kwargs = obj_kwargs
        elif self.obj_kwargs is None and obj_kwargs is None:
            self.obj_kwargs = {}
    
    def print_opt_result(self, opt_result):
        print(f"Status: {opt_result['status']}")
        print(f"Objective value: {opt_result['fbest']}")
        print(f"Objective calls: {opt_result['fcalls']}")
        print(opt_result['pbest'])

    def print_mcmc_result(self, mcmc_result):
        print(f"ln(L): {mcmc_result['lnL']}")
        print(f"Objective calls: {mcmc_result['n_steps']}")
        print("Median Parameters:")
        print(mcmc_result['pmed'])

    def __repr__(self):
        return self.name