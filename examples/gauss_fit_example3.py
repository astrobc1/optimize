
# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Import optimize
import optimize as opt

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Define a model for a Gaussian function.
class GaussianModel(opt.DeterministicModel):

    def build(self, pars):
        return gauss(self.data.x, pars["amp"].value, pars["mu"].value, pars["sigma"].value)

# An x grid
dx = 0.05
x = np.arange(-10, 10 + dx, dx)

# True parameters
pars_true = opt.BayesianParameters()
pars_true["amp"] = opt.BayesianParameter(value=2.5)
pars_true["mu"] = opt.BayesianParameter(value=-1)
pars_true["sigma"] = opt.BayesianParameter(value=0.8)

# Noisy data
y_true = gauss(x, pars_true["amp"].value, pars_true["mu"].value, pars_true["sigma"].value)
noise_level = 0.1
y_true += noise_level * np.random.randn(y_true.size)

# Create the opt problem
optprob = opt.BayesianProblem()

# Create a data object
data = opt.SimpleSeries(x=x, y=y_true, yerr=np.full(y_true.size, 0), label="my_data")

# Guess parameters and model
pars_guess = opt.BayesianParameters()
pars_guess["amp"] = opt.BayesianParameter(value=2.0)
pars_guess["amp"].add_prior(opt.priors.Positive())

pars_guess["mu"] = opt.BayesianParameter(value=-0.4, latex_str="$\mu$")
pars_guess["sigma"] = opt.BayesianParameter(value=0.4, latex_str="$\sigma$")
pars_guess["sigma"].add_prior(opt.priors.Positive())
pars_guess["jitter_my_data"] = opt.BayesianParameter(value=0.5, latex_str="$\mathrm{JIT}_{my-data}$")
pars_guess["jitter_my_data"].add_prior(opt.priors.Positive())
model_guess = gauss(x, pars_guess["amp"].value, pars_guess["mu"].value, pars_guess["sigma"].value)

# Make a model
model = opt.NoiseModel(det_model=GaussianModel(data=data), noise_process=opt.WhiteNoiseProcess(data=data), data=data)

# Create a Posterior obj function
post = opt.Posterior()
post["my_like"] = opt.GaussianLikelihood(model=model)

# Bayesian problem
optprob = opt.BayesianProblem(p0=pars_guess, post=post, optimizer=opt.SciPyMinimizer(method="Nelder-Mead"), sampler=opt.emceeSampler())

# Optimize the model
opt_result = optprob.optimize()
pars_best = opt_result["pbest"]

# Print the best fit pars
print(pars_best)

# Build the best fit model
model_best = model.build(pars_best)

# Plot
plt.errorbar(x, y_true, yerr=model.compute_data_errors(pars_best), marker='o', lw=0, label="my data", c='grey', alpha=0.8, elinewidth=1)
plt.plot(x, model_guess, label='Starting Model', c='blue')
plt.plot(x, model_best, label='Best Fit Model', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Run the mcmc
# The mccmc will break once converged according to the auto-correlation time
mcmc_result = optprob.run_mcmc()
fig = optprob.corner_plot(mcmc_result)

# Show the corner plot
fig.show()