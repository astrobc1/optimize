
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
dx = 0.1
x = np.arange(-10, 10 + dx, dx)

# True parameters
pars_true = opt.Parameters()
pars_true["amp"] = opt.Parameter(value=2.5)
pars_true["mu"] = opt.Parameter(value=-1)
pars_true["sigma"] = opt.Parameter(value=0.8)

# Noisy data
y_true = gauss(x, pars_true["amp"].value, pars_true["mu"].value, pars_true["sigma"].value)
noise_level = 0.1
y_true += noise_level * np.random.randn(y_true.size)

# Create the opt problem
optprob = opt.OptProblem()

# Create a data object
data = opt.SimpleSeries(x=x, y=y_true, yerr=np.full(y_true.size, noise_level), label="My data")

# Guess parameters and model
pars_guess = opt.Parameters()
pars_guess["amp"] = opt.Parameter(value=2.0)
pars_guess["mu"] = opt.Parameter(value=-0.4)
pars_guess["sigma"] = opt.Parameter(value=0.4)
model_guess = gauss(x, pars_guess["amp"].value, pars_guess["mu"].value, pars_guess["sigma"].value)

# Make a model
model = GaussianModel(data=data)

# Create a Chi2 score function
obj = opt.Chi2(model=model)

optprob = opt.OptProblem(p0=pars_guess, obj=obj, optimizer=IterativeNelderMead())

# Optimize the model via Nelder Mead
opt_result = optprob.optimize()
pars_best = opt_result["pbest"]

# Print the best fit pars
print(pars_best)

# Build the best fit model
model_best = model(pars_best)

# Plot
plt.errorbar(x, y_true, yerr=noise_level, marker='o', lw=0, label="my data", c='grey', alpha=0.8, elinewidth=1)
plt.plot(x, model_guess, label='Starting Model', c='blue')
plt.plot(x, model_best, label='Best Fit Model', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()