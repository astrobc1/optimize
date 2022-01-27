
# Standard imports
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt

# Import optimize
import optimize as opt

# Gaussian function
def gauss(x, pars):
    return pars['amp'].value * np.exp(-0.5 * ((x - pars['mu'].value) / pars['sigma'].value)**2)

# An x grid
dx = 0.01
x = np.arange(-10, 10 + dx, dx)

# True parameters
pars_true = opt.Parameters()
pars_true["amp"] = opt.Parameter(value=2.5)
pars_true["mu"] = opt.Parameter(value=-1)
pars_true["sigma"] = opt.Parameter(value=0.8)

# Noisy data
y_true = gauss(x, pars_true)
y_errors = np.random.uniform(0.05, 0.08, size=len(y_true))
y_true += np.array([y_errors[i] * np.random.randn() for i in range(len(y_true))])

# Guess parameters and model
pars_guess = opt.Parameters()
pars_guess["amp"] = opt.Parameter(value=2.0)
pars_guess["mu"] = opt.Parameter(value=-0.4)
pars_guess["sigma"] = opt.Parameter(value=0.4)
model_guess = gauss(x, pars_guess)

# Create objective.
# The x, data, and errors attributes can be named anything but must be kwargs and used appropriatly below
# Chi2Loss doesn't expect us to compute the actual objective now, but we still must define the default methods compute_residuals and compute_data_errors.
# Alternatively, we could just override compute_obj(self, pars) and handle everything in there.
# Alternatively, alternatively, we could define a new class extending Chi2Loss and override these methods there.
obj = opt.Chi2Loss(x=x, data=y_true, errors=y_errors)

# Here self is a modified instance of Chi2Loss (obj above)
# One liner >>> obj.compute_residuals = lambda self, pars: self.data - gauss(self.x, pars)
def compute_residuals(self, pars):
    return self.data - gauss(self.x, pars)

def compute_data_errors(self, pars):
    return self.errors

obj.compute_residuals = compute_residuals
obj.compute_data_errors = compute_data_errors

# Create the Optimization Problem
optprob = opt.OptProblem(obj=obj, p0=pars_guess)

# Optimize the model
opt_result = optprob.optimize(optimizer=opt.IterativeNelderMead())

# Get best fit pars
pbest = opt_result["pbest"]

# Build the best fit model
model_best = gauss(x, pbest)

# Plot
plt.errorbar(x, y_true, yerr=y_errors, marker='o', lw=0, elinewidth=1, label="my data", c='grey', alpha=0.8, zorder=0)
plt.plot(x, model_guess, label='Starting Model', c='blue')
plt.plot(x, model_best, label='Best Fit Model', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()