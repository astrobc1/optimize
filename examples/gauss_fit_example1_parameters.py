
# Standard imports
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt

# Import optimize
import optimize as opt

# Gaussian function
def gauss(x, pars):
    return pars['amp'].value * np.exp(-0.5 * ((x - pars['mu'].value) / pars['sigma'].value)**2)

# Define an objective for use with the Parameters object
def compute_obj(pars, x, data):
    model = gauss(x, pars)
    residuals = data - model
    rms = opt.RMSLoss.rmsloss(residuals)
    return rms

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
y_true += 0.01 * np.random.randn(len(y_true))

# Guess parameters and model
pars_guess = opt.Parameters()
pars_guess["amp"] = opt.Parameter(value=2.0)
pars_guess["mu"] = opt.Parameter(value=-0.4)
pars_guess["sigma"] = opt.Parameter(value=0.4)
model_guess = gauss(x, pars_guess)

# Create the optimizer
optimizer = opt.IterativeNelderMead()

# Optimize the model
opt_result = optimizer.optimize(p0=pars_guess, obj=compute_obj, obj_args=(x, y_true))

# Get best fit pars
pbest = opt_result["pbest"]

# Build the best fit model
model_best = gauss(x, pbest)

# Plot
plt.plot(x, y_true, marker='o', lw=0, label="my data", c='grey', alpha=0.8)
plt.plot(x, model_guess, label='Starting Model', c='blue')
plt.plot(x, model_best, label='Best Fit Model', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()