
# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Optimize imports
import optimize.knowledge as optknow
import optimize.models as optmodels
import optimize.optimizers as optimizers
import optimize.scores as optscores
import optimize.data as optdatasets
import optimize.frameworks as optframeworks

# Define a build method for the model
# By default, models are called via builder(pars, *args, **kwargs)
# where pars is an instance of optimize.Parameters
# The score (target / obejctive) function comes later
def gauss(pars, x):
    return pars["amp"].value * np.exp(-0.5 * ((x - pars["mu"].value) / pars["sigma"].value)**2)

# Create some noisy data
dx = 0.01
x = np.arange(-10, 10 + dx, dx)
pars_true = optknow.Parameters()
pars_true["amp"] = optknow.Parameter(value=2.5)
pars_true["mu"] = optknow.Parameter(value=-1)
pars_true["sigma"] = optknow.Parameter(value=0.8)
y_true = gauss(pars_true, x)
y_true += 0.01 * np.random.randn(y_true.size)

# Guess parameters and model
pars_guess = optknow.Parameters()
pars_guess["amp"] = optknow.Parameter(value=2.0)
pars_guess["mu"] = optknow.Parameter(value=-0.4)
pars_guess["sigma"] = optknow.Parameter(value=0.4)
model_guess = gauss(pars_guess, x)

# Create dedicated data and model objects
data = optdatasets.Data(label="data", x=x, y=y_true)
model = optmodels.Model(builder=gauss, p0=pars_guess, args_to_pass=(x,))

# Create a mean squared error score function
scorer = optscores.MSE(data=data, model=model)

# Create an optimizer
optimizer = optimizers.NelderMead(scorer=scorer)

# Create an optimize problem.
optprob = optframeworks.OptProblem(data=data, model=model, p0=pars_guess, optimizer=optimizer)

# Optimize the model via Nelder Mead
opt_result = optprob.optimize()
pars_fit = opt_result["pbest"]
pars_fit.pretty_print()

# Build the best fit model
model_best = gauss(pars_fit, x)

# Plot
plt.plot(x, y_true, marker='o', lw=0, label=data.label, c='grey', alpha=0.8)
plt.plot(x, model_guess, label='Starting Model', c='blue')
plt.plot(x, model_best, label='Best Fit Model', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()