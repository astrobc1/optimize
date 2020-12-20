import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("gadfly_stylesheet")
import numpy as np

b = 8;
f = lambda x,y: (1-x)**2 + b*(y-x**2)**2

# Initialize figure 
import matplotlib
figRos = plt.figure(figsize=(12, 7))
axRos = figRos.gca(projection='3d')

# Evaluate function
X = np.arange(-2, 1.5, 0.001)
Y = np.arange(-1, 2.5, 0.001)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)

# Plot the surface
surf = axRos.plot_surface(X, Y, Z, cmap=cm.Spectral,
                       linewidth=0, antialiased=False, norm=matplotlib.colors.LogNorm())
#axRos.set_zlim(0, 200)
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-1, 0, 1, 2, 3])
ax.set_zticks([0, 50, 100, 150, 200, 250])
#plt.scatter(1, 1, 0, marker='X', c='black')
plt.savefig('myfig.png',  dpi=300)
plt.show()