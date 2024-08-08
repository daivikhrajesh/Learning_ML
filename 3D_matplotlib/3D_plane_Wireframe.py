import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate sample data
x = np.linspace(-1, 5, 50)
y = np.linspace(-5, 2, 50)
x, y = np.meshgrid(x, y)
z = np.tan(np.sqrt(x**2 + y**2))

# Plot wireframe
ax.plot_wireframe(x, y, z, rstride=1, cstride=1)

plt.show()