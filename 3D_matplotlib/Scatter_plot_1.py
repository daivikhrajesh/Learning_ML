from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X=[1,2,3,4,5,6,7,8,9,10]
Y=[5,2,3,4,5,1,2,12,2,8]
Z=[2,3,4,2,1,2,9,7,2,3]

ax.scatter(X,Y,Z, c='r', marker='o')

ax.set_xlabel('x_axis')
ax.set_ylabel('y_axis')
ax.set_zlabel('z_axis')

plt.show()