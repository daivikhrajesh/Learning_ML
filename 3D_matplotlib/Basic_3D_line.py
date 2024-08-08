from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X,Y,Z = [1,2,3,4,5,6,7,8,9,10],[5,6,3,13,4,1,2,4,8,10],[2,3,3,3,5,7,9,11,9,10]

ax.plot(X,Y,Z)

plt.show()