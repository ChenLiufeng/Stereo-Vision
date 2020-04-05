import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy import asarray
from mpl_toolkits.mplot3d import Axes3D
from disparity import *

depth = generate_disparity_map('im0.png', 'im1.png', 'new2', 1000, 1482, 140, 11)
depth = depth * (-1) + 255
row, col = depth.shape

x = range(row)
y = range(col)

xs, ys, zs = np.meshgrid(x, y, depth)
print(xs, ys)
zs = depth(xs, ys)
print(zs)
print(zs.shape)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(xs, ys, zs, c=zs, cmap='hot')
#plt.show()

