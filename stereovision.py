import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy import asarray

imgL = cv2.imread('im0.png',0)
imgR = cv2.imread('im1.png',0)

left = asarray(imgL, dtype='int64')
right = asarray(imgR, dtype='int64')

row_size, col_size = left.shape

block_size = 11
offset = block_size // 2

disparity_matrix = np.ndarray(shape=(row_size - block_size + 1, col_size - block_size + 1), dtype=np.float32)
disparity_matrix[:, :] = 0
maxdisp = 280

for x in range(offset, row_size-offset):
    print(x)
    for y in range(offset, col_size-offset):
        lefttest = left[x-offset:x+offset+1, y-offset:y+offset+1]
        array = []
        for z in range(offset, maxdisp-offset):
            righttest = right[z-offset:z+offset+1, y-offset:y+offset+1]
            array += [np.sum(abs(lefttest-righttest))]
        disparity_matrix[x-offset, y-offset] = max(array)

'''
i_1 = 0
i_2 = 0
i_3 = 0
i_4 = 0

image = left

for x in range(offset, row_size-offset):

    for y in range(offset, col_size-offset):

        i_1 += (image[x][y] - image[x][y+1])**2
        i_2 += (image[x][y] - image[x+1][y])**2
        i_3 += (image[x][y] - image[x+1][y+1])**2
        i_4 += (image[x][y] - image[x+1][y-1])**2    

        disparity_matrix[x-offset, y-offset] = min(i_1, i_2, i_3, i_4)


plt.plot(disparity_matrix)
plt.show()
'''