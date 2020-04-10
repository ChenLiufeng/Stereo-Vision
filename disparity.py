import cv2
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
from pfmconversion import *

def main():
    disparity = Disparity('im0_left.png', 'im1_right.png', 'disparitynonoise', 1000, 1482, 11, 140)
    disparity = Denoising(disparity, 15, 'mean')
    disparity = Denoising(disparity, 15, 'mode')
    disparity = np.float32(disparity)
    save_pfm('disp0.pfm', disparity, scale = 1)
    PlotDepth2D(disparity, 'depthd')
    return True

def ReadImage(path, xsize, ysize):
    image = cv2.imread(path, 0)
    image = cv2.resize(image, dsize=(xsize, ysize), interpolation=cv2.INTER_CUBIC)
    image = np.float32(image)
    return image

def LeftShift(matrix, n):
    shifted = np.zeros(matrix.shape)
    rows, cols = matrix.shape
    shifted[:, 0:cols-n] = matrix[:, n:cols]
    return shifted

def SADValues(matrix, n):
    horz = np.cumsum(matrix, axis=1, dtype=float)
    horz[:, n:] = horz[:, n:] - horz[:, :-n]
    horz = horz[:, n-1:]
    vert = np.cumsum(horz, axis=0, dtype=float)
    vert[n:, :] = vert[n:, :] - vert[:-n, :]
    ret = vert[n-1:, :]
    return ret

def Disparity(imleft, imright, name, xsize, ysize, wsize, crange):
    left = ReadImage(imleft, ysize, xsize)
    right = ReadImage(imright, ysize, xsize)
    rows, cols = left.shape
    matrices = np.zeros(shape=(crange, rows, cols), dtype=float)
    final = np.zeros(shape=(crange, rows - wsize+1, cols - wsize+1), dtype=float)
    for n in range(0, crange):
        print(n)
        matrices[n] = np.abs(right - LeftShift(left, n))
    for i in range(0, len(matrices), 1):
        print(i)
        final[i] = SADValues(matrices[i], wsize)
    disparity = np.argmin(final, 0).astype(np.uint8)
    disparity = np.float32(disparity)
    return disparity

def Denoising(disparity, wsize, stat):
    disparity = disparity - np.amin(disparity)
    disparity = disparity * (255 / np.amax(disparity))
    offset = wsize // 2
    rows, cols = disparity.shape
    for row in range(offset, rows-offset):
        for col in range(offset, cols-offset):
            window = disparity[row-offset:row+offset+1, col-offset:col+offset+1]
            mean = np.mean(window)
            mode = np.median(stats.mode(window)[0][0])
            std = np.std(window)
            print(std)
            if stat == 'mean':
                stat = mean
            else: 
                stat = mode
            if disparity[row, col] > mean+1.5*std or disparity[row, col] < mean-1.5*std :
                disparity[row:row+2, col:col+3] = stat
    return disparity

def PlotDepth2D(disparity, name):
    depth =  (193.001/4) * (3979.911/4) / (disparity + (124.343/4) )
    colourmap = depth - np.amin(depth)
    colourmap = colourmap * (350 / np.amax(colourmap))
    colourmap = colourmap + 1000
    plt.imshow(colourmap, cmap='hsv', interpolation='nearest')
    plt.colorbar()
    plt.show()
    cv2.imwrite(name+'.png', colourmap)
    return True

main()