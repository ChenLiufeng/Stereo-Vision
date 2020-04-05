import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    disparity = Disparity('im0.png', 'im1.png', 1000, 1482, 11, 140)
    PlotDepth2D(disparity, 'finald')
    return True

def ReadImage(path, xsize, ysize):
    image = cv2.imread(path, 0)
    image = cv2.resize(image, dsize=(xsize, ysize), interpolation=cv2.INTER_CUBIC)
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

def Disparity(imleft, imright, xsize, ysize, wsize, crange):
    left = ReadImage(imleft, ysize, xsize)
    right = ReadImage(imright, ysize, xsize)
    rows, cols = left.shape
    matrices = np.zeros(shape=(crange, rows, cols), dtype=float)
    final = np.zeros(shape=(crange, rows - wsize+1, cols - wsize+1), dtype=float)
    for n in range(0, crange, 1):
        matrices[n] = np.abs(right - LeftShift(left, n))
    for i in range(0, len(matrices), 1):
        final[i] = SADValues(matrices[i], wsize)
    disparity = np.argmin(final, 0).astype(np.uint8)
    return disparity

def PlotDepth2D(disparity, name):
    depth =  (193.001/2) * (3979.911/2) / (disparity + (124.343/2) )
    colourmap = depth - np.amin(depth)
    colourmap = colourmap * (255 / np.amax(colourmap))
    plt.imshow(colourmap, cmap='hsv', interpolation='nearest')
    plt.colorbar()
    plt.show()
    cv2.imwrite(name+'.png', colourmap)
    return True

main()