import cv2
import numpy as np
import matplotlib.pyplot as plt

def left(matrix, n):
    out = np.zeros(matrix.shape)
    rows, cols = matrix.shape
    out[:, 0:cols-n] = matrix[:, n:cols]
    return out

def SAD(matrix, n):
    horz = np.cumsum(matrix, axis=1, dtype=float)
    horz[:, n:] = horz[:, n:] - horz[:, :-n]
    horz = horz[:, n-1:]
    vert = np.cumsum(horz, axis=0, dtype=float)
    vert[n:, :] = vert[n:, :] - vert[:-n, :]
    ret = vert[n-1:, :]
    return ret

def generate_disparity_map(left_path, right_path, name, xsize, ysize, cmp_range, block_size):

    gray_left = cv2.imread(left_path, 0)
    gray_left = cv2.resize(gray_left, dsize=(ysize, xsize), interpolation=cv2.INTER_CUBIC)

    gray_right = cv2.imread(right_path, 0)
    gray_right = cv2.resize(gray_right, dsize=(ysize, xsize), interpolation=cv2.INTER_CUBIC)

    row_size, col_size = gray_right.shape

    matrices = np.zeros(shape=(cmp_range, row_size, col_size), dtype=float)
    final = np.zeros(shape=(cmp_range, row_size - block_size+1, col_size - block_size+1), dtype=float)

    for i in range(0, cmp_range):
        matrices[i] = np.abs(gray_right - left(gray_left, i))

    for i in range(len(matrices)):
        final[i] = SAD(matrices[i], block_size)

    disparity_matrix = np.argmin(final, 0).astype(np.uint8)
    
    #cv2.imwrite(str(name) + '.png', disparity_matrix)

    row, col = disparity_matrix.shape

    # To find depth coordinate --> Z = baseline * ___
    # We then map the pixels to the range [0, 255]

    depth_map =  ( (193.001/2) * (3979.911/2) / (disparity_matrix + (124.343/2) ) )
    '''
    #colourmap = depth_map - np.amin(depth_map)
    #colourmap = colourmap * (255 / np.amax(colourmap))

    #plt.imshow(colourmap, cmap='hsv', interpolation='nearest')
    #plt.colorbar()
    #plt.show()
    #plt.savefig('new123.png')
    '''
    return depth_map

#generate_disparity_map('im0.png', 'im1.png', 'new2', 1000, 1482, 140, 11)




