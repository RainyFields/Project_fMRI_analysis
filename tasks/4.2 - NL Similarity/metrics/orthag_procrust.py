import numpy as np
from scipy.linalg import orthogonal_procrustes

def scale_matrice(matrix):
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    # Apply Min-Max scaling to each matrix in the list
    scaled_matrix = (matrix - min_value) / (max_value - min_value)
    return scaled_matrix

def std_matrix(data1):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    
    if mtx1.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    trans_mtx = np.mean(mtx1, 0)
    mtx1 -= trans_mtx

    norm1 = np.linalg.norm(mtx1)

    if norm1 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    return mtx1, norm1, trans_mtx # org_mtx = mtx * norm1 + trans_mtx

def procrustes_disparity(mtx1, mtx2):
    # data are after std_matrix
    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return disparity