import numpy as np


def cam_fufvcucv_to_matrix(fu, fv, cu, cv, n=3):
    K = np.eye(n)
    K[0, 0] = fu
    K[1, 1] = fv
    K[0, 2] = cu
    K[1, 2] = cv
    return K


def matrix_to_cam_fufvcucv(K):
    if K.ndim == 3:
        return K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    else:
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
