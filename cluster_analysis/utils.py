import numpy as np

def get_flat(array):
    _, h, w = array.shape
    data = np.empty((h * w, len(array)), dtype=np.int16)
    for i in range(len(array)):
        data[:, i] = array[i, :, :].flatten()

    return data