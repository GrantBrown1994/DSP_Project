import numpy as np

def filter_2d(x1: np.ndarray, hn: np.ndarray, mode="full") -> np.ndarray:
    """
    Filter a 2D image with the filter hn.

    Parameters:
    ----------
    x1: np.ndarray
        input signal
    hn: np.ndarray
        filter impulse response
    mode : str
        either 'same' or 'full'. See np.convolve
    """
    hN = len(hn)
    xM, xN = x1.shape

    # determine the row and column lengths of the input and output signals
    if mode == "full":
        rM, rN = (xM + hN - 1, xN + hN - 1)
    else:
        rM, rN = xM, xN

    # apply the filter in the horizontal direction over each row
    result = np.zeros((rM, rN))
    for i in range(xM):
        result[i] = np.convolve(x1[i], hn, mode=mode)

    # apply the filter in the vertical direction over each column
    for j in range(rN):
        result[:, j] = np.convolve(result[:xM, j], hn, mode=mode)

    return result