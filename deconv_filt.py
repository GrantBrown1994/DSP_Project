import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import matplotlib

dir_ = Path(__file__).parent

def filter_2d(x1, hn, mode='full'):
    """
    Filter a 2D image with the filter hn.
    """
    hN = len(hn)
    xM, xN = x1.shape

    if mode == "full":
        rM, rN = (xM + hN - 1, xN + hN - 1)
    else:
        rM, rN = xM, xN

    result = np.zeros((rM, rN))
    for i in range(xM):
        result[i] = np.convolve(x1[i], hn, mode=mode)

    for j in range(rN):
        result[:, j] = np.convolve(result[:xM, j], hn, mode=mode)

    return result

q = 0.9
r = 0.9
M = 22
h1_n = np.array([1, -q])

n = np.arange(M)
h2_n = r **n

echart = loadmat(dir_ / 'data/echart.mat')["echart"]

# normalize so image max is 1
echart = 1 - (echart / 255)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
ax1.imshow(echart, cmap='binary')
ax1.set_title("echart.mat")
ax1.grid(False)

ech90 = filter_2d(echart, h1_n)
ax2.imshow(ech90, cmap='binary')
ax2.set_title("FILTER-1")
ax2.grid(False)

ech90_decv = filter_2d(ech90, h2_n)
ax3.imshow(ech90_decv, cmap='binary')
ax3.set_title("FILTER-2")
ax3.grid(False)

plt.show()