import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math as m
import os
cwd = os.getcwd()

def compute_fir_filter_output(x_n):
    w_n = np.zeros(len(x_n) + 1)
    for i in range(0, len(x_n)):
        if i == 0:
            w_n[i] = x_n[i]
        elif i == len(x_n) + 1:
            w_n[i] = -0.9*x_n[i-1]
        else:
            w_n[i] = x_n[i] - 0.9*x_n[i-1]
    return w_n


if __name__ == "__main__":
    pass