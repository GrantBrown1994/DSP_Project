import matplotlib.pyplot as plt     ## Libary used for displaying plots and formatting plots
import numpy as np                  ## Library used for arrays functions, mathematical functions, etc.
from matplotlib import cm
import math
import cmath
from scipy.fft import fft,ifft
# import sys
np.set_printoptions(threshold=np.inf)
import unicodedata


def firconv(p):
    x_n = 256*((np.arange(0,100,1)%50)<10)
    bk = [1,0.9]
    w_n = np.convolve(x_n,bk)
    n = np.arange(0,100)
    if p == 0:
        print(x_n)
        print(w_n)                                                                     ########FIX 3.1 a) Text response########
        print("3.1 a) I am unsure of what the effect of these filter coefficients is, will read into.")       
        print("3.1 b) The length of the filtered signal is the length of x_n added to the length of bk minus 1.")
        print("The nonzero portion of w_n is one value longer than x_n as well, producing 11 vales rather than 10.")
        print("length of x_n is ", len(x_n), "samples")
        print("length of w_n is ", len(w_n), "samples")
        fig, axs = plt.subplots(2,1, figsize = (12,12))
        axs[0].stem(n[0:75],x_n[0:75], markerfmt = 'red')
        axs[0].title.set_text("x(n)")
        axs[0].grid()
        axs[0].set_ylabel("x(n)")
        axs[0].set_xlabel("Samples")
        axs[1].stem(n[0:75],w_n[0:75])
        axs[1].title.set_text("w(n)")
        axs[1].set_ylabel("w(n)")
        axs[1].set_xlabel("samples")
        axs[1].grid()
        plt.show()
    return w_n,x_n
    
def restor():
    w_n,x_n = firconv(1)
    # print(w_n)
    # print(x_n)
    r = 0.9
    M = 22
    l = np.arange(0,M)
    n = np.arange(len(w_n))
    y_sum = np.zeros(len(w_n), dtype=np.complex128)
    y_n = np.zeros(len(w_n),dtype=np.complex128)
    
    for ii in range(0,M):
        for k in range(0,len(w_n)):
            y_sum[k] = (r**ii)*w_n[k-ii]
        # print("ii = ",ii)
        # print(y_sum)
        y_n[ii] = np.sum(y_sum)
    
    fig, axs = plt.subplots(2,1, figsize = (12,12))
    axs[0].stem(n,w_n, markerfmt = 'red')
    axs[0].title.set_text("w(n)")
    axs[0].grid()
    axs[0].set_ylabel("w(n)")
    axs[0].set_xlabel("Samples")
    axs[1].stem(n,y_n)
    axs[1].title.set_text("y(n)")
    axs[1].set_ylabel("y(n)")
    axs[1].set_xlabel("samples")
    axs[1].grid()
    plt.show()
    
def main():
    # firconv("no")
    restor()
if __name__== "__main__":
    main()