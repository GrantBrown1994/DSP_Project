import matplotlib.pyplot as plt     ## Libary used for displaying plots and formatting plots
import numpy as np                  ## Library used for arrays functions, mathematical functions, etc.
# from matplotlib import cm
# import math
# import cmath
# from scipy.fft import fft,ifft
# import sys
np.set_printoptions(threshold=np.inf)
# import unicodedata
 

def zeropad(x,y):
    xpad = np.zeros((len(x)+2*len(y)-2),dtype=np.complex128)
    ylength = len(y)-1
    for ii in range(0,len(x)):
        xpad[ii + ylength] = x[ii]
    return xpad
 

def convo(x,b):
    bk = b
    x_n = zeropad(x,bk)
    y_padd = np.zeros(len(x_n),dtype=np.complex128)
    y_n = np.zeros(len(x),dtype=np.complex128)
    b_ind = len(b)-1
    for n in range(0,len(x_n)):
        for k in range(0, len(bk)):
            y_padd[n] += (bk[k]*x_n[n-k])
 
    for ii in range(0,len(x)):
        y_n[ii] = y_padd[ii+b_ind]
    return y_n
 
def firconv(p):
    x_n = 256*((np.arange(0,100,1)%50)<10)
    bk = [1,-0.9]
    w_n = convo(x_n,bk)
    n = np.arange(0,100)
    if p == 0:
        print(x_n)
        print(w_n)
        print("3.1 a) I am unsure of what the effect of these filter coefficients is")
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
   
def restor(w_n,x_n):
    M = 22
    r = 0.9
    y_n = np.zeros(len(w_n),dtype=np.complex128)
   
    for n in range(0,len(w_n)):
        for l in range(0, M):
            y_n[n] += (r**l)*w_n[n-l]
   
def restor_plot(w_n,x_n):
   
    M = 22
    r = 0.9
    n_val = np.arange(len(w_n))
    y_n = np.zeros(len(w_n),dtype=np.complex128)
    y_err = np.zeros(len(w_n),dtype=np.complex128)
   
    for n in range(0,len(w_n)):
        for l in range(0, M):
            y_n[n] += (r**l)*w_n[n-l]
    for ii in range(0,len(x_n)):
        y_err[ii] = y_n[ii] - x_n[ii]
 
    fig, axs = plt.subplots(4,1, figsize = (12,12))
    axs[0].stem(n_val,x_n)
    axs[0].grid()
    axs[0].set_ylabel("x(n)")
    axs[0].set_xlabel("Samples")
    axs[1].stem(n_val,w_n)
    axs[1].grid()
    axs[1].set_ylabel("w(n)")
    axs[1].set_xlabel("Samples")
    axs[2].stem(n_val,y_n)
    axs[2].set_ylabel("y(n)")
    axs[2].set_xlabel("samples")
    axs[2].grid()
    axs[3].stem(n_val[0:50],y_err[0:50])
    axs[3].set_ylabel("error y(n) - x(n)")
    axs[3].set_xlabel("samples")
    axs[3].grid()
   
    plt.show()
   
   
   
def main():
    w_n,x_n = firconv(1)
    # firconv(1")
    # firconv(0)
    restor_plot(w_n,x_n)
if __name__== "__main__":
    main()