"""
This script generates figures with larger text for the poster. Report has similar figures but formatted
slightly differently.
"""
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "../").replace("\\", r"\\"))

import matplotlib.pyplot as plt
import numpy as np

from IPython import display
from cycler import cycler

from utils import stem_plot, upc_decode, decode_image, filter_2d
from scipy.io import loadmat, wavfile

default_cycler = cycler(color=["teal", "m", "y", "k"])
plt.rc("axes", prop_cycle=default_cycler, grid=True)
plt.rc("xtick", direction="inout", labelsize="large")
plt.rc("ytick", direction="inout", labelsize="large")
plt.rc("axes", labelsize="large", titlesize="xx-large")

fig_dir = Path(__file__).parent / "figures"
data_dir = Path(__file__).parent / "../data"


def fig_9_31():
    # create xn signal
    n = np.arange(0, 101)
    x_n = 256 * ((n % 50) < 10)

    # apply difference FIR filter to create wn
    w_n = np.convolve([1, -0.9], x_n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    stem_plot(ax1, n[0:75], x_n[0:75])
    stem_plot(ax2, n[0:75], w_n[0:75])

    ax1.set_title("x(n)")
    ax2.set_title("w(n)")
    ax2.set_xlabel("Samples")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_9_31a.svg")

    # build restoration filter cooeficients
    M = 22
    l = np.arange(0, M + 1, dtype=np.int8)
    r = 0.9**l
    y_n = np.convolve(r, w_n)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7))
    stem_plot(ax1, n, w_n[:-1])
    stem_plot(ax2, n, y_n[: -M - 1])
    stem_plot(ax3, n, x_n)

    ax1.set_title("w(n)")
    ax2.set_title("y(n)")
    ax3.set_title("x(n)")
    ax3.set_xlabel("Samples [n]")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_9_31b.svg")

    error = np.abs(y_n[:50] - x_n[:50])
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.stem(error, markerfmt="red")
    ax.title.set_text("Error after Restoration")
    ax.set_xlabel("Samples [n]")
    fig.savefig(fig_dir / "fig_9_31c.svg")


def fig_9_32():
    q = 0.9
    r = 0.9
    M = 22
    h1_n = np.array([1, -q])

    n = np.arange(M)
    h2_n = r**n

    hn = np.convolve(h2_n, h1_n)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    stem_plot(ax, np.arange(len(hn)), hn)
    ax.set_title("$h[n] = h1[n] * h2[n]$")
    ax.set_xlabel("Samples [n]")
    fig.savefig(fig_dir / "fig_9_32_1.svg")

    # create impulse response for FILTER-1
    q = 0.9
    r = 0.9
    M = 22
    h1_n = np.array([1, -q])

    # create impulse response for FILTER-2
    n = np.arange(M)
    h2_n = r**n

    # load image file
    echart = loadmat(data_dir / "echart.mat")["echart"]

    # normalize so image max is 1
    echart = 1 - (echart / 255)

    # apply FILTER-1 to image in both horizontal and vertical directions
    ech90 = filter_2d(echart, h1_n)
    # undo the effects of FILTER 1 by applying the deconvolutional filter to the result
    ech90_decv = filter_2d(ech90, h2_n)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 5))
    ax1.imshow(echart, cmap="binary")
    ax1.set_title("echart.mat")

    ax1.imshow(echart, cmap="binary")
    ax1.set_title("echart.mat")

    ax2.imshow(ech90, cmap="binary")
    ax2.set_title("FILTER-1")

    ax3.imshow(ech90_decv, cmap="binary")
    ax3.set_title("FILTER-2")

    for ax in (ax1, ax2, ax3):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_9_32c.svg")

    # create impulse responses for filters with M= 11, 22, and 33
    M0 = 11
    M1 = 22
    M2 = 33

    h2_n0 = r ** np.arange(M0)
    h2_n1 = r ** np.arange(M1)
    h2_n2 = r ** np.arange(M2)

    # apply each filter to the FILTER-1 result
    ech90_decv0 = filter_2d(ech90, h2_n0)
    ech90_decv1 = filter_2d(ech90, h2_n1)
    ech90_decv2 = filter_2d(ech90, h2_n2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 6))
    ax1.imshow(ech90_decv0, cmap="binary")
    ax1.set_title("M = 11")

    ax2.imshow(ech90_decv1, cmap="binary")
    ax2.set_title("M = 22")

    ax3.imshow(ech90_decv2, cmap="binary")
    ax3.set_title("M = 33")

    for ax in (ax1, ax2, ax3):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_9_323a.svg")


def fig_10_31():
    # create xn signal
    n = np.arange(1, 160)
    xn = 255 * ((n % 30) > 19)

    # apply the FIR filter to the input signal using convolution
    hn = np.array([1, -1])
    yn = np.convolve(xn, hn)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    stem_plot(ax1, np.arange(len(xn)), xn, markersize=3)
    stem_plot(ax2, np.arange(len(yn)), yn, markersize=3)
    ax1.set_title("$x[n]$")
    ax2.set_title("$y[n]$")
    ax2.set_xlabel("Samples [n]")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_31a.svg")

    # normalized edge detector. A transition is considered an edge if the differnce is half the maximum value of xn
    tau = 255 / 2
    edges_n = np.where(np.abs(yn) > tau, 1, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    stem_plot(ax1, np.arange(len(xn)), xn, markersize=3)
    stem_plot(ax2, np.arange(len(edges_n)), edges_n, markersize=3)
    ax2.set_ylim([-0.1, 1.5])
    ax1.set_title("$x[n]$")
    ax2.set_title("$d[n]$")
    ax2.set_xlabel("Samples [n]")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_31d.svg")

    # nonzero returns the indices of the non-zero values, and the non-zero values. We only care
    # about the indices, which we name edges_samples.
    edges_samples = np.nonzero(edges_n)[0]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))
    stem_plot(ax1, np.arange(len(edges_samples)), edges_samples)
    ax1.grid(True)
    ax1.set_ylabel("Sample")
    ax1.set_xlabel("Edges")
    ax1.set_title("Edge Locations of $x[n]$")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_31e.svg")


def fig_10_32():
    imagepath = data_dir / "HP110v3.png"
    im = 1 - plt.imread(imagepath)

    # x and y vectors along image [pixels]
    m, n = np.arange(len(im)), np.arange(len(im[0]))

    # extract one row to read
    read_row = int(len(m) / 2) - 1
    xn = im[read_row]

    fig, (im1, ax1) = plt.subplots(2, 1, figsize=(7, 5))
    # plot the image, n is the columns which we want on the x-axis
    n_mesh, m_mesh = np.meshgrid(n, np.flip(m))
    im1.pcolormesh(n_mesh, m_mesh, im, cmap="binary")

    # plot the location of the single row used for processing
    im1.axhline(y=read_row, linestyle="dashed")
    ax1.plot(np.arange(len(xn)), xn)
    ax1.legend(["$x[n]$"], fontsize=11, loc="upper right", framealpha=1)

    im1.grid(False)
    ax1.set_xlabel("Pixel [n]")
    im1.set_xticks([])
    for ax in (im1, ax1):
        ax.set_xlim([n[0], n[-1]])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32a.png")

    # first difference filter of row signal
    hn = np.array([1, -1])
    yn = np.convolve(xn, hn)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    ax1.plot(np.arange(len(xn)), xn)
    stem_plot(ax2, np.arange(len(yn)), yn, markersize=3)

    ax1.set_title("$x[n]$")
    ax2.set_title("$y[n]$")

    ax2.set_xlabel("Pixel [n]")
    for ax in (im1, ax1, ax2):
        ax.set_xlim([n[0], n[-1]])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32b.svg")

    # apply edge threshold, difference must be 0.5 to be considered a valid edge
    dn = np.where(np.abs(yn) > 0.5, 1, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

    stem_plot(ax1, np.arange(len(yn)), yn, markersize=3)
    stem_plot(ax2, np.arange(len(dn)), dn, markersize=3)
    ax2.set_ylim([-0.1, 1.5])

    ax1.set_title("$y[n]$")
    ax2.set_title("$d [n]$")

    for ax in (im1, ax1, ax2):
        ax.set_xlim([n[0], n[-1]])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32c.svg")

    # pixel locations where edges occur
    edge_loc = np.nonzero(dn)[0]

    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 4))
    stem_plot(ax1, np.arange(len(edge_loc)), edge_loc)
    ax1.set_ylabel("Pixel")
    ax1.set_xlabel("Edges")
    ax1.set_title(r"$\ell[n]$ (Edge Location)")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32c_1.svg")

    # difference filter on the location signal
    delta_n = np.convolve(edge_loc, hn, mode="same")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    stem_plot(ax1, np.arange(len(edge_loc)), edge_loc)
    stem_plot(ax2, np.arange(len(delta_n)), delta_n)

    ax1.set_ylabel("Pixel")
    ax2.set_ylabel("Pixel")
    ax2.set_xlabel("Edges")

    ax1.set_title("$\ell[n]$ (Edge Location)")
    ax2.set_title("$\Delta[n]$ (Bar Widths)")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32d.svg")

    # determine minimum bar width in pixels
    # take all widths smaller than 1.8x the minimum and take the average
    min_width_group = delta_n[delta_n <= (np.min(delta_n) * 1.5)]
    delta_1 = np.average(min_width_group)

    # normalize the bar widths by delta_1, and clip between 1 and 4.
    delta_norm_n = np.clip(np.round(delta_n / delta_1), 1, 4)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    stem_plot(ax1, np.arange(len(delta_n)), delta_n)
    stem_plot(ax2, np.arange(len(delta_n)), delta_norm_n)

    ax1.set_ylabel("px")
    ax2.set_ylabel("Coded Values")
    ax2.set_xlabel("Edges")
    ax2.set_yticks([0, 1, 2, 3, 4])

    ax1.set_title("$\Delta[n]$")
    ax2.set_title("$\Delta_N[n]$")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32g.svg")

    imagepath = data_dir / "OFFv3.png"
    decode_image(imagepath)
    fig = plt.gcf()
    _, ax1, ax2 = fig.axes
    ax1.set_title("$\ell[n]$ (Edge Location)")
    ax2.set_title("$\Delta[n]$ (Bar Widths)")

    ax1.set_ylabel("px")
    ax2.set_ylabel("Coded Values")
    ax2.set_xlabel("Edges")
    ax1.legend([])
    ax2.legend([])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_10_32j.png")


if __name__ == "__main__":
    fig_9_31()
    fig_9_32()
    fig_10_31()
    fig_10_32()
