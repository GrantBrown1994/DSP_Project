import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from cycler import cycler

dir_ = Path(__file__).parent

warnings.filterwarnings("ignore")
default_cycler = cycler(color=["teal", "m", "y", "k"])
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("font", **{"size": 8})


def stem_plot(
    ax: plt.Axes, xd, yd, color="teal", markersize=6, linestyle="solid", label=None
):
    """Create customized stem plot on axes with data (xd, yd)"""
    markerline, stemlines, baseline = ax.stem(xd, yd, label=label)
    plt.setp(stemlines, color=color, linestyle=linestyle)
    plt.setp(markerline, markersize=markersize, color=color)


def section_31():
    x = np.arange(1, 160)

    xn = 255 * ((x % 30) > 19)

    hn = np.array([1, -1])
    # simple edge detector
    yn = np.convolve(xn, hn)
    # normalized edge detector
    tau = 255 / 2
    edges_n = np.where(np.abs(yn) > tau, 1, 0)

    # sample locations where edges occur
    edges_samples = np.nonzero(edges_n)[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))
    stem_plot(ax1, np.arange(len(xn)), xn, markersize=3)
    stem_plot(ax2, np.arange(len(yn)), yn, markersize=3)
    stem_plot(ax3, np.arange(len(edges_n)), edges_n, markersize=3)
    ax3.set_ylim([-0.1, 1.5])

    ax1.legend(["$x[n]$"], fontsize=10, loc="upper right")
    ax2.legend(["$y[n]$"], fontsize=10, loc="upper right")
    ax3.legend([r"$|y[n]| > \tau$"], fontsize=10, loc="upper right")

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    stem_plot(ax1, np.arange(len(edges_samples)), edges_samples)
    ax1.grid(True)
    ax1.set_ylabel("Edge Locations [sample]", fontsize=10)
    ax1.set_xlabel("Edges", fontsize=10)


if __name__ == "__main__":
    section_31()
