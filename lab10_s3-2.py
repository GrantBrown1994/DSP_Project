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


def upc_decode(bar_w):
    assert len(bar_w) == 59

    # check start and stop delimiters
    if np.any(bar_w[:3] != 1) or np.any(bar_w[-3:] != 1):
        raise ValueError("invalid delimiters")

    # remove delimters
    bar_codes = bar_w[3:-3]

    # check and remove the middle delimiter
    if np.any(bar_codes[(6 * 4) : (6 * 4) + 5] != 1):
        raise ValueError("invalid delimiters")
    bar_codes = np.concatenate([bar_codes[: (6 * 4)], bar_codes[(6 * 4) + 5 :]])

    # reshape so each digit is on its own row
    bar_codes_shp = np.reshape(bar_codes, (12, 4))

    code_map = np.array(
        [
            [3, 2, 1, 1],  # 0
            [2, 2, 2, 1],  # 1
            [2, 1, 2, 2],  # 2
            [1, 4, 1, 1],  # 3
            [1, 1, 3, 2],  # 4
            [1, 2, 3, 1],  # 5
            [1, 1, 1, 4],  # 6
            [1, 3, 1, 2],  # 7
            [1, 2, 1, 3],  # 8
            [3, 1, 1, 2],  # 9
        ]
    )

    result = []
    for c in bar_codes_shp:
        idx = [i for i, m_i in enumerate(code_map) if np.all(c == m_i)]
        result.append(idx[0] if len(idx) else -1)
    return result


def section_32():
    """
    Simple barcode reader for horizontally level images. Reads a single row in the middle of the image.
    """
    imagepath = dir_ / "data/HP110v3.png"
    im = 1 - plt.imread(imagepath)

    m, n = np.arange(len(im)), np.arange(len(im[0]))

    # extract one row to read
    read_row = int(len(m) / 2) - 1
    xn = im[read_row]

    hn = np.array([1, -1])
    yn = np.convolve(xn, hn)
    edges_n = np.where(np.abs(yn) > 0.5, 1, 0)

    fig, (im1, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(9, 10))
    # plot the image, n is the columns which we want on the x-axis
    n_mesh, m_mesh = np.meshgrid(n, np.flip(m))
    im1.pcolormesh(n_mesh, m_mesh, im, cmap="binary")

    ax1.plot(np.arange(len(xn)), xn)
    stem_plot(ax2, np.arange(len(yn)), yn, markersize=3)
    stem_plot(ax3, np.arange(len(edges_n)), edges_n, markersize=3)
    ax3.set_ylim([-0.1, 1.5])

    ax1.legend(["$x[n]$"], fontsize=11, loc="upper right", framealpha=1)
    ax2.legend(["$y[n]$"], fontsize=11, loc="upper right", framealpha=1)
    ax3.legend([r"$|y[n]| > \tau$"], fontsize=11, loc="upper right", framealpha=1)

    for ax in (im1, ax1, ax2, ax3):
        ax.set_xlim([n[0], n[-1]])
        ax.set_xticks([])

    im1.axhline(y=read_row, linestyle="dashed")
    plt.tight_layout()

    # sample locations where edges occur
    edge_loc = np.nonzero(edges_n)[0]

    # difference filter on the location signal
    delta_n = np.convolve(edge_loc, hn, mode="same")

    # determine minimum bar width in pixels
    # drop all widths greater than 1.5x the min and take the average
    min_width_group = delta_n[delta_n <= (np.min(delta_n) * 1.5)]
    min_average = np.average(min_width_group)

    delta_norm_n = np.clip(np.round(delta_n / min_average), 1, 4)

    # find the first sequence of 3 ones, this marks the beginning and end of a valid code
    delimiter_match = np.convolve(delta_norm_n, [1, 1, 1])
    delimiter_loc = np.argwhere(delimiter_match == 3).flatten()
    start_loc, stop_loc = delimiter_loc[0] - 2, delimiter_loc[-1]

    # clip start loc to 0
    start_loc = 0 if start_loc < 0 else start_loc
    # clip the sequence at the delimters
    bar_w = delta_norm_n[start_loc : stop_loc + 1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10))
    stem_plot(ax1, np.arange(len(edge_loc)), edge_loc)

    stem_plot(ax2, np.arange(len(delta_n)), delta_n)

    stem_plot(ax3, np.arange(len(delta_n)), delta_norm_n)
    ax3.axvspan(start_loc - 0.5, start_loc + 2.5, color="k", alpha=0.3)
    ax3.axvspan(stop_loc - 2.5, stop_loc + 0.5, color="k", alpha=0.3)

    ax1.grid(True)
    ax1.set_ylabel("Edge Locations [px]", fontsize=10)
    ax3.set_xlabel("Edges", fontsize=10)
    ax2.set_ylabel("Bar Widths [px]", fontsize=10)
    ax2.grid(True)
    ax3.set_ylabel("Bar Widths (Normalized)", fontsize=10)
    ax3.grid(True)
    ax3.set_yticks([0, 1, 2, 3, 4])
    plt.tight_layout()

    # check that length is 59
    assert len(bar_w) == 59

    return upc_decode(bar_w)


def section_3b():
    """
    Improved barcode reader that can read slanted images. 
    """
    imagepath = dir_ / "data/OFFv3.png"
    im = 1 - plt.imread(imagepath)

    m, n = np.arange(len(im)), np.arange(len(im[0]))

    mid_row = int(len(m) / 2)
    read_rows = np.arange(mid_row - 10, mid_row + 10)

    # each row must have a maximum of 100 edges (at least 59 expected)
    edge_loc_rows = np.zeros((len(read_rows), 100))
    for i, r in enumerate(read_rows):
        xn = im[r]

        hn = np.array([1, -1])
        yn = np.convolve(xn, hn)
        edges_n = np.where(np.abs(yn) > 0.5, 1, 0)

        # sample locations where edges occur
        edge_loc_i = np.nonzero(edges_n)[0]

        edge_loc_rows[i, : len(edge_loc_i)] = edge_loc_i

    edge_loc = np.average(edge_loc_rows, axis=0)
    edge_loc = edge_loc[: np.max(np.nonzero(edge_loc)) + 1]

    # difference filter on the location signal, in pixels
    delta_n = np.convolve(edge_loc, np.array([1, -1]))[:-1]

    # determine minimum bar width in pixels
    # drop all widths greater than 2x the min and take the average
    min_width_group = delta_n[delta_n <= (np.min(delta_n) * 2)]
    min_average = np.average(min_width_group)

    delta_norm_n = np.clip(np.round(delta_n / min_average), 1, 4)

    # find the first sequence of 3 ones, this marks the beginning and end of a valid code
    delimiter_match = np.convolve(delta_norm_n, [1, 1, 1])
    delimiter_loc = np.argwhere(delimiter_match == 3).flatten()
    start_loc, stop_loc = delimiter_loc[0] - 2, delimiter_loc[-1]

    # clip start loc to 0
    start_loc = 0 if start_loc < 0 else start_loc
    # clip the sequence at the delimters
    bar_w = delta_norm_n[start_loc : stop_loc + 1]

    fig, (im1, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 7))
    # plot the image, n is the columns which we want on the x-axis
    n_mesh, m_mesh = np.meshgrid(n, np.flip(m))
    im1.pcolormesh(n_mesh, m_mesh, im, cmap="binary")

    stem_plot(ax1, edge_loc, delta_n)
    stem_plot(ax2, edge_loc, delta_norm_n)

    ax2.axvspan(
        edge_loc[start_loc] - 0.5, edge_loc[start_loc] + 2.5, color="k", alpha=0.3
    )
    ax2.axvspan(
        edge_loc[stop_loc] - 2.5, edge_loc[stop_loc] + 0.5, color="k", alpha=0.3
    )

    ax1.legend(["$x[n]$"], fontsize=11, loc="upper right", framealpha=1)
    ax2.legend(["$y[n]$"], fontsize=11, loc="upper right", framealpha=1)

    for ax in (im1, ax1, ax2):
        ax.set_xlim([n[0], n[-1]])
        ax.set_xticks([])

    im1.axhline(y=mid_row, linestyle="dashed")
    plt.tight_layout()

    ax1.grid(True)
    ax1.set_ylabel("Bar Widths [px]", fontsize=10)
    ax2.grid(True)
    ax2.set_ylabel("Bar Widths (Normalized)", fontsize=10)
    ax2.set_yticks([0, 1, 2, 3, 4])
    plt.tight_layout()

    # check that length is 59
    assert len(bar_w) == 59

    return upc_decode(bar_w)
    # [0, 4, 6, 5, 0, 0, 7, 0, 3, 1, 9, 5]


if __name__ == "__main__":
    print(section_3b())
