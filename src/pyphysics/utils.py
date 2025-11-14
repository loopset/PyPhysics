import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import root_scalar
import hist
from matplotlib.colorbar import Colorbar
import matplotlib.axes as mplaxes
import matplotlib.ticker as mpltick
from typing import Callable, List


def parse_txt(file: str, ncols: int = 2) -> np.ndarray:
    """
    Function that parses a .txt file in the same fashion as the TGraphErrors ctor
    """
    with open(file) as f:
        lines = f.readlines()
    ret = []
    for line in lines:
        split = line.split()
        if len(split) == ncols:
            try:
                ret.append([float(col) for col in split])
            except:  ## probably we are parsing a string; skip in this case
                continue
    return np.array(ret)


def create_spline3(x, y) -> CubicSpline:
    return CubicSpline(x, y)


def create_trans_imshow(
    x0: tuple, x1: tuple, y0: tuple, y1: tuple, logx: bool = False, logy: bool = False
) -> tuple:
    """
    Function that creates mappings from physical XY data to imshow pixels.
    Each point comes as:
    (pixel coordinate, physical coordinate)
    """

    def build_trans(p0: tuple, p1: tuple, log: bool) -> Callable[[float], float]:
        phys = [p[1] for p in [p0, p1]]
        pix = [p[0] for p in [p0, p1]]
        if log:
            phys = np.log10(phys)
        fit = np.poly1d(np.polyfit(phys, pix, 1))

        def trans(x):
            return fit(x if not log else np.log10(x))

        return trans

    # And create each trans
    xtrans = build_trans(x0, x1, logx)
    ytrans = build_trans(y0, y1, logy)

    return (xtrans, ytrans)


def create_interp1d(x, y) -> interp1d:
    return interp1d(x, y)


def find_root(func, y, interval: list | None = None) -> float:
    finder = root_scalar(lambda x: func(x) - y, bracket=interval)
    if finder.converged:
        return finder.root
    else:
        raise ValueError("Cannot determine root")


def set_hist_overflow(h: hist.BaseHist, overflow: float) -> None:
    """
    Sets real overflow of hist.Hist,
    because histplot cmin/cmax sets it to None
    """
    contents = h.view(flow=True)
    contents[contents > overflow] = overflow  # type: ignore
    return None


def annotate_subplots(
    axs: mplaxes.Axes | List | np.ndarray, x=0.125, y=0.925, color="black"
) -> None:
    if isinstance(axs, mplaxes.Axes):
        axs = np.array([axs])
    for i, ax in enumerate(axs):
        ax: mplaxes.Axes
        ax.annotate(
            chr(97 + i) + ")",
            xy=(x, y),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=color,
        )
    return None


class ROOTLogFormatter(mpltick.LogFormatterSciNotation):
    """
    Custom class overriding log formatter in pcolormesh
    to print 10^0 tick as 1, as in ROOT
    """

    def __init__(self, base=10, labelOnlyBase=True, **kwargs):
        super().__init__(base=base, labelOnlyBase=labelOnlyBase, **kwargs)

    def __call__(self, x, pos=None):
        if x == 1:
            return "1"
        return super().__call__(x, pos)


def apply_log_formatter(ax: mplaxes.Axes) -> None:
    ax.yaxis.set_major_formatter(ROOTLogFormatter())
    return


def apply_ROOT_colorbar(cbar: Colorbar) -> None:
    cbar.formatter = ROOTLogFormatter()
    cbar.update_ticks()
    return
