import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import root_scalar
from typing import Callable


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
