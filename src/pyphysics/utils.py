import numpy as np
from scipy.interpolate import CubicSpline


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
