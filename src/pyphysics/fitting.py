from typing import List
import numpy as np

import uncertainties as un


def fit_poln(x: List | np.ndarray, y: List | np.ndarray, n: int = 1) -> List:
    res, cov = np.polyfit(x, y, deg=n, cov=True)
    sigmas = np.sqrt(np.diag(cov))
    return [un.ufloat(v, u) for (v,u) in zip(res, sigmas)]
