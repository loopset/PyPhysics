import ROOT as r
import numpy as np


def parse_tgraph(g) -> np.ndarray:
    """
    Parse either a TGraphErrors or TGraph to np.ndarray
    """
    if isinstance(g, r.TGraphErrors):  # type: ignore
        return np.array(
            [[g.GetPointX(i), g.GetPointY(i), g.GetErrorY(i)] for i in range(g.GetN())]
        )
    else:
        return np.array([[g.GetPointX(i), g.GetPointY(i)] for i in range(g.GetN())])
