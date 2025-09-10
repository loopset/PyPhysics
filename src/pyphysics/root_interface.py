import ROOT as r
import numpy as np
import hist


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


def parse_th1(h) -> hist.BaseHist | None:
    """
    Convert a TH1 to boost histogram
    """
    if not isinstance(h, r.TH1):  # type: ignore
        return None
    nbins = h.GetNbinsX()
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    label = h.GetXaxis().GetTitle()
    ret = hist.Hist.new.Reg(nbins, xmin, xmax, label=label).Double()
    x = []
    y = []
    for i in range(1, nbins + 1):
        x.append(h.GetBinCenter(i))
        y.append(h.GetBinContent(i))
    ret.fill(x, weight=y)
    return ret


def parse_tcutg(g) -> np.ndarray:
    """
    Just a wrapper to TCutG
    """
    return parse_tgraph(g)


def parse_tf1(f) -> np.ndarray:
    """
    Reads into [x,y] array a TF1 in the range [xmin, xmax]
    """
    if not isinstance(f, r.TF1):  # type: ignore
        raise ValueError("Expected TF1 object")
    x = np.linspace(f.GetXmin(), f.GetXmax(), 100)
    y = []
    for val in x:
        y.append(f.Eval(val))
    y = np.array(y)
    return np.column_stack((x, y))
