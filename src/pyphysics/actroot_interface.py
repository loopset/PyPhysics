import numpy as np
import uncertainties as un
import uncertainties.unumpy as unp
import hist
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Dict
import ROOT as r  # type: ignore

from pyphysics.root_interface import parse_tgraph, parse_th1
from pyphysics.utils import create_spline3


class DataManInterface:
    def __init__(self, file: str, mode: str, runs: tuple) -> None:
        self.fChains = {}
        self.fCurrent: int = 0
        self.fBranchName: str = ""
        self.fObj: object = None
        self._init(file, mode, runs)
        return

    def _init(self, file: str, mode: str, runs: tuple) -> None:
        dataman = r.ActRoot.DataManager(file)  # type: ignore

        # Determine ActRoot mode
        actmode = None
        if mode == "tpc":
            actmode = r.ActRoot.ModeType.EReadTPC  # type: ignore
        elif mode == "sil":
            actmode = r.ActRoot.ModeType.EReadSil  # type: ignore
        elif mode == "filter":
            actmode = r.ActRoot.ModeType.EFilter  # type: ignore
        elif mode == "merger":
            actmode = r.ActRoot.ModeType.EMerger  # type: ignore
        else:
            raise ValueError("Invalid mode str passed")

        # Fill dictionary with TTree by run
        for run in np.arange(runs[0], runs[1] + 1):
            dataman.SetRuns(int(run), int(run))
            self.fChains[run] = dataman.GetChain(actmode)

        # Get first run
        self.fCurrent = next(iter(self.fChains))
        return

    def set_branch_address(self, bname: str, o: object) -> None:
        self.fChains[self.fCurrent].SetBranchAddress(bname, o)
        self.fBranchName = bname
        self.fObj = o
        return

    def get_run_entry(self, run: int, entry: int) -> None:
        if run != self.fCurrent:
            if run in self.fChains:
                self.fCurrent = run
                self.set_branch_address(self.fBranchName, self.fObj)
            else:
                raise ValueError(f"Run {run} not in inner map")
        self.fChains[self.fCurrent].GetEntry(int(entry))
        return


class TPCInterface:
    def __init__(self, tpc: object, dims: tuple = (128, 128, 128)) -> None:
        self.fHist = (
            hist.Hist.new.Regular(dims[0], 0, dims[0], name="X", label="X [pads]")
            .Regular(dims[1], 0, dims[1], name="Y", label="Y [pads]")
            .Regular(dims[2], 0, dims[2], name="Z", label="Z [btb]")
        ).Double()
        self._fill(tpc)
        return

    def _fill(self, tpc) -> None:
        # Noise
        for v in tpc.fRaw:
            pos = v.GetPosition()
            self.fHist.fill(pos.X(), pos.Y(), pos.Z(), weight=v.GetCharge())
        # Clusters
        for c in tpc.fClusters:
            for v in c.GetVoxels():
                pos = v.GetPosition()
                self.fHist.fill(pos.X(), pos.Y(), pos.Z(), weight=v.GetCharge())

    def plot(self, proj: str = "xy", **kwargs) -> None:
        axes = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
        if proj not in axes:
            raise ValueError("Invalid projection")
        args = dict(cmin=1, cmax=7000, cmap="managua_r")
        args.update(kwargs)
        self.fHist.project(*axes[proj]).plot(**args)  # type: ignore


class LineInterface:
    def __init__(self, line: object) -> None:
        self.fPoint: np.ndarray = np.array(
            [
                line.GetPoint().X(),  # type: ignore
                line.GetPoint().Y(),  # type: ignore
                line.GetPoint().Z(),  # type: ignore
            ]
        )
        self.fDir: np.ndarray = np.array(
            [
                line.GetDirection().Unit().X(),  # type: ignore
                line.GetDirection().Unit().Y(),  # type: ignore
                line.GetDirection().Unit().Z(),  # type: ignore
            ]
        )
        return

    def plot(self, proj: str = "xy", **kwargs) -> None:
        x = np.array([0, 128])
        t = (x - self.fPoint[0]) / self.fDir[0]
        y = self.fPoint[1] + t * self.fDir[1]
        z = self.fPoint[2] + t * self.fDir[2]
        if proj == "xy":
            plt.plot(x, y, **kwargs)
        elif proj == "xz":
            plt.plot(x, z, **kwargs)
        elif proj == "yz":
            plt.plot(y, z, **kwargs)
        else:
            raise ValueError("Invalid proj passed to plot")
        return

    def __str__(self) -> str:
        return f"Dir: {self.fDir[0]:.2f}, {self.fDir[1]:.2f}, {self.fDir[2]:.2f}"


class KinInterface:
    def __init__(self, reac: str) -> None:
        self.fKin = r.ActPhysics.Kinematics(reac)  # type: ignore
        return

    def plot_kin3(self, **kwargs) -> None:
        graph = self.fKin.GetKinematicLine3()
        data = parse_tgraph(graph)
        args = dict()
        args.update(**kwargs)
        plt.plot(data[:, 0], data[:, 1], **args)
        return


class FitInterface:
    def __init__(self, file: str) -> None:
        self.fEx = {}
        self.fSigmas = {}
        self.fAmps = {}
        self.fLgs = {}
        self.fGlobal: np.ndarray | None = None
        self.fFuncs = {}
        self.fHistPS = {}

        self._read(file)
        return

    def _read(self, file: str) -> None:
        with r.TFile(file) as f:  # type: ignore
            names = f.Get("ParNames")
            res = f.Get("FitResult")
            for i, name in enumerate(names):
                state, par = name.split("_")
                value = res.Parameter(i)
                error = res.Error(i)
                var = None
                if error == 0:
                    var = value
                else:
                    var = un.ufloat(value, error)
                if par == "Amp":
                    self.fAmps[state] = var
                if par == "Mean":
                    self.fEx[state] = var
                if par == "Sigma":
                    self.fSigmas[state] = var
                if par == "Lg":
                    self.fLgs[state] = var
            # Global fit
            self.fGlobal = parse_tgraph(f.Get("GraphGlobal"))
            # Functions
            self._init_funcs()
            # PS histograms
            for obj in f.Get("HistoPeaks"):
                name = obj.GetName()
                if "ps" in name:
                    h = parse_th1(obj)
                    self.fHistPS[name[1:]] = h

        return

    def _init_funcs(self) -> None:
        def gaus(x, amp, mean, sigma):
            if isinstance(x, (float, int)):
                x = [x]
            return [amp * r.TMath.Gaus(e, mean, sigma) for e in x]  # type: ignore

        # to transform ROOT's gauss into scipy's: * sqrt(2pi)sigma
        # I didnt find a way of correlating root and scipy voigt

        def voigt(x, amp, mean, sigma, lg):
            if isinstance(x, (float, int)):
                x = [x]
            return [amp * r.TMath.Voigt(e - mean, sigma, lg) for e in x]  # type: ignore

        for key, amp in self.fAmps.items():
            a = unp.nominal_values(amp)
            if "g" in key:  # gaussian
                mean = unp.nominal_values(self.fEx[key])
                sigma = unp.nominal_values(self.fSigmas[key])
                self.fFuncs[key] = partial(gaus, amp=a, mean=mean, sigma=sigma)
            if "v" in key:  # voigt
                mean = unp.nominal_values(self.fEx[key])
                sigma = unp.nominal_values(self.fSigmas[key])
                lg = unp.nominal_values(self.fLgs[key])
                self.fFuncs[key] = partial(voigt, amp=a, mean=mean, sigma=sigma, lg=lg)
            if "cte" in key or "ps" in key:  ## cte or PS
                self.fFuncs[key] = a
        return

    def get(self, state: str) -> tuple:
        if state in self.fEx and state in self.fSigmas:
            return (self.fEx[state], self.fEx[state])
        else:
            return (None, None)

    def plot_global(self, **kwargs) -> None:
        if self.fGlobal is None:
            return
        args = dict(color="red")
        args.update(kwargs)
        plt.plot(self.fGlobal[:, 0], self.fGlobal[:, 1], **args)  # type: ignore

    def plot_func(
        self, key: str, nbins: int, xmin: float, xmax: float, **kwargs
    ) -> None:
        if key not in self.fFuncs:
            return
        if "g" not in key and "v" not in key:
            print("Do not use plot_func for funcs other than gaus or voigt")
            return
        h = hist.Hist.new.Reg(nbins, xmin, xmax).Double()
        # Fill histogram
        for i, x in enumerate(h.axes[0].centers):
            vals = self.fFuncs[key](x)
            h[i] = vals[0]
        args = dict(histtype="step", yerr=False, flow=None)
        args.update(kwargs)
        h.plot(**args)  # type: ignore
        # # Fill function
        # x = np.linspace(-5, 20, 500)
        # y = self.fFuncs[key](x)
        # plt.plot(x, y, color="orange")


class SFModel:
    def __init__(
        self, name: str, sf: un.UFloat, chi: float, g: np.ndarray | None = None
    ) -> None:
        self.fName = name
        self.fSF = sf
        self.fChi = chi
        self.fGraph = g
        return

    def __str__(self) -> str:
        return f"--SF:\n  Model : {self.fName}\n  SF : {self.fSF:2uS}\n  Chi2 : {self.fChi:.4f}"

    def __repr__(self) -> str:
        return f"SFModel ({self.fName}, {self.fSF}, {self.fChi})"


class SFInterface:
    def __init__(self, file: str) -> None:
        self.fSFs: Dict[str, List[SFModel]] = {}
        self.fExps: Dict[str, np.ndarray] = {}

        self._read(file)
        return

    def _read(self, file: str) -> None:
        with r.TFile(file) as f:  # type: ignore
            keys = f.GetListOfKeys()
            # 1-> Search for unique states
            states = set()
            for key in keys:
                name = key.GetName()
                if "sfs" not in name:
                    continue
                state, _ = name.split("_")
                states.add(state)
            # 2-> Iterate over states
            states = list(states)
            states = sorted(states, key=lambda x: (x[0] != "g", int(x[1:])))
            for state in states:
                # SFs from collection
                col = f.Get(f"{state}_sfs")
                ## Graphs from mg
                gs = f.Get(f"{state}_mg")
                # List of models
                lst = []
                for model in col.GetModels():
                    sf = col.Get(model)
                    # Find graph of model
                    g = None
                    for aux in gs.GetListOfGraphs():
                        if aux.GetTitle() == model:
                            g = parse_tgraph(aux)
                            break
                    # Create SFModel class
                    lst.append(
                        SFModel(
                            model,
                            un.ufloat(sf.GetSF(), sf.GetUSF()),
                            sf.GetChi2Red(),
                            g,
                        )
                    )
                # Append models
                self.fSFs[state] = lst
                # Add experimental graph for peak
                for aux in gs.GetListOfGraphs():
                    if aux.GetName() == f"xs{state}":
                        self.fExps[state] = parse_tgraph(aux)
                        break
        return

    def get(self, state: str) -> List[SFModel]:
        if state in self.fSFs:
            return self.fSFs[state]
        else:
            return []

    def get_model(self, state: str, model: str) -> SFModel | None:
        if state not in self.fSFs:
            return None
        for m in self.fSFs[state]:
            if m.fName == model:
                return m
        return None

    def get_best(self, state: str) -> SFModel | None:
        if state in self.fSFs:
            if not len(self.fSFs[state]):
                return None
            self.fSFs[state].sort(key=lambda sf: sf.fChi)
            return self.fSFs[state][0]
        else:
            return None

    def remove_model(self, state: str, name: str) -> None:
        lst = self.fSFs.get(state)
        if not lst:
            return
        lst[:] = [model for model in lst if model.fName != name]

    def plot_exp(self, state: str, ax: mplaxes.Axes, **kwargs) -> None:
        exp = self.fExps.get(state)
        if exp is None:
            return
        args = {"color": "black", "marker": "s", "ms": 5, "ls": "none"}
        args.update(kwargs)
        ax.errorbar(exp[:, 0], exp[:, 1], yerr=exp[:, 2], **args)
        # ax.set_xlabel(r"$\theta_{\mathrm{CM}}$ [$^{\circ}$]")
        # ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\Omega$ [mb/sr]")
        return

    def plot_models(self, state: str, ax: mplaxes.Axes, **kwargs) -> list:
        ret = []
        models = self.fSFs.get(state)
        exp = self.fExps.get(state)
        if models is None or exp is None:
            return ret
        # X axis settings
        # xmin = exp[:, 0].min()
        # xmax = exp[:, 0].max()
        # offset = 3
        # xaxis = np.linspace(max(0, xmin - offset), xmax + offset, 200)
        xaxis = np.linspace(0, 180, 800)
        # Y axis settings
        ymin = exp[:, 1].min()
        ymax = exp[:, 1].max()
        scale = 0.9
        args: dict = {"marker": "none", "lw": 1.25}
        args.update(kwargs)
        for model in reversed(models):
            g = model.fGraph
            if g is None:
                continue
            spe = create_spline3(g[:, 0], g[:, 1])
            ret.append(ax.plot(xaxis, spe(xaxis), label=model.fName, **args)[0])
            ax.set_ylim(ymin * (1 - scale), ymax * (1 + scale))
        return ret
