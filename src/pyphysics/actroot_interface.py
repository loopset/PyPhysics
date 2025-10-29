from collections import defaultdict
import numpy as np
import uncertainties as un
import uncertainties.unumpy as unp
import hist
import matplotlib.axes as mplaxes
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
from functools import partial
from typing import Any, Callable, List, Dict, Tuple, Union
import ROOT as r  # type: ignore

from pyphysics.root_interface import parse_tgraph, parse_th1
from pyphysics.utils import create_spline3
from pyphysics.styling import sty_hist2d
from pyphysics.utils import set_hist_overflow


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

    def plot(
        self, proj: str = "xy", xmin: float = 0, xmax: float = 128, **kwargs
    ) -> None:
        x = np.array([xmin, xmax])
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

    def moveToX(self, x: float) -> List[float]:
        t = (x - self.fPoint[0]) / self.fDir[0]
        y = self.fPoint[1] + t * self.fDir[1]
        z = self.fPoint[2] + t * self.fDir[2]
        return [x, y, z]

    def __str__(self) -> str:
        return f"Point: {self.fPoint[0]:.2f}, {self.fPoint[1]:.2f}, {self.fPoint[2]:.2f}\nDir: {self.fDir[0]:.2f}, {self.fDir[1]:.2f}, {self.fDir[2]:.2f}"


class TPCInterface:
    def __init__(
        self, tpc: object, dims: Tuple[int, int, int] = (128, 128, 128)
    ) -> None:
        self.fVoxels: Dict[int, List[object]] = defaultdict(
            list
        )  # Vector of voxels per cluster or noise (-1)
        self.fHist = (
            hist.Hist.new.Regular(dims[0], 0, dims[0], name="X", label="X [pads]")
            .Regular(dims[1], 0, dims[1], name="Y", label="Y [pads]")
            .Regular(dims[2], 0, dims[2], name="Z", label="Z [btb]")
        ).Double()  # 3D histogram with charge
        self.fLines: Dict[int, LineInterface] = {}  # Dict with lines
        self.fRP: List[float] = []
        self._fill(tpc)  # Fill histograms and dicts
        return

    def _fill(self, tpc) -> None:
        # Noise
        for v in tpc.fRaw:
            pos = v.GetPosition()
            self.fHist.fill(pos.X(), pos.Y(), pos.Z(), weight=v.GetCharge())
            self.fVoxels[-1].append(v)
        # Clusters
        for c in tpc.fClusters:
            for v in c.GetVoxels():
                pos = v.GetPosition()
                self.fHist.fill(pos.X(), pos.Y(), pos.Z(), weight=v.GetCharge())
                self.fVoxels[c.GetClusterID()].append(v)
            # And also line
            self.fLines[c.GetClusterID()] = LineInterface(c.GetLine())
        # RP if any
        if tpc.fRPs.size():
            rp = tpc.fRPs.front()
            self.fRP = [rp.X(), rp.Y(), rp.Z()]

    def plot(
        self,
        proj: str = "xy",
        isCluster: bool = False,
        withNoise: bool = False,
        **kwargs,
    ) -> None:
        axes = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
        if proj not in axes:
            raise ValueError("Invalid projection")
        # Maximum charge
        maxq = 4096  # Until pad saturates
        args = sty_hist2d
        args.update({"cmax": maxq})
        args.update(kwargs)

        # Plot projection of point cloud
        if not isCluster:
            h = self.fHist.project(*axes[proj])
            set_hist_overflow(h, maxq)  # type: ignore
            h.plot(**args)  # type: ignore
            return

        # Plot clusters
        hcl = self._get_cluster_hist(proj)
        C, x, y = hcl.to_numpy()  # type: ignore
        cmax = len([k for k in self.fVoxels.keys() if k >= 0])
        main_cmap = plt.cm.get_cmap(sty_hist2d.get("cmap"), cmax)
        # Add white color to represent empty bins
        colors = ["white"] + [main_cmap(i) for i in range(cmax)]
        # And construct a listed color map with them
        cmap = mplcolors.ListedColormap(colors)
        # Set color for underflow bins: noise, which have been tagged with -1
        cmap.set_under("pink" if withNoise else "white")
        # Use BoundaryNorm to map values exactly
        bounds = np.arange(0, cmax + 2)  # Color bins
        norm = mplcolors.BoundaryNorm(bounds, cmap.N)
        # Draw empty histogram to set the axes
        ret = hcl.plot(**args)  # type: ignore
        mesh = plt.pcolormesh(x, y, C.T, cmap=cmap, norm=norm, rasterized=True)
        # Draw cbar on automatic cbar axes from hist package
        if ret[1] is not None:  # type: ignore
            cbar = plt.colorbar(mesh, cax=ret[1].ax)  # type: ignore
            cbar.ax.set_ylim(1)

    def _get_cluster_hist(self, proj) -> hist.BaseHist:
        axes = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
        if proj not in axes:
            raise ValueError("Invalid projection")
        ax = axes[proj]
        xaxis = self.fHist.axes[ax[0]]
        yaxis = self.fHist.axes[ax[1]]
        h = (
            hist.Hist.new.Reg(
                len(xaxis.centers),
                xaxis.edges[0],
                xaxis.edges[-1],
                name=xaxis.name,
                label=xaxis.label,
            )
            .Reg(
                len(yaxis.centers),
                yaxis.edges[0],
                yaxis.edges[-1],
                name=yaxis.name,
                label=yaxis.label,
            )
            .Double()
        )
        for id, vals in self.fVoxels.items():
            color = id + 1 if id >= 0 else id
            for v in vals:
                pos = v.GetPosition()  # type: ignore
                bins = (int(pos.X()), int(pos.Y()), int(pos.Z()))
                if proj == "xy":
                    h[bins[0], bins[1]] = color
                elif proj == "xz":
                    h[bins[0], bins[2]] = color
                else:
                    h[bins[1], bins[2]] = color
        return h

    def plot_3d(self, **kwargs) -> None:
        vals = self.fHist.values()
        ok = vals > 0
        x, y, z = np.where(ok)
        maxq = 2000
        q = vals[x, y, z]
        q = np.clip(q, None, maxq)
        x = x + 0.5
        y = y + 0.5
        z = z + 0.5
        # Get axis
        ax = plt.gca()
        ax.scatter(  # type: ignore
            x,
            y,
            z,
            c=q,
            marker="o",
            edgecolor="none",
            linewidth=0.3,
            # alpha=0.75,
            **kwargs,
        )
        ax.set_xlim(0, self.fHist.axes[0].edges[-1])  # type: ignore
        ax.set_ylim(0, self.fHist.axes[1].edges[-1])  # type: ignore
        ax.set_zlim(0, self.fHist.axes[2].edges[-1])  # type: ignore
        return


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
    def __init__(self, file: str, simple: bool = False) -> None:
        self.fEx: Dict[str, Union[float, un.UFloat]] = {}
        self.fSigmas: Dict[str, Union[float, un.UFloat]] = {}
        self.fAmps: Dict[str, Union[float, un.UFloat]] = {}
        self.fLgs: Dict[str, Union[float, un.UFloat]] = {}
        self.fGlobal: np.ndarray | None = None
        self.fFuncs: Dict[str, Callable[..., Any]] = {}
        self.fHistPS: Dict[str, hist.BaseHist] = {}
        self.fChi: float = 0
        self.fNdof: int = 0

        self._read(file, simple)
        return

    def _read(self, file: str, simple: bool) -> None:
        with r.TFile(file) as f:  # type: ignore
            names = f.Get("ParNames")
            res = f.Get("FitResult")
            # Parameters of fit
            self.fChi = res.Chi2()
            self.fNdof = res.Ndf()
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
            # If simple, exist here
            if simple:
                return
            # Global fit
            self.fGlobal = parse_tgraph(f.Get("GraphGlobal"))
            # Functions
            self._init_funcs()
            # PS histograms
            for obj in f.Get("HistoPeaks"):
                name = obj.GetName()
                if "ps" in name:
                    h = parse_th1(obj)
                    if h is not None:
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
                self.fFuncs[key] = a  # type: ignore
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
            label = rf"{model.fName}"
            label = label.replace("l", rf"$\ell$")
            ret.append(ax.plot(xaxis, spe(xaxis), label=label, **args)[0])
            ax.set_ylim(ymin * (1 - scale), ymax * (1 + scale))
        return ret
