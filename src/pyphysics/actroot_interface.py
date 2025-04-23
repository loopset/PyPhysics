import numpy as np
import uncertainties as un
import hist
import matplotlib.pyplot as plt
import ROOT as r  # type: ignore

from pyphysics.root_interface import parse_tgraph


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
                if par == "Mean":
                    self.fEx[state] = var
                if par == "Sigma":
                    self.fSigmas[state] = var
        return

    def get(self, state: str) -> tuple:
        if state in self.fEx and state in self.fSigmas:
            return (self.fEx[state], self.fEx[state])
        else:
            return (None, None)


class SFModel:
    def __init__(self, name: str, sf: un.UFloat, chi: float) -> None:
        self.fName = name
        self.fSF = sf
        self.fChi = chi
        return

    def __str__(self) -> str:
        return f"--SF:\n  Model : {self.fName}\n  SF : {self.fSF:2uS}\n  Chi2 : {self.fChi:.4f}"


class SFInterface:
    def __init__(self, file: str) -> None:
        self.fSFs = {}

        self._read(file)
        return

    def _read(self, file: str) -> None:
        with r.TFile(file) as f:  # type: ignore
            keys = f.GetListOfKeys()
            for key in keys:
                name = key.GetName()
                if "sfs" not in name:
                    continue
                state, _ = name.split("_")
                lst = []
                # Read data
                collection = f.Get(name)
                for model in collection.GetModels():
                    sf = collection.Get(model)
                    lst.append(
                        SFModel(
                            model, un.ufloat(sf.GetSF(), sf.GetUSF()), sf.GetChi2Red()
                        )
                    )
                self.fSFs[state] = lst
        return

    def get(self, state: str) -> list:
        if state in self.fSFs:
            return self.fSFs[state]
        else:
            return []

    def get_best(self, state: str) -> SFModel | None:
        if state in self.fSFs:
            if not len(self.fSFs[state]):
                return None
            self.fSFs[state].sort(key=lambda sf: sf.fChi)
            return self.fSFs[state][0]
        else:
            return None
