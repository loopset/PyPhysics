import numpy as np
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
