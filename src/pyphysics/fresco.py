from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import uncertainties as un
import re

from .cross_section import Comparator
from .utils import parse_txt


class SystematicOverlap:
    def __init__(
        self,
        infile: str,
        sysdir: str,
        kp: int = 4,
        what: str = "heavy",
        values: List[Dict[int, Dict[int, float]]] | None = None,
    ) -> None:
        self.fConfig: List[str] = []
        # Read infile
        with open(infile) as f:
            self.fConfig = f.readlines()
        self.fSysDir = sysdir
        # Parameters that will be changed in the &POT namelist
        self.fkp = kp
        self.fValues = values
        # List[Dict[int, Dict[int, float]]]
        # -> List for each iteration in the for loop of variations
        #       -> Outter dict for which TYPES change
        #                 -> Inner dict for which PARS put to given float
        self.fWhat = what
        # Init default values depending on what str
        if self.fValues is None:
            self.fValues = []
            if what == "heavy":
                # Potential
                rv = np.arange(1.1, 1.6, 0.05)
                rso = (1.10 / 1.25) * rv
                for i, r in enumerate(rv):
                    dic = {1: {1: r}, 3: {1: rso[i]}}
                    self.fValues.append(dic)
            else:
                raise ValueError("This class is not yet config to what you want to do")
        # Store keys of each for iteration
        self.fKeys: List[float | str] = []
        self.fOuts: List[Dict[str, NDArray]] = []
        self.fComps: Dict[str, Comparator] = {}
        return

    @staticmethod
    def get_param(line: str, param: str, default: int = -1) -> int:
        """
        Get int parameter from fresco config
        """
        pattern = rf"{re.escape(param)}\s*=\s*([^\s/]+)"
        match = re.search(pattern, line)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return default
        return default

    @staticmethod
    def modify_file(config: List[str], kp, values: Dict[int, Dict[int, float]]):
        out = []
        active = False
        current_type = None

        for line in config:
            if "&pot" in line and SystematicOverlap.get_param(line, "kp") == kp:
                active = True
                if "type" in line:
                    current_type = SystematicOverlap.get_param(line, "type")
                    # current_type = int(line.split("type=")[1].split()[0])

            if active and current_type in values and "p(" in line:
                # find the position of 'p('
                idx = line.index("p(")
                # split the line into left, right around the '=' after p(
                lhs = line[:idx] + line[idx : line[idx:].index("=") + idx]
                rhs = line[idx + line[idx:].index("=") + 1 :]
                vals, tail = rhs.split("/", 1)

                cols = vals.split()
                for i, v in values[current_type].items():
                    cols[i] = f"{v:.4f}"

                line = f"{lhs}= {' '.join(cols)} /{tail}"

            if active and line.strip().startswith("&pot /"):
                active = False
                current_type = None

            out.append(line)

        return out

    def run(self, keep: List[str] = ["202"], overwrite: bool = False):
        if self.fWhat != "heavy" or self.fValues is None:
            return

        rs = [dic[1][1] for dic in self.fValues]
        self.fKeys = rs  # type: ignore

        for i, r in enumerate(rs):
            path = f"ho_r_{r:.2f}"
            full = os.path.join(self.fSysDir, path)
            fortOk = all(os.path.exists(os.path.join(full, f"fort.{k}")) for k in keep)

            # Check if we need to overwrite or create
            if overwrite or not fortOk:
                if os.path.exists(full) and overwrite:
                    shutil.rmtree(full)
                os.makedirs(full, exist_ok=True)

                # Modify file
                mod = self.modify_file(self.fConfig, self.fkp, self.fValues[i])

                # Write fresco.in
                fresco_in = os.path.join(full, "fresco.in")
                with open(fresco_in, "w") as f:
                    f.writelines(mod)

                # Execute subprocess in the directory
                fresco = subprocess.run(
                    ["zsh", "-i", "-c", "fresco <fresco.in> fresco.out"],
                    cwd=full,
                    capture_output=True,
                    text=True,
                )
                if len(fresco.stdout):
                    print(fresco.stdout)

            # Store theoretical outputs in either case
            dic = {k: parse_txt(os.path.join(full, f"fort.{k}")) for k in keep}
            self.fOuts.append(dic)

    def compare(self, exps: Dict[str, NDArray]) -> None:
        self.fComps = {k: Comparator(v) for k, v in exps.items()}
        for key, theos in zip(self.fKeys, self.fOuts):
            for state, data in theos.items():
                self.fComps[state].add_model(
                    key=self._build_comp_key(key), file="", data=data
                )
        for comp in self.fComps.values():
            comp.fit()
        return

    def plot(self) -> None:
        fig, axs = plt.subplots(1, len(self.fComps))

        # ensure list always
        if len(self.fComps) == 1:
            axs = [axs]

        for i, (k, comp) in enumerate(self.fComps.items()):
            ax = axs[i]
            comp.draw(ax=ax)
        fig.tight_layout()
        return

    def _build_comp_key(self, val) -> str:
        return val if isinstance(val, str) else f"{val:.2f}"

    def get(self, which: str = "202") -> Tuple[NDArray, NDArray, NDArray]:
        x = []
        y = []
        ey = []
        for key in self.fKeys:
            x.append(key)
            comp = self.fComps.get(which)
            if comp is None:
                raise ValueError(f"Cannot locate key {which} in comparator dict")
            sf = comp.get_sf(self._build_comp_key(key))
            y.append(un.nominal_value(sf))
            ey.append(un.std_dev(sf))
        return (np.array(x), np.array(y), np.array(ey))
