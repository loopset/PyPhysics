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


class ReproduceSTA:
    """Run FRESCO while varying ``r0`` and ``a0``."""

    block_indices = (1, 2)  # Matching potential blocks to modify.
    columns = (1, 2)  # Zero-based columns for r0 and a0 in p(...).
    second_block_r0_offset = -0.15  # Offset applied to r0 in the SO (2nd) block.
    rms_index = 24  # Field containing rms in the selected form-factor line.
    b_index = 27  # Field containing b in the selected form-factor line.

    def __init__(
        self,
        infile: str,
        sysdir: str,
        kp: int = 4,
        kn: int = 2,
    ) -> None:
        """Create a scan from an input file and select its output record.

        Parameters
        ----------
        infile : str
            Original FRESCO input file.
        sysdir : str
            Directory where scan runs are created.
        kp : int, optional
            Potential index to modify, by default 4.
        kn : int, optional
            Form-factor record to parse, by default 2.
        """
        with open(infile) as file:
            self.fConfig = file.readlines()

        self.infile = infile
        self.fSysDir = sysdir
        self.fkp = kp
        self.fkn = kn
        self.results: List[Dict[str, float]] = []

    @staticmethod
    def modify_file(
        config: List[str],
        kp: int,
        r0: float,
        a0: float,
    ) -> List[str]:
        """Change ``r0`` and ``a0`` in the last two matching blocks."""
        output = []
        active = False
        block_number = -1

        for line in config:
            if "&pot" in line and SystematicOverlap.get_param(line, "kp") == kp:
                active = True
                block_number += 1

            if active and block_number in ReproduceSTA.block_indices and "p(" in line:
                start = line.index("p(")
                equals = line.index("=", start)
                values_text, tail = line[equals + 1 :].split("/", 1)
                columns = values_text.split()
                block_r0 = r0
                if block_number == 2:
                    block_r0 += ReproduceSTA.second_block_r0_offset
                columns[ReproduceSTA.columns[0]] = f"{block_r0:.4f}"
                columns[ReproduceSTA.columns[1]] = f"{a0:.4f}"
                line = f"{line[:equals + 1]} {' '.join(columns)} /{tail}"

            if active and line.strip().startswith("&pot /"):
                active = False

            output.append(line)

        return output

    def run(
        self,
        r0_range: tuple[float, float, float],
        a0_range: tuple[float, float, float],
        overwrite: bool = False,
    ) -> None:
        """Run one FRESCO job for every ``r0``/``a0`` pair.

        Ranges are ``(start, stop, step)`` tuples, like ``numpy.arange``.
        """
        self.results.clear()
        r0_values = np.arange(*r0_range)
        a0_values = np.arange(*a0_range)

        for r0 in r0_values:
            for a0 in a0_values:
                name = f"r0_{r0:.2f}_" f"a0_{a0:.2f}"
                run_dir = os.path.join(self.fSysDir, name)
                input_path = os.path.join(run_dir, "fresco.in")
                output_path = os.path.join(run_dir, "fresco.out")
                if not overwrite and os.path.exists(output_path):
                    rms, b = self.parse_output(output_path)
                    self.results.append(
                        {"r0": float(r0), "a0": float(a0), "rms": rms, "b": b}
                    )
                    continue

                if os.path.exists(run_dir):
                    if overwrite:
                        shutil.rmtree(run_dir)
                os.makedirs(run_dir, exist_ok=True)

                modified = self.modify_file(
                    self.fConfig,
                    self.fkp,
                    r0=float(r0),
                    a0=float(a0),
                )
                with open(input_path, "w") as file:
                    file.writelines(modified)

                print(f"Running r0 = {float(r0):.2f}, a0 = {float(a0):.2f}")
                subprocess.run(
                    ["zsh", "-i", "-c", "fresco <fresco.in> fresco.out"],
                    cwd=run_dir,
                    check=True,
                )

                rms, b = self.parse_output(output_path)
                self.results.append(
                    {"r0": float(r0), "a0": float(a0), "rms": rms, "b": b}
                )

    def parse_output(self, file: str) -> Tuple[float, float]:
        """Read ``(rms, b)`` and remove other files from its run directory."""
        form_factors_found = False
        target = f"{self.fkn}:"

        with open(file) as output_file:
            for line in output_file:
                if "SINGLE-PARTICLE FORM FACTORS" in line:
                    form_factors_found = True
                    continue

                if form_factors_found and line.lstrip().startswith(target):
                    try:
                        fields = re.split(r"\s+|(?<=\d)(?=-\d)", line.strip())
                        # for index, value in enumerate(fields):
                        #     print(f"{index}: {value}")
                        rms = float(fields[self.rms_index])
                        b = float(fields[self.b_index])
                    except (IndexError, ValueError) as error:
                        raise ValueError(
                            f"Cannot read rms and b from form-factor line '{target}'"
                        ) from error
                    self._clean_run_directory(os.path.dirname(os.path.abspath(file)))
                    return rms, b

        if not form_factors_found:
            raise ValueError("Cannot find the single-particle form-factor section")
        raise ValueError(f"Cannot find form-factor line '{target}'")

    @staticmethod
    def _clean_run_directory(directory: str) -> None:
        """Keep only the input and output files in a completed run."""
        keep = {"fresco.in", "fresco.out"}
        for entry in os.scandir(directory):
            if entry.name in keep:
                continue
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
