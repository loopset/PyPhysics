from collections import defaultdict
import uncertainties as unc
import pandas as pd
from fractions import Fraction
import re
import math
import copy
from dataclasses import dataclass
from typing import Dict, List, Callable, Any
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Orbital:
    """
    A class representing the (nlj)
    quantum numbers that identify a nucleon orbital
    """

    letters = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}

    def __init__(self, n: int, l: int, j: float, t: float = 0) -> None:
        self.n = n
        self.l = l
        self.j = j
        self.t = t
        return

    @classmethod
    def from_str(cls, string: str):
        letter = re.search(r"[spdfghi]", string)
        if not letter:
            raise ValueError("Cannot read l letter from str")
        it = letter.start()
        n = int(string[:it])
        l = -1
        for i, val in cls.letters.items():
            if val == string[it]:
                l = i
        if l == -1:
            raise ValueError(
                "Cannot parse string as QuantumNumber. Check the given letter"
            )
        j = float(Fraction(string[it + 1 :]))
        return cls(n, l, j)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Orbital):
            return NotImplemented
        return (
            self.n == other.n
            and self.l == other.l
            and self.j == other.j
            and self.t == other.t
        )

    def __hash__(self) -> int:
        return hash((self.n, self.l, self.j, self.t))

    def __str__(self) -> str:
        return f"Quantum number:\n n : {self.n}\n l : {self.l}\n j : {self.j}\n t : {self.t}"

    def __repr__(self) -> str:
        return f"nljt:({self.n},{self.l},{self.j},{self.t})"

    def format(self) -> str:
        # If spectroscopic information
        if self.l >= 0:
            frac = Fraction(self.j).limit_denominator()
            ret = rf"{self.n}{Orbital.letters[self.l]}$_{{{frac}}}$"
        else:  # Summary mode: no info on l nor n. n is "state counter" and j = parity * j
            frac = Fraction(abs(self.j)).limit_denominator()
            pi = "+" if self.j > 0 else "-"
            ret = rf"${frac}^{{{pi}}}_{{{self.n}}}$"
        return ret

    def format_simple(self) -> str:
        if self.l >= 0:
            frac = Fraction(self.j).limit_denominator()
            ret = f"{self.n}{Orbital.letters[self.l]}{frac}"
        else:
            frac = Fraction(abs(self.j)).limit_denominator()
            pi = "+" if self.j > 0 else "-"
            ret = f"{frac}{pi}{self.n}"
        return ret

    def get_j_fraction(self) -> str:
        frac = Fraction(self.j).limit_denominator()
        return f"{frac}"

    def degeneracy(self) -> int:
        return int(2 * self.j + 1)


## Legacy: create aliases for backwards compatibility
QuantumNumbers = Orbital


@dataclass(frozen=True)
class StateJpi:
    """
    A class representing the (Jpi) of a state
    """

    j: float = -1
    pi: int = 0

    def __str__(self) -> str:
        return f"State:\n J : {self.j}\n pi : {self.pi}"

    def __repr__(self) -> str:
        return f"Jpi:({self.j},{self.pi})"

    def degeneracy(self) -> int:
        return int(2 * self.j + 1)

    def format(self) -> str:
        frac = Fraction(self.j).limit_denominator()
        pi = "+" if self.pi > 0 else "-"
        return rf"${frac}^{{{pi}}}$"

    def format_simple(self) -> str:
        frac = Fraction(self.j).limit_denominator()
        pi = "+" if self.pi > 0 else "-"
        return f"{frac}{pi}"


class ShellModelData:
    """
    A class containing the Ex and SF data from a shell-model calculation
    """

    def __init__(
        self,
        ex: float | unc.UFloat,
        sf: float | unc.UFloat,
        jpii: StateJpi = StateJpi(),
        jpif: StateJpi = StateJpi(),
    ) -> None:
        self.Ex = ex
        self.SF = sf
        self.Jpii = jpii
        self.Jpif = jpif
        return

    def __str__(self) -> str:
        return f"Data:\n  Ex : {self.Ex:.2f}\n  SF : {self.SF:.2f}\n Jpii : {self.Jpii.format_simple()}\n  Jpif : {self.Jpif.format_simple()}"

    def __repr__(self) -> str:
        return f"SMData(Ex: {self.Ex:.2f}, SF: {self.SF:.2f}, Jpii: {self.Jpii.format_simple()}, Jpif: {self.Jpif.format_simple()})"


# Alias
SMDataDict = Dict[Orbital, List[ShellModelData]]
GroupExJpif = Dict[
    tuple[str, float | unc.Variable], list[tuple[Orbital, float | unc.Variable]]
]
GroupJpif = Dict[
    str, list[tuple[float | unc.Variable, list[tuple[Orbital, float | unc.Variable]]]]
]


class ShellModel:
    def __init__(
        self, files: list = [], is_lsf: bool = False, is_adding: bool = False
    ) -> None:
        self.data: SMDataDict = defaultdict(list)
        self.BE = 0
        self.is_lsf = is_lsf
        self.is_adding = is_adding

        if len(files):
            if not is_lsf:
                self.__buildFromKSHELL(files)
            else:
                self.__buildFromLSF(files)
        return

    def __buildFromKSHELL(self, files: list) -> None:
        """
        Meant to read KSHELL-generated files, such as
        those of SFO-tls
        """
        # Parse each file
        for file in files:
            self.__parseKSHELL(file)

        # Determine binding energy
        # If adding reaction
        if self.is_adding:
            max_ex = max(o.Ex for lst in self.data.values() for o in lst)  # type: ignore
            self.BE = max_ex
        else:  # removal reaction
            min_ex = min(o.Ex for lst in self.data.values() for o in lst)  # type: ignore
            self.BE = min_ex
        # print(f"Binding energy: {self.BE}")

        ####################################################################
        # LEGACY: DO NOT USE
        # maxSF = max(
        #     [s for sublist in self.data.values() for s in sublist],
        #     key=lambda sm: unc.nominal_value(sm.SF),
        # )
        # # print(maxSF)
        # self.BE = maxSF.Ex
        #####################################################################

        # And substract it from states
        for _, sublist in self.data.items():
            for state in sublist:
                if self.is_adding:
                    state.Ex = self.BE - state.Ex  # type: ignore
                else:
                    state.Ex = state.Ex - self.BE  # type: ignore
                state.Ex = round(state.Ex, 3)
        return

    def __buildFromLSF(self, files: list) -> None:
        """
        Meant to read WBT (.lsf) input files
        """
        # Parse each file
        for file in files:
            self.__parse_lsf(file)
        return

    def __parseKSHELL(self, file: str) -> None:
        ret = defaultdict(list)
        with open(file, "r") as f:
            n, l, j = -1, -1, -1
            pif, pii = 0, 0  # parities
            for lin in f:
                line = lin.strip()
                if not line:
                    continue
                if "orbit" in line:
                    # Set nlj of current states
                    for c, column in enumerate(line.split()):
                        if c == 2:
                            n = int(column)
                        elif c == 3:
                            l = int(column)
                        elif c == 4:
                            j = int(column)
                if "parity" in line:
                    for c, column in enumerate(line.split()):
                        if c == 2:
                            pif = int(column)
                        elif c == 3:
                            pii = int(column)
                if re.match(
                    r"^\d+\(", line
                ):  # States start with 2*Jf(. This is their clear signature
                    # 2Jf
                    jf = float(line[0].strip()) / 2
                    # 2Ji
                    ji = float(line[17].strip()) / 2
                    # Ex
                    ex = float(line[34:41].strip())
                    # C2S
                    c2s = float(line[45:51].strip())
                    # Define orbital
                    q = Orbital(n, l, j / 2)
                    # Define states
                    state_ji = StateJpi(ji, pii)
                    state_jf = StateJpi(jf, pif)
                    # Define values
                    sm = ShellModelData(ex, c2s, state_ji, state_jf)
                    # Push to dict
                    ret[q].append(sm)
        # And update dict
        for k, v in ret.items():
            self.data[k].extend(v)

    def __parse_lsf(self, file) -> None:
        ret = defaultdict(list)
        with open(file, "r") as f:
            for lin in f:
                line = lin.strip()
                if not line:
                    continue
                if line.startswith("!"):
                    continue
                if "sum" in line:
                    continue
                # Get n, l and 2j
                n = line[29]
                l = line[31]
                j2 = line[33:35]
                if n == "n":
                    continue

                # Convert to q
                # n -> n - 1 to use 0 convention
                q = Orbital(int(n) - 1, int(l), float(j2) / 2)

                # Get C2S and Ex
                c2s = float(line[62:69])
                exi = float(line[72:79])
                exf = float(line[81:88])
                ex = max(exi, exf)

                # Build and add data
                # WARNING: Here for the moment ignore Ji and Jf
                data = ShellModelData(ex, c2s, StateJpi(), StateJpi())
                ret[q].append(data)
        # And update dict
        for k, v in ret.items():
            self.data[k].extend(v)

    def add_summary(self, file: str) -> None:
        summary: SMDataDict = defaultdict(list)
        # Parse summary file
        with open(file, "r") as f:
            for line in f:
                if not line:
                    continue
                try:
                    N = int(line[0:5])
                except ValueError:
                    continue
                j = line[7:11]
                pi = +1 if line[12] == "+" else -1
                count = line[14:19]
                t = line[21:25]
                ex = line[37:45]
                # Convert
                count = int(count)
                j = float(Fraction(j))
                t = float(Fraction(t))
                ex = float(ex)
                # Build key in this format. L = -1 indicates that is "summary" version instead of "spectroscopic one"
                key = Orbital(count, -1, pi * j, t)
                # Add information on Jpi to INITIAL part only
                state_ji = StateJpi(j, pi)
                state_jf = StateJpi()
                val = ShellModelData(ex, -1, state_ji, state_jf)
                summary[key].append(val)
        # Overwrite
        self.data = summary
        return

    def add_isospin(self, file: str, df: pd.DataFrame | None = None) -> None:
        newdict: SMDataDict = defaultdict(list)
        # Parse summary file
        with open(file, "r") as f:
            for line in f:
                if not line:
                    continue
                try:
                    N = int(line[0:5])
                except ValueError:
                    continue
                j = line[7:11]
                # pi = +1 if line[12] == "+" else -1
                t = line[21:25]
                ex = line[37:45]
                # Convert
                j = float(Fraction(j))
                t = float(Fraction(t))
                ex = float(ex)
                # If df is passed, get 2T from it
                if df is not None:
                    gated = df[df["index"] == N]
                    if gated is not None and gated.shape[0] > 0:
                        try:
                            t = float(gated["2T"].iloc[0]) / 2  # type: ignore
                        except ValueError:
                            t = -1
                # Find old key
                for key, vals in self.data.items():
                    for val in vals:
                        if (
                            math.isclose(unc.nominal_value(val.Ex), ex, abs_tol=0.00105)
                            and key.j == j
                        ):
                            newkey = copy.deepcopy(
                                key
                            )  # otherwise we are modifying it inplace... python :(
                            newkey.t = t
                            newdict[newkey].append(val)
        # Overwrite
        self.data = newdict
        return

    def set_max_Ex(self, maxEx: float) -> None:
        for key in list(self.data):
            kept = [v for v in self.data[key] if unc.nominal_value(v.Ex) <= maxEx]
            if kept:
                self.data[key] = kept
            else:
                del self.data[key]
        return

    def set_min_SF(self, minSF: float) -> None:
        for key in list(self.data):
            kept = [v for v in self.data[key] if unc.nominal_value(v.SF) >= minSF]
            if kept:
                self.data[key] = kept
            else:
                del self.data[key]
        return

    def set_allowed_isospin(self, t: float) -> None:
        """
        Set allowed isospin number. For backwards compatibility, set t to 0
        to avoid having to specify the t in all old code (0 is the default value in case no t is provided to Q)
        """
        self.data = {
            Orbital(k.n, k.l, k.j): vals for k, vals in self.data.items() if k.t == t
        }

        return

    def sum_strength(self, q: Orbital) -> float | unc.UFloat:
        """
        Summed strength for the given orbital
        """
        if self.data.get(q) is None:
            return 0
        if self.is_adding:  # consider SPIN FACTOR
            return sum(v.SF * (2 * v.Jpif.j + 1) / (2 * v.Jpii.j + 1) for v in self.data[q])  # type: ignore
        else:
            return sum(v.SF for v in self.data[q])  # type: ignore

    def print(self) -> None:
        print("-- Shell Model --")
        for key, vals in self.data.items():
            print(key)
            for val in vals:
                print(val)
            print("---------------")
        return

    def group_by_Ex_and_Jpif(self, tol: float = 0.001) -> GroupExJpif:
        items = sorted(((k, o) for k, lst in self.data.items() for o in lst), key=lambda o: o[1].Ex)  # type: ignore
        ret = {}
        key = None
        for k, item in items:
            jpif = item.Jpif.format_simple()
            if key is None or (
                unc.nominal_value(item.Ex) - unc.nominal_value(key[1]) > tol
            ):
                key = (jpif, item.Ex)
            elif jpif != key[0]:
                raise ValueError(
                    f"Ex={item.Ex} (name {item.Jpif.format_simple()}) falls in the group of {key[0]} at Ex={key[1]}"
                )
            ret.setdefault(key, []).append((k, item.SF))
        return ret

    @staticmethod
    def group_by_Jpif(grouped: GroupExJpif) -> GroupJpif:
        ret = {}
        for (jpif, ex), lst in grouped.items():
            ret.setdefault(jpif, []).append((ex, lst))
        return ret

    def plot_bars(
        self,
        grouped: GroupExJpif,
        ax=None,
        width: float = 0.6,
        height: float = 0.15,
        right_padding: float = 0.15,
        colors: dict = {},
    ) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        x = 0
        default_colors = {
            Orbital.from_str("0p3/2"): "slategrey",
            Orbital.from_str("0p1/2"): "green",
            Orbital.from_str("0d5/2"): "dodgerblue",
            Orbital.from_str("1s1/2"): "crimson",
            Orbital.from_str("0d3/2"): "orange",
        }
        colors = {**default_colors, **colors}
        occurrences = set()
        texts = []
        for (jpif, ex), lst in grouped.items():
            # Y position = Ex
            ex = unc.nominal_value(ex)
            # Left position
            left = x - width / 2
            # Widths!
            # 1: background
            back_w = width
            ax.barh(ex, back_w, left=left, height=height, color="lightgray", alpha=0.3)
            # 2: fraction of each component
            norm = sum(unc.nominal_value(sf) for q, sf in lst)
            aux_x = left
            for q, sf in lst:
                frac = unc.nominal_value(sf) / norm
                w = back_w * frac
                label = f"{q.format()}" if q not in occurrences else None
                ax.barh(
                    ex,
                    w,
                    left=aux_x,
                    height=height,
                    color=colors.get(q, "black"),
                    # alpha=0.8,
                    label=label,
                )
                aux_x += w
                # Append to set
                occurrences.add(q)
            # Annotate Jpif
            if right_padding != -1:
                aux = rf"{jpif[:-1]}$^{{{jpif[-1]}}}$"
                tr = ax.annotate(
                    aux,
                    xy=(left + width + right_padding, ex),
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                texts.append(tr)

        # Legend
        ax.legend(ncol=2)
        # Axis settings
        ax.set_xlim(-1, 1)
        ax.set_ylabel(r"$E_x$ [MeV]")
        return ax
