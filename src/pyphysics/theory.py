from collections import defaultdict
import uncertainties as unc
import pandas as pd
from fractions import Fraction
import re
import math
import copy
from typing import Dict, List


class QuantumNumbers:
    """
    A class representing the (nlj)
    quantum numbers that identify a state
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
        if not isinstance(other, QuantumNumbers):
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
            ret = rf"{self.n}{QuantumNumbers.letters[self.l]}$_{{{frac}}}$"
        else:  # Summary mode: no info on l nor n. n is "state counter" and j = parity * j
            frac = Fraction(abs(self.j)).limit_denominator()
            pi = "+" if self.j > 0 else "-"
            ret = rf"${frac}^{{{pi}}}_{{{self.n}}}$"
        return ret

    def format_simple(self) -> str:
        if self.l >= 0:
            frac = Fraction(self.j).limit_denominator()
            ret = f"{self.n}{QuantumNumbers.letters[self.l]}{frac}"
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


class ShellModelData:
    """
    A class containing the Ex and SF data from a shell-model calculation
    """

    def __init__(self, ex: float | unc.UFloat, sf: float | unc.UFloat) -> None:
        self.Ex = ex
        self.SF = sf
        return

    def __str__(self) -> str:
        return f"Data:\n  Ex : {self.Ex:.2f}\n  SF : {self.SF:.2f}"

    def __repr__(self) -> str:
        return f"SMData(Ex: {self.Ex:.2f}, SF: {self.SF:.2f})"


# Alias
SMDataDict = Dict[QuantumNumbers, List[ShellModelData]]


class ShellModel:
    def __init__(self, files: list = []) -> None:
        self.data: SMDataDict = {}

        if len(files):
            self.__buildFromFiles(files)
        return

    def __buildFromFiles(self, files: list) -> None:
        # Parse each file
        for file in files:
            input = self.__parse(file)
            self.data.update(input)

        # Determine binding energy
        maxSF = max(
            [s for sublist in self.data.values() for s in sublist],
            key=lambda sm: unc.nominal_value(sm.SF),
        )
        # print(maxSF)
        self.BE = maxSF.Ex
        # And substract it from states
        for _, sublist in self.data.items():
            for state in sublist:
                state.Ex = state.Ex - self.BE  # type: ignore
                state.Ex = round(state.Ex, 3)
        return

    def __parse(self, file: str) -> dict:
        ret = {}
        with open(file, "r") as f:
            n, l, j = -1, -1, -1
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
                if re.match(
                    r"^\d+\(", line
                ):  # States start with 2*Jf(. This is their clear signature
                    ex = float(line[35:41].strip())
                    c2s = float(line[45:51].strip())
                    # Define key
                    q = QuantumNumbers(n, l, j / 2)
                    # Define values
                    sm = ShellModelData(ex, c2s)
                    # Push to dict
                    if q not in ret:
                        ret[q] = [sm]
                    else:
                        ret[q].append(sm)
        return ret

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
                key = QuantumNumbers(count, -1, pi * j, t)
                val = ShellModelData(ex, -1)
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
        for key, vals in self.data.items():
            newlist = []
            for val in vals:
                if unc.nominal_value(val.Ex) <= maxEx:
                    newlist.append(val)
            self.data[key] = newlist
        return

    def set_min_SF(self, minSF: float) -> None:
        for key, vals in self.data.items():
            newlist = []
            for val in vals:
                if unc.nominal_value(val.SF) >= minSF:
                    newlist.append(val)
            self.data[key] = newlist
        return

    def set_allowed_isospin(self, t: float) -> None:
        """
        Set allowed isospin number. For backwards compatibility, set t to 0
        to avoid having to specify the t in all old code (0 is the default value in case no t is provided to Q)
        """
        self.data = {
            QuantumNumbers(k.n, k.l, k.j): vals
            for k, vals in self.data.items()
            if k.t == t
        }

        return

    def sum_strength(self, q: QuantumNumbers) -> float | unc.UFloat:
        """
        Summed strength for the given quantum number
        """
        if self.data.get(q) is None:
            return 0
        return sum(pair.SF for pair in self.data[q])  # type: ignore

    def print(self) -> None:
        print("-- Shell Model --")
        for key, vals in self.data.items():
            print(key)
            for val in vals:
                print(val)
            print("---------------")
        return
