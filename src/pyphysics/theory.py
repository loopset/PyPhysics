import uncertainties as unc
from fractions import Fraction
import re
from typing import Dict, List


class QuantumNumbers:
    """
    A class representing the (nlj)
    quantum numbers that identify a state
    """

    letters = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}

    def __init__(self, n: int, l: int, j: float) -> None:
        self.n = n
        self.l = l
        self.j = j
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
        return self.n == other.n and self.l == other.l and self.j == other.j

    def __hash__(self) -> int:
        return hash((self.n, self.l, self.j))

    def __str__(self) -> str:
        return f"Quantum number:\n n : {self.n}\n l : {self.l}\n j : {self.j}"

    def __repr__(self) -> str:
        return f"nlj:({self.n},{self.l},{self.j})"

    def format(self) -> str:
        frac = Fraction(self.j).limit_denominator()
        ret = rf"{self.n}{QuantumNumbers.letters[self.l]}$_{{{frac}}}$"
        return ret

    def format_simple(self) -> str:
        frac = Fraction(self.j).limit_denominator()
        ret = f"{self.n}{QuantumNumbers.letters[self.l]}{frac}"
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
                if "0(" in line:
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
