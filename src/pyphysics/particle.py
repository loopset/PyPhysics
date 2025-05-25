import os
from scipy.constants import physical_constants

utoMeV = physical_constants["atomic mass constant energy equivalent in MeV"][0]
emassKg = physical_constants["electron mass"][0]
emassU = emassKg / physical_constants["atomic mass constant"][0]


class Particle:
    def __init__(self, symbol: str):
        self.symbol = self._convert_names(symbol)
        self.A: int = 0
        self.Z: int = 0
        self.N: int = 0
        self.mass: float = 0
        self.mass_excess: float = 0
        # And init from symbol
        self._parse(symbol=self.symbol)
        return

    def _parse(
        self, symbol: str | None = None, A: int | None = None, Z: int | None = None
    ) -> None:
        path = os.path.join(os.path.dirname(__file__), "data", "nubase20.txt")

        with open(path, "r") as f:
            for line in f:
                try:
                    lA = int(line[0:3])
                    lZ = int(line[4:7])
                    lsymbol = line[11:16].strip()
                    isomer = int(line[7:8].strip())
                    if isomer != 0:
                        continue
                    # Symbol case
                    if symbol is not None:
                        if symbol.lower() == lsymbol.lower():
                            pass
                        else:
                            continue
                    ## (A, Z) construction
                    elif Z is not None and A is not None:
                        if Z == lZ and A == lA:
                            pass
                        else:
                            continue
                    else:
                        continue

                    # Extract info
                    self.A = lA
                    self.Z = lZ
                    self.N = self.A - self.Z
                    self.symbol = lsymbol
                    self.mass_excess = float(line[18:31])
                    self.mass = (
                        self.A * utoMeV
                        + (self.mass_excess * 1e-3)
                        - self.Z * (emassU * utoMeV)
                    )
                    return
                except Exception:
                    continue
        raise ValueError(f"Cannot find particle {self.symbol}")

    @classmethod
    def from_numbers(cls, A: int, Z: int) -> "Particle":
        obj = cls.__new__(cls)
        obj._parse(A=A, Z=Z)
        return obj

    @staticmethod
    def _convert_names(name: str) -> str:
        if name == "p":
            return "1H"
        elif name == "d":
            return "2H"
        elif name == "t":
            return "3H"
        elif name == "a":
            return "4He"
        elif name == "n":
            return "1n"
        else:
            return name

    def __str__(self) -> str:
        return f"Particle (A, Z, N) = ({self.A}, {self.Z}, {self.N})"

    def __repr__(self) -> str:
        return f"Particle (A, Z, N) = ({self.A}, {self.Z}, {self.N})"
