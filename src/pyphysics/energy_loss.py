import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from typing import Dict, List


class EnergyLoss:
    """
    A class that reads SRIM  files and performs interpolations.
    TODO: add other input files
    """

    def __init__(self, cubic: bool = True) -> None:
        self.keys: List[str] = []
        self.sp_direct: Dict[str, InterpolatedUnivariateSpline] = {}
        self.sp_inverse: Dict[str, InterpolatedUnivariateSpline] = {}
        self.sp_stopping: Dict[str, InterpolatedUnivariateSpline] = {}
        self.sp_longstragg: Dict[str, InterpolatedUnivariateSpline] = {}
        self.sp_latstragg: Dict[str, InterpolatedUnivariateSpline] = {}
        self._k: int = 3 if cubic else 1
        return

    @staticmethod
    def _convert_to_float(value: str, unit: str) -> float:
        """Convert (value, unit) str to value float in MeV and mm units"""
        value = value.replace(",", ".")
        units = {
            "eV": 1e-6,
            "keV": 1e-3,
            "MeV": 1,
            "GeV": 1e3,
            "A": 1e-7,
            "um": 1e-3,
            "mm": 1,
            "cm": 1e1,
            "m": 1e3,
            "km": 1e6,
            "None": 1,
        }
        try:
            factor = units[unit]
        except KeyError:
            raise ValueError(f"Unknown unit: {unit}")
        return float(value) * factor

    @staticmethod
    def _is_break_line(line: str) -> bool:
        return line.count("-") + line.count("=") > 6

    def read(self, key: str, file: str) -> None:
        """
        Function that parses a SRIM.txt file in MeV/mm units
        """
        es, stopps, ranges, longs, lats = [], [], [], [], []
        reading = False
        with open(file, "r") as f:
            for line in f:
                if self._is_break_line(line):
                    continue
                if "Straggling" in line:
                    reading = True
                    continue
                if "Multiply" in line:
                    break
                if reading:
                    tokens = line.split()
                    if len(tokens) < 10:
                        continue
                    e, ue, electro, nucl, r, ur, ls, uls, as_, uas = tokens[:10]
                    es.append(self._convert_to_float(e, ue))
                    stopps.append(
                        self._convert_to_float(electro, "None")
                        + self._convert_to_float(nucl, "None")
                    )
                    ranges.append(self._convert_to_float(r, ur))
                    longs.append(self._convert_to_float(ls, uls))
                    lats.append(self._convert_to_float(as_, uas))
        # And init splines
        self.sp_direct[key] = InterpolatedUnivariateSpline(es, ranges, k=self._k)
        self.sp_inverse[key] = InterpolatedUnivariateSpline(ranges, es, k=self._k)
        self.sp_stopping[key] = InterpolatedUnivariateSpline(es, stopps, k=self._k)
        self.sp_longstragg[key] = InterpolatedUnivariateSpline(ranges, longs, k=self._k)
        self.sp_latstragg[key] = InterpolatedUnivariateSpline(ranges, lats, k=self._k)
        self.keys.append(key)
        return

    def eval_range(self, key: str, energy: float) -> float:
        """
        Get range given an energy
        """
        return self.sp_direct[key](energy).item()  # type: ignore

    def eval_energy(self, key: str, range: float) -> float:
        """
        Get energy given a range
        """
        return self.sp_inverse[key](range).item()  # type: ignore

    def eval_longstragg(self, key: str, range_: float) -> float:
        """
        Get longitudinal straggling for given range
        """
        return self.sp_longstragg[key](range_).item()  # type: ignore

    def slow(self, key: str, Tini: float, thick: float, angle: float = 0) -> float:
        """Propagates a particle in a medium a given length"""
        Rini = self.eval_range(key, Tini)
        dist = thick / np.cos(angle)
        Rafter = Rini - dist
        if Rafter <= 0:
            return 0
        Tafter = self.eval_energy(key, Rafter)
        if Tafter > Tini:
            return Tini
        return Tafter

    def slow_with_straggling(
        self, key: str, Tini: float, thick: float, angle: float = 0
    ) -> float:
        """Propagates a particle in a medium a given length but implementing straggling!"""
        Rini = self.eval_range(key, Tini)
        dist = thick / np.cos(angle)
        uRini = self.eval_longstragg(key, Rini)
        Rafter = Rini - dist
        if Rafter <= 0:
            return 0
        uRafter = self.eval_longstragg(key, Rafter)
        udist = np.sqrt(uRini**2 - uRafter**2)
        dist = np.random.normal(dist, udist)
        Rafter = Rini - dist
        if Rafter <= 0:
            return 0
        Tafter = self.eval_energy(key, Rafter)
        if Tafter > Tini:
            return Tini
        return Tafter

    def eval_initial_energy(
        self, key: str, Tafter: float, thick: float, angle: float = 0
    ) -> float:
        """
        reconstruct initial energy of particle before propagating a distance
        """
        Rafter = self.eval_range(key, Tafter)
        dist = thick / np.cos(angle)
        Rini = Rafter + dist
        return self.eval_energy(key, Rini)

    def draw(self) -> None:
        for key in self.keys:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"Energy losses for {key}", fontsize=16)

            # X axes
            e_knots = self.sp_direct[key].get_knots()
            r_knots = self.sp_inverse[key].get_knots()
            es = np.linspace(e_knots[0], e_knots[-1], 300)
            rs = np.linspace(r_knots[0], r_knots[-1], 300)

            # Direct: range vs energy
            axs[0, 0].plot(es, self.sp_direct[key](es))
            axs[0, 0].set_xlabel("Energy [MeV]")
            axs[0, 0].set_ylabel("Range [mm]")
            axs[0, 0].set_title("Range vs Energy")

            # Inverse: energy vs range
            axs[0, 1].plot(rs, self.sp_inverse[key](rs))
            axs[0, 1].set_xlabel("Range [mm]")
            axs[0, 1].set_ylabel("Energy [MeV]")
            axs[0, 1].set_title("Energy vs Range")

            # Stopping: stopping power vs energy
            axs[1, 0].plot(es, self.sp_stopping[key](es))
            axs[1, 0].set_xlabel("Energy [MeV]")
            axs[1, 0].set_ylabel("Stopping Power [MeV/mm]")
            axs[1, 0].set_title("Stopping Power vs Energy")

            # Longitudinal straggling: longstragg vs range
            axs[1, 1].plot(rs, self.sp_longstragg[key](rs))
            axs[1, 1].set_xlabel("Range [mm]")
            axs[1, 1].set_ylabel("Long. Straggling [mm]")
            axs[1, 1].set_title("Longitudinal Straggling vs Range")

            plt.tight_layout()
        plt.show()
