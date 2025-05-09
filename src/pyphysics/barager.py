from __future__ import annotations
import pyphysics.theory as th
import uncertainties as unc
import pickle
from typing import Dict, List


class BaragerRes:
    def __init__(self) -> None:
        self.NumRem: float | unc.UFloat = 0
        self.NumAdd: float | unc.UFloat = 0
        self.DenRem: float | unc.UFloat = 0
        self.DenAdd: float | unc.UFloat = 0
        self.ESPE: float | unc.UFloat = 0

        return

    def __str__(self) -> str:
        return f"-- BaragerRes --\n N-   : {self.NumRem}\n N+   : {self.NumAdd}\n D-   : {self.DenRem}\n D+   : {self.DenAdd}\n ESPE : {self.ESPE}"

    def do_adding(
        self, q: th.QuantumNumbers, add: th.SMDataDict, sn: float, scale: float = 1
    ) -> None:
        if q in add:
            for state in add[q]:
                # Numerator
                self.NumAdd += (2 * q.j + 1) * (scale * state.SF) * (state.Ex - sn)  # type: ignore
                # Denominator
                self.DenAdd += (2 * q.j + 1) * (scale * state.SF)  # type: ignore
        return

    def do_removal(
        self, q: th.QuantumNumbers, rem: th.SMDataDict, sn: float, scale: float = 1
    ) -> None:
        if q in rem:
            for state in rem[q]:
                self.NumRem += (scale * state.SF) * (-sn - state.Ex)  # type: ignore
                self.DenRem += scale * state.SF  # type: ignore
        return

    def do_espe(self) -> None:
        try:
            self.ESPE = (self.NumAdd + self.NumRem) / (self.DenAdd + self.DenRem)  # type: ignore
        except ZeroDivisionError:
            print("Barager:do_spe() got a zero in denominator. Check your inputs")
        return


class Barager:
    def __init__(self) -> None:
        self.Rem: th.SMDataDict | None = None
        self.Add: th.SMDataDict | None = None
        self.SnRem: float = 0
        self.SnAdd: float = 0
        self.ScaleAdd: float = 1
        self.ScaleRem: float = 1
        self.Results: Dict[th.QuantumNumbers, BaragerRes] = {}
        return

    def set_removal(
        self, rem: th.ShellModel | th.SMDataDict, sn: float, scale: float = 1
    ) -> None:
        self.Rem = rem.data if isinstance(rem, th.ShellModel) else rem
        self.SnRem = sn
        self.ScaleRem = scale
        return

    def set_adding(
        self, add: th.ShellModel | th.SMDataDict, sn: float, scale: float = 1
    ) -> None:
        self.Add = add.data if isinstance(add, th.ShellModel) else add
        self.SnAdd = sn
        self.ScaleAdd = scale
        return

    def do_for(self, qs: List[th.QuantumNumbers]) -> None:
        for q in qs:
            res = BaragerRes()
            # Removal
            if self.Rem is not None:
                res.do_removal(q, self.Rem, self.SnRem, self.ScaleRem)
            # Adding
            if self.Add is not None:
                res.do_adding(q, self.Add, self.SnAdd, self.ScaleAdd)
            res.do_espe()
            # Results
            self.Results[q] = res
        return

    def get_gap(
        self, q0: th.QuantumNumbers, q1: th.QuantumNumbers
    ) -> float | unc.UFloat:
        if q0 in self.Results and q1 in self.Results:
            res = self.Results[q0].ESPE - self.Results[q1].ESPE  # type: ignore
            if unc.nominal_value(res) < 0:
                res *= -1
            return res
        return 0

    def get_ESPE(self, q: th.QuantumNumbers) -> float | unc.UFloat | None:
        res = self.Results.get(q)
        if res is None:
            return None
        return res.ESPE

    def print(self) -> None:
        print("----- Barager -----")
        for q, val in self.Results.items():
            print(q)
            print(val)
        print("--------------------")
        return

    def write(self, file: str) -> None:
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read(cls, file: str) -> Barager:
        with open(file, "rb") as f:
            return pickle.load(f)
