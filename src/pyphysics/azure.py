from pathlib import Path
import re
import uuid
import lmfit
from brick import AZR
from brick.output import Output
import numpy as np
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional
from types import SimpleNamespace
import numpy as np
import pandas as pd
import uncertainties as un
import uncertainties.unumpy as unp


class AZUREInterface:
    def __init__(self, file: str):
        self.file: Path = Path(file).resolve()
        self.energy_range: tuple[float, float, float] | None = None
        self.angle: float | None = None
        self.level_energy: float | None = None
        self.level_width: float | None = None
        self.segment_data_file: Path | None = None
        self.copied_file: Path | None = None
        self.azure: AZR | None = None
        self.results = None

    def set_parameters(
        self,
        energy_range: tuple[float, float, float],
        angle: float,
        level_energy: float,
        level_width: float,
    ) -> Path:
        energy_min, energy_max, energy_step = energy_range
        if energy_step <= 0 or energy_max < energy_min:
            raise ValueError("energy_range must have positive step and emax >= emin")

        self.energy_range = energy_range
        self.angle = angle
        self.level_energy = level_energy
        self.level_width = level_width

        token = uuid.uuid4().hex
        self.segment_data_file = self.file.with_name(f"{self.file.stem}_{token}.dat")
        self.copied_file = self.file.with_name(f"{self.file.stem}_{token}.azr")

        try:
            self._write_toy_segment_data()
            self._modify_first_level()
        except Exception:
            self.erase_copy()
            raise

        copied_file = self.copied_file
        if copied_file is None:
            raise RuntimeError("Copied AZURE file was not created")
        return copied_file

    def erase_copy(self) -> None:
        self._remove_file(self.copied_file)
        self._remove_file(self.segment_data_file)
        self.copied_file = None
        self.segment_data_file = None

    @staticmethod
    def _remove_file(path: Path | None) -> None:
        if path is not None and path.exists():
            path.unlink()

    def _write_toy_segment_data(self) -> None:
        energy_range = self.energy_range
        angle = self.angle
        segment_data_file = self.segment_data_file
        copied_file = self.copied_file
        if (
            energy_range is None
            or angle is None
            or segment_data_file is None
            or copied_file is None
        ):
            raise RuntimeError("Call set_parameters before writing segment data")

        energy_min, energy_max, energy_step = energy_range
        energies = np.arange(
            energy_min,
            energy_max + energy_step / 2,
            energy_step,
        )

        with segment_data_file.open("w", encoding="utf-8") as file:
            for energy in energies:
                file.write(f"{energy:.8g}\t{angle:g}\t1.0\t0.1\n")

        text = self.file.read_text(encoding="utf-8")
        match = re.search(
            r"(<segmentsData>\s*\n)(.*?)(\n\s*</segmentsData>)",
            text,
            re.DOTALL,
        )
        if not match:
            raise ValueError("Missing <segmentsData> block")

        line = match.group(2)
        fields = list(re.finditer(r"\S+", line))
        if len(fields) < 12:
            raise ValueError("segmentsData does not contain the expected fields")

        replacements = [
            (fields[3].span(), f"{energy_min:g}"),
            (fields[4].span(), f"{energy_max:g}"),
            (fields[5].span(), f"{angle:g}"),
            (fields[6].span(), f"{angle:g}"),
            (fields[-1].span(), str(segment_data_file)),
        ]
        for (start, end), value in reversed(replacements):
            line = line[:start] + value + line[end:]

        replacement = match.group(1) + line + match.group(3)
        copied_file.write_text(
            text[: match.start()] + replacement + text[match.end() :],
            encoding="utf-8",
        )

    def _modify_first_level(self) -> None:
        copied_file = self.copied_file
        level_energy = self.level_energy
        level_width = self.level_width
        if copied_file is None or level_energy is None or level_width is None:
            raise RuntimeError("Call set_parameters before modifying levels")

        text = copied_file.read_text(encoding="utf-8")
        match = re.search(
            r"(<levels>\s*\n)(.*?)(\n\s*</levels>)",
            text,
            re.DOTALL,
        )
        if not match:
            raise ValueError("Missing <levels> block")

        first_level = re.search(r"^[ \t]*\S.*$", match.group(2), re.MULTILINE)
        if not first_level:
            raise ValueError("Missing first level")

        line = first_level.group(0)
        fields = list(re.finditer(r"\S+", line))
        if len(fields) <= 11:
            raise ValueError("First level does not contain energy and width")

        replacements = [
            (fields[2].span(), f"{level_energy:g}"),
            (fields[11].span(), f"{level_width:g}"),
        ]
        for (start, end), value in reversed(replacements):
            line = line[:start] + value + line[end:]

        block = match.group(2)
        start, end = first_level.span()
        updated_block = block[:start] + line + block[end:]
        copied_file.write_text(
            text[: match.start(2)] + updated_block + text[match.end(2) :],
            encoding="utf-8",
        )

    def call_azure(self) -> None:
        copied_file = self.copied_file
        if copied_file is None:
            raise RuntimeError("Call set_parameters before calling AZURE")

        self.azure = AZR(copied_file)
        # print(f"Running azure in file {copied_file}")
        theta = self.azure.config.get_input_values()
        try:
            self.results = self.azure.predict(theta)[0]
        except Exception as error:
            raise RuntimeError("Azure prediction failed") from error


@dataclass
class Parameter:
    """An AZURE2 parameter with bounds"""

    name: str
    value: float
    fixed: bool = False
    lower: float = -np.inf
    upper: float = np.inf
    safe_name: str = field(init=False)

    def __post_init__(self) -> None:
        # '$\Gamma_{1,1}^{(+2.5)}$' -> 'Gamma_(5/2+)_1_1'
        # '$E_{1,1}^{(+2.5)}$'      -> 'E_(5/2+)_1_1'
        m = re.fullmatch(r"\$\\?(\w+)_\{(.*?)\}\^\{\(([+-]?)([\d.]+)\)\}\$", self.name)
        if m:
            sym, sub, sign, num = m.groups()
            f = Fraction(num).limit_denominator(100)
            frac = (
                str(f.numerator)
                if f.denominator == 1
                else f"{f.numerator}/{f.denominator}"
            )
            self.name = f"{sym}_({frac}{sign})_{sub.replace(',', '_')}"
        else:
            # '$n_{1}$' -> 'n_1'
            self.name = re.sub(r"[${}()]", "", self.name)

        # lmfit-safe: 'Gamma_(5/2+)_1_1' -> 'Gamma_5_2p_1_1'
        s = (
            re.sub(r"[()]", "", self.name)
            .replace("/", "_")
            .replace("+", "p")
            .replace("-", "m")
        )
        self.safe_name = re.sub(r"\W", "_", s)

    def set_auto_bounds(self) -> None:
        """Automatically sets bounds depending on the parameter type"""
        if "n_" in self.name:
            self.lower = 0.5
            self.upper = 1.5
        if "Gamma" in self.name:
            self.lower = 0
            self.upper = 1e8

    def __str__(self):
        return f"{self.name} = {self.value} in [{self.lower}, {self.upper}]"


class Chi2Minimizer:
    def __init__(self, azr: AZR, pars: List[Parameter], n_workers: int = 1) -> None:
        self.azr = azr
        self.pars = pars
        self.n_workers = n_workers
        self.mini = None
        self.res = None

        # Disable AZURE2 MT
        import os

        os.environ["OMP_NUM_THREADS"] = "1"

    def residuals(self, pars):
        """Compute residuals (y - mu) / uy"""
        pred = self.azr.predict(pars)[0]
        mu = pred.xs_com_fit  # type: ignore
        y = pred.xs_com_data  # type: ignore
        uy = pred.xs_err_com_data  # type: ignore
        return (y - mu) / uy

    def fit(self) -> None:
        """ "Chi2 minimisation using least_squares in lmfit"""
        # Define parameters
        pars = lmfit.Parameters()
        for p in self.pars:
            pars.add(
                p.safe_name, value=p.value, min=p.lower, max=p.upper, vary=not p.fixed
            )

        # auxiliar function to pass arguments
        def call_residuals(params):
            theta = [params[p.safe_name].value for p in self.pars]
            return self.residuals(theta)

        # build minimizer
        self.mini = lmfit.Minimizer(call_residuals, pars, scale_covar=False)
        # and call it!
        out = self.mini.minimize(
            method="least_squares",
            x_scale="jac",
            diff_step=1e-4,
            max_nfev=2000,
            verbose=0,
        )
        print(lmfit.fit_report(out))
        # create results object
        self.res = SimpleNamespace(
            x=np.array([out.uvars[p.safe_name] for p in self.pars]),  # type: ignore
            chi2=out.chisqr,
            success=out.success,  # type: ignore
            message=out.message,  # type: ignore
        )
        if not self.res.success:
            print(f"Warning: minimization did not fully converge: {self.res.message}")

    def summary(self) -> pd.DataFrame:
        """Return fitted parameters as DataFrame."""
        if self.res is None:
            raise ValueError("Run .fit() first")
        return pd.DataFrame(
            {
                "Parameter": [p.name for p in self.pars],
                "Initial": [p.value for p in self.pars],
                "Value": self.res.x,
                "Fixed": [p.fixed for p in self.pars],
                "Lower Bound": [p.lower for p in self.pars],
                "Upper Bound": [p.upper for p in self.pars],
            }
        )

    def get_params(self) -> Dict[str, float]:
        """Get fitted parameters as dict"""
        if self.res is None:
            raise ValueError("Run .fit() first")
        return {p.name: val for p, val in zip(self.pars, self.res.x)}

    def get_best_fit(self) -> Output | None:
        """Get best fit prediction"""
        if self.res is None:
            return None
        return self.azr.predict(unp.nominal_values(self.res.x))[0]  # type: ignore
