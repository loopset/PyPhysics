from typing import Callable, Dict, Union
import numpy as np
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt

import lmfit as lm
import uncertainties as un
from pyphysics.utils import parse_txt, create_spline3


class Comparator:
    def __init__(self, xy: np.ndarray) -> None:
        self.fExp = xy  # Experimental data
        self.fModels: Dict[str, np.ndarray] = {}  # Model data
        self.fSplines: Dict[str, Callable] = {}  # Model splines
        self.fFitted: Dict[str, np.ndarray] = {}  # Fitted data
        self.fFitSplines: Dict[str, Callable] = {}  # Fitted splines
        self.fSFs: Dict[str, Union[float, un.UFloat]] = {}  # SF values
        return

    def add_models(self, files: Dict[str, str]) -> None:
        for key, file in files.items():
            self.add_model(key, file)
        ## and why not call fit directly!
        self.fit()
        return

    def add_model(self, key: str, file: str) -> None:
        data = parse_txt(file)
        self.fModels[key] = data
        ## And parse theoretical data
        self._parse_theoretical(key)

    def _parse_theoretical(self, key: str) -> None:
        """
        This function parses the model data and builds and spline3 with knot data
        integrated in the experimental bin width
        """
        ## Initial spline with X axis in radians!
        xtheo = np.deg2rad(self.fModels[key][:, 0])
        spline = create_spline3(xtheo, self.fModels[key][:, 1])
        ## Theoretical boundaries
        tMin, tMax = xtheo.min(), xtheo.max()
        ## Experimental settings
        eMin, _ = self.fExp[:, 0].min(), self.fExp[:, 0].max()
        if len(self.fExp[:, 0]) > 1:
            eBW = self.fExp[1, 0] - self.fExp[0, 0]
        else:
            eBW = 1
        # Convert to radians
        eMin = np.deg2rad(eMin)
        eBW = np.deg2rad(eBW)
        # Find starting point
        start = 0
        for x in np.arange(eMin, tMin, -eBW):
            start = x
        # And build binned theoretical graph!
        binnedx = []
        binnedy = []
        for x in np.arange(start, tMax, eBW):
            # Boundaries in rads!
            low = x - eBW / 2
            up = x + eBW / 2
            integral = spline.integrate(low, up) / eBW
            binnedx.append(np.rad2deg(x))
            binnedy.append(integral)
        # And replace contents with this
        self.fModels[key] = np.column_stack((binnedx, binnedy))
        # And build spline
        self.fSplines[key] = create_spline3(
            self.fModels[key][:, 0], self.fModels[key][:, 1]
        )
        return

    def _build_model(self, key: str) -> lm.Model:
        # Model to eval
        def eval(x, sf):
            return sf * self.fSplines[key](x)

        model = lm.Model(eval)
        return model

    def _build_fitted(self, key: str, res) -> None:
        def eval_fit(x):
            return res.params["sf"].value * self.fSplines[key](x)

        self.fFitSplines[key] = eval_fit

        # X same as theoretical input
        x = self.fModels[key][:, 0]
        yeval = eval_fit(x)
        self.fFitted[key] = np.column_stack((x, yeval))
        return

    def fit(self, show: bool = False, scale_covar=False) -> None:
        ## Weighted fit or not
        shape = self.fExp.shape[1]
        # INFO: weights are ^2 in the built-in chi2, because it does
        # ((exp - fit) * weight)^2 !
        weights = (1.0 / self.fExp[:, 2]) if shape == 3 else None
        # print(weights)
        for key, _ in self.fModels.items():
            # Declare model
            model = self._build_model(key)
            # INFO: pass scale_cover=False to have reliable error estimation
            # With this, results are exactly as ROOT's
            res = model.fit(
                self.fExp[:, 1],
                x=self.fExp[:, 0],
                weights=weights,
                sf=1,
                scale_covar=scale_covar,
            )
            # Build fitted graph
            self._build_fitted(key, res)
            # And add SF
            if hasattr(res, "uvars"):
                self.fSFs[key] = res.uvars["sf"]
            else:
                self.fSFs[key] = res.params["sf"].value
            # And print!
            if show:
                print(f"---- Comparator::Fit() for {key}")
                res.params.pretty_print()
        return

    def _get_key(self, key: str) -> str:
        if len(self.fFitted) == 0:
            raise ValueError("fitted array is empty! call fit method")
        if len(key) == 0:  # return first element, regardless of size of keys > 0
            it = next(iter(self.fFitted))
            return it
        if key in self.fFitted:
            return key
        else:
            raise ValueError(f"annot locate key: {key} in dict")

    def get_fitted(self, key: str = "") -> np.ndarray:
        k = self._get_key(key)
        return self.fFitted[k]

    def eval(self, x: np.ndarray, key: str = "") -> np.ndarray:
        k = self._get_key(key)
        return self.fFitSplines[k](x)

    def get_sf(self, key: str = "") -> float | un.UFloat:
        k = self._get_key(key)
        return self.fSFs[k]

    def draw(self, ax: mplaxes.Axes | None = None, title: str | None = None) -> None:
        if ax is None:
            ax = plt.gca()
        ax.errorbar(
            self.fExp[:, 0],
            self.fExp[:, 1],
            yerr=self.fExp[:, 2] if self.fExp.shape[1] == 3 else None,
            marker="s",
            ls="none",
            label="Exp.",
        )
        if len(self.fFitted):
            for key, fit in self.fFitted.items():
                value = self.fSFs[key]
                if isinstance(value, un.UFloat):
                    strval = f"{value:.2uS}"
                else:
                    strval = f"{value:.2f}"
                label = rf"{key} $\Rightarrow$ SF = {strval}"
                finex = np.linspace(fit[:, 0].min(), fit[:, 0].max(), len(fit) * 4)
                ax.plot(
                    finex,
                    self.fFitSplines[key](finex),
                    marker="none",
                    lw=2,
                    label=label,
                )
        ncols = int((len(self.fFitted) + 1) / 7) + 1
        ax.legend(
            fontsize=12 - (ncols - 1) * 2,
            ncol=ncols,
            frameon=True,
            framealpha=0.75,
            edgecolor="w",
            shadow=False,
        )
        # Ranges
        xmin = np.min(self.fExp[:, 0])
        xmax = np.max(self.fExp[:, 0])
        xoff = 3
        ax.set_xlim(max(0, xmin - xoff), xmax + xoff)
        # Titles
        ax.set_xlabel(r"$\theta_{\mathrm{CM}}$ [$^{\circ}$]", fontsize=16)
        ax.set_ylabel(r"d$\sigma$/d$\Omega$ [mb/sr]", fontsize=16)
        if title is not None:
            ax.set_title(title, fontsize=18)
