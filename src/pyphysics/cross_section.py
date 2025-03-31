from ctypes import ArgumentError
import numpy as np

import lmfit as lm
import uncertainties as un
from pyphysics.utils import parse_txt, create_spline3


class Comparator:
    def __init__(self, xy: np.ndarray) -> None:
        self.fExp = xy  # Experimental data
        self.fModels: dict = {}  # Model data
        self.fSplines: dict = {}  # Model splines
        self.fFitted: dict = {}  # Fitted data
        self.fFitSplines: dict = {}  # Fitted splines
        self.fSFs: dict = {}  # SF values
        return

    def add_model(self, key: str, file: str) -> None:
        data = parse_txt(file)
        self.fModels[key] = data

    def _build_model(self, key: str) -> lm.Model:
        spline = create_spline3(self.fModels[key][:, 0], self.fModels[key][:, 1])
        self.fSplines[key] = spline

        def eval(x, sf):
            return sf * spline(x)

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

    def fit(self) -> None:
        ## Weighted fit or not
        shape = self.fExp.shape[1]
        weights = (1.0 / self.fExp[:, 2] ** 2) if shape == 3 else None
        for key, _ in self.fModels.items():
            # Declare model
            model = self._build_model(key)
            res = model.fit(self.fExp[:, 1], x=self.fExp[:, 0], weights=weights, sf=1)
            # Build fitted graph
            self._build_fitted(key, res)
            # And add SF
            self.fSFs[key] = res.uvars["sf"]
            # And print!
            print(f"---- Comparator::Fit() for {key}")
            res.params.pretty_print()
        return

    def _get_key(self, key: str) -> str:
        if len(self.fFitted) == 0:
            raise ArgumentError("fitted array is empty! call fit method")
        if len(key) == 0:  # return first element, regardless of size of keys > 0
            it = next(iter(self.fFitted))
            return it
        if key in self.fFitted:
            return self.fFitted[key]
        else:
            raise ArgumentError(f"annot locate key: {key} in dict")

    def get_fitted(self, key: str = "") -> np.ndarray:
        k = self._get_key(key)
        return self.fFitted[k]

    def eval(self, x: np.ndarray, key: str = "") -> np.ndarray:
        k = self._get_key(key)
        return self.fFitSplines[k](x)

    def get_sf(self, key: str = "") -> un.UFloat:
        k = self._get_key(key)
        return self.fSFs[k]
