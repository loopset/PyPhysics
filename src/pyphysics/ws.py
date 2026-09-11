from .particle import Particle

import numpy as np
from typing import Union
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


class WoodSaxonOverlap:
    """
    Wood-Saxon potential for implementing a particle overlap in fresco
    (A,Z) x (a,z) with nlj quantum numbers. WS depth fitted to yield particle separation energy
    """

    def __init__(
        self,
        A: int,
        Z: int,
        a: int,
        z: int,
        n: int,
        l: int,
        j: float,
        be: float,
        s: float = 1 / 2,
    ):
        # Core
        self.A = A
        self.Z = Z
        # Bound particle
        self.a = a
        self.z = z
        self.n = n
        self.l = l
        self.j = j
        self.s = s
        self.be = be

        # Default potential settings
        # Coulomb
        self.rc: float = 1.25  # fm
        # Real
        self.V: float = -60.0  # MeV
        self.r0: float = 1.25  # fm
        self.a0: float = 0.65  # fm
        # Spin-orbit
        self.Vso: float = -6 * 2  # MeV
        self.rso: float = self.r0 - 0.15
        self.aso: float = self.a0

        # Default solver parameters
        self.dr = 0.1  # fm
        self.N = 100
        self.r: NDArray = np.arange(1, self.N + 1) * self.dr  # fm

        # Build (only once) hbar^2/2mu factor
        self.hbar2_2mu = 197.33**2 / (2 * self._reduced_mass())  # MeV fm^2
        print(
            f"Reduced mass = {self._reduced_mass():.4f} MeV/c^2, hbar^2/2mu = {self.hbar2_2mu:.4f} MeV fm^2"
        )

        # If solution
        self.eigenE: Union[float, None] = None
        self.eigenV: Union[NDArray, None] = None

    def _toR(self, r):
        """
        Convert any r radius to R = r * A^(1/3)
        """
        return r * self.A ** (1 / 3)

    def _real(self, r):
        """
        Real part
        """
        f = 1 / (1 + np.exp((r - self._toR(self.r0)) / self.a0))
        return self.V * f

    def _spin_orbit(self, r):
        """
        Spin-orbit part
        """
        f = 1 / (1 + np.exp((r - self._toR(self.rso)) / self.aso))
        # Radial derivative of f
        g = 1.0 / self.aso * np.exp((r - self._toR(self.rso)) / self.aso) * f**2
        # Expectation value of l.s (from fresco manual but also from quantum mechanics)
        ls = 0.5 * (
            self.j * (self.j + 1) - self.l * (self.l + 1) - self.s * (self.s + 1)
        )
        return self.Vso / r * g * ls

    def _coulomb(self, r):
        """
        Coulomb part
        e^2 = alpha * hbar * c ~= 1.44 MeV fm
        """
        Rc = self._toR(self.rc)
        inside = self.z * self.Z * 1.44 / Rc * (3 / 2 - r**2 / (2 * Rc**2))
        outside = self.z * self.Z * 1.44 / r
        return np.where(r <= Rc, inside, outside)

    def _potential(self, r):
        """
        Return the total potential at radius r
        """
        return self._real(r) + self._spin_orbit(r) + self._coulomb(r)

    def _reduced_mass(self):
        """
        Reduced mass of the system
        """
        core = Particle.from_numbers(self.A, self.Z)
        nucleon = Particle.from_numbers(self.a, self.z)
        mu = core.mass * nucleon.mass / (core.mass + nucleon.mass)
        return mu

    def solve_eigen(self):
        """
        Solve:
            -Cu'' + [V + C * l(l+1)/r**2]u = Eu, where C = (hbar**2/2mu)**(-1)
        using finite differences
        """
        C = self.hbar2_2mu
        Veff = self._potential(self.r) + C * (self.l * (self.l + 1)) / self.r**2
        # Diagonal terms of EDO
        diag = 2 * C / self.dr**2 + Veff
        offdiag = -C * np.ones(len(self.r) - 1) / self.dr**2

        gs, wf = eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, self.n + 1)
        )
        # print(f"Nucleon n {self.n} l {self.l} j {self.j} s {self.s} BE {self.be:.4f} MeV")
        # print(gs)
        return gs[self.n], wf[:, self.n]

    def solve_be(self):
        """
        Solve for the potential depth that yields the desired binding energy
        """

        def func(V):
            self.V = V
            gs, _ = self.solve_eigen()
            return gs - self.be

        # Fastest solver
        res = root_scalar(func, x0=self.V, x1=-40, method="secant")
        self.V = res.root
        self.eigenE, self.eigenV = self.solve_eigen()
        print(
            "Solved for V = {:.2f} MeV to yield BE = {:.4f} MeV".format(self.V, self.be)
        )

    def plot(self, ax=None):
        """
        Plot the potential
        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.r, self._potential(self.r), label="Total")
        # Plot the components
        ax.plot(self.r, self._real(self.r), "--", label="Real")
        ax.plot(self.r, self._spin_orbit(self.r), "--", label="Spin-orbit")
        ax.plot(self.r, self._coulomb(self.r), "--", label="Coulomb")
        if self.eigenE is not None and self.eigenV is not None:
            ax.axhline(
                self.eigenE, color="k", ls=":", label=f"E = {self.eigenE:.2f} MeV"
            )
            # ax.plot(self.r, self.eigenV, label="wf")
        ax.legend()
        ax.set_xlabel("r [fm]")
        ax.set_ylabel("V [MeV]")
        ax.set_xlim(0)
