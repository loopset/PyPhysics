from .particle import Particle
from .theory import QuantumNumbers

import numpy as np
from typing import Union
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


class WoodsSaxonOverlap:
    """
    Wood-Saxon potential for implementing a particle overlap reproducing the FRESCO calculation
    """

    def __init__(
        self,
        core: str,
        valence: str,
        q: Union[QuantumNumbers, str],
        be: float,
        s: float = 1 / 2,
    ):
        """Initialize Woods-Saxon overlap.

        Args:
            core: Core nucleus, as in 12Be or 19O
            valence: Valence particle identifier, as in p, n or even 3He (if not nucleon, must specify s)
            q: Quantum numbers of sp level, as str with nlj, as in 0p1/2
            be: Binding energy, typically the effective separation energy in MeV. Opposite sign convention: bound here is negative! (BE this code = -1 *  BE FRESCO)
            s: Valence-particle intrinsic spin. Default 1/2 for nucleons
        """
        ################################## Declare particles
        # Core
        self.core = Particle(core)
        self.A = self.core.A
        self.Z = self.core.Z
        # Valence
        self.valence = Particle(valence)
        self.a = self.valence.A
        self.z = self.valence.Z
        # Quantum numbers
        if isinstance(q, str):
            self.q = QuantumNumbers.from_str(q)
        else:
            self.q = q
        # Binding energy
        self.be = be
        # (Intrinsic) spin of valence particle
        self.s = s  # (by default is 1/2)

        ################################## Default potential settings
        # Coulomb
        self.rc: float = 1.25  # fm
        self.addCoulombPot: bool = (
            False  # for overlap calculations, fresco DOES NOT include the Coulomb potential (compared with FRESCO output)
        )
        # Real
        self.V: float = -60.0  # MeV
        self.r0: float = 1.25  # fm
        self.a0: float = 0.65  # fm
        # Spin-orbit
        self.Vso: float = -6 * 2 * 2  # MeV [*2 *2 required to match FRESCO formula]
        self.rso: float = self.r0 - 0.15
        self.aso: float = self.a0

        ################################## Default solver parameters
        self.dr = 0.1  # fm
        self.rmax = 50.0  # fm
        self.N = int(self.rmax / self.dr)  # number of points
        self.r: NDArray = np.arange(1, self.N + 1) * self.dr  # fm

        # Build (only once) hbar^2/2mu factor
        self.hbar2_2mu = 197.33**2 / (2 * self._reduced_mass())  # MeV fm^2

        ####################################### Solutions
        self.eigenE = np.nan
        self.eigenWF: NDArray = np.array([])  # WF = R(r), not u(r) = r*R(r)
        self.rms = np.nan

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
            self.q.j * (self.q.j + 1)
            - self.q.l * (self.q.l + 1)
            - self.s * (self.s + 1)
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
        return (
            self._real(r) + self._spin_orbit(r) + self.addCoulombPot * self._coulomb(r)
        )

    def _reduced_mass(self):
        """
        Reduced mass of the system
        """
        mu = self.core.mass * self.valence.mass / (self.core.mass + self.valence.mass)
        return mu

    def _solve_eigen(self):
        """
        Solve:
            -Cu'' + [V + C * l(l+1)/r**2]u = Eu, where C = (hbar**2/2mu)**(-1)
        using finite differences
        u(r) = r* R(r), being R(r) the radial wavefunction
        """
        C = self.hbar2_2mu
        # Effective potential
        Veff = self._potential(self.r) + C * (self.q.l * (self.q.l + 1)) / self.r**2
        # Diagonal terms of EDO
        diag = 2 * C / self.dr**2 + Veff
        offdiag = -C * np.ones(len(self.r) - 1) / self.dr**2

        es, wfs = eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, self.q.n + 1)
        )
        # print(gs)
        # Pick n
        e = es[self.q.n]
        u = wfs[:, self.q.n]
        # Convert to R aka wf
        wf = u / self.r
        # And normalise
        wf = self._normalise(wf)
        return e, wf  # Return R(r) = u(r)/r

    def _solve_be(self):
        """
        Solve for the potential depth that yields the desired binding energy
        """
        V0 = self.V

        def func(V):
            self.V = V
            e, _ = self._solve_eigen()
            return e - self.be

        # Fastest solver is secant
        res = root_scalar(func, x0=self.V, method="secant")
        self.V = res.root
        self.eigenE, self.eigenWF = self._solve_eigen()
        print(
            f"Converged ? {res.converged} for BE = {self.be:.3f} MeV, with V = {self.V:.3f} MeV and V/Vini = {self.V/V0:.3f}"
        )

    def _normalise(self, wf):
        I = np.sum(wf**2 * self.r**2) * self.dr
        return wf / np.sqrt(I)

    def _do_rms(self):
        """
        Calculate the root-mean-square radius of the wavefunction
        """
        # Second self.r**2 is from the differential volume element in spherical coordinates
        r2 = np.sum(self.r**2 * self.eigenWF**2 * self.r**2) * self.dr
        self.rms = np.sqrt(r2)

    def solve(self):
        """
        Solve for the potential depth that yields the desired binding energy
        """
        self._solve_be()
        self._do_rms()

    def plot(self):
        """
        Plot the potential
        """
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        # Potential
        ax = axs[0]
        ax.plot(self.r, self._potential(self.r), label="Total")
        # Plot the components
        # ax.plot(self.r, self._real(self.r), "--", label="Real")
        # ax.plot(self.r, self._spin_orbit(self.r), "--", label="Spin-orbit")
        # ax.plot(self.r, self._coulomb(self.r), "--", label="Coulomb")
        ax.axhline(0, color="k")
        if not np.isnan(self.eigenE):
            ax.axhline(
                self.eigenE,
                color="crimson",
                ls="--",
                label=f"E = {self.eigenE:.2f} MeV",
            )
            ax.axvline(
                self.rms, color="royalblue", ls=":", label=f"RMS = {self.rms:.2f} fm"
            )
        ax.legend()
        ax.set_xlabel("r [fm]")
        ax.set_ylabel("V [MeV]")
        ax.set_xlim(0, self.rmax * 0.5)

        # Wavefunction
        ax = axs[1]
        if not np.isnan(self.eigenE):
            ax.plot(self.r, self.eigenWF, color="dodgerblue")
            ax.axhline(0, color="k")
            ax.set_xlabel("r [fm]")
            ax.set_ylabel(r"$u(r)/r$ [$fm^{-3/2}$]")
            ax.set_xlim(0, self.rmax)

        # Figure settings
        fig.suptitle(
            f"({self.core.symbol}) x ({self.valence.symbol}) in {self.q.format_simple()}"
        )
        return fig, axs

    def set_solver_params(self, dr: float = 0.1, rmax: float = 50.0):
        """
        Set the solver parameters
        """
        self.dr = dr
        self.rmax = rmax
        self.N = int(self.rmax / self.dr)  # number of points
        self.r: NDArray = np.arange(1, self.N + 1) * self.dr  # fm

    def print_config(self):
        """
        Print the configuration of the Woods-Saxon potential
        """
        print("=" * 35)
        print(f"Core    : {self.core.symbol} (A={self.A}, Z={self.Z})")
        print(f"Valence : {self.valence.symbol} (a={self.a}, z={self.z})")
        print(f"q       : {self.q.format_simple()}")
        print(f"BE      : {self.be:.3f} MeV")
        print(f"s       : {self.s}")
        print(f"rc      : {self.rc:.3f} fm")
        print(f"V       : {self.V:.3f} MeV")
        print(f"r0      : {self.r0:.3f} fm")
        print(f"a0      : {self.a0:.3f} fm")
        print(f"Vso      : {self.Vso:.3f} MeV")
        print(f"rso      : {self.rso:.3f} fm")
        print(f"aso      : {self.aso:.3f} fm")
