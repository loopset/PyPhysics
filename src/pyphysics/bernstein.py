from pyphysics.particle import Particle
from scipy.constants import physical_constants
import uncertainties as un
import uncertainties.umath as umath
import math


class Radii:
    def __init__(self, p: Particle):
        self.fPart: Particle = p
        if self.fPart.A >= 40:
            raise ValueError(
                "Radii implementation only works for A < 40. Check the paper for A >= 40 parametrization"
            )
        self.fRn: float = self._set_neutron()
        self.fRp: float = self._set_proton()
        return

    def _eval_radius(self, r0, r1, r2, r3) -> float:
        a = self.fPart.A
        eps = (self.fPart.N - self.fPart.Z) / self.fPart.A
        return r0 * a ** (1 / 3) + r1 + r2 * eps + r3 * eps**2

    def _set_neutron(self) -> float:
        r0 = 1.02
        r1 = 0.75
        r2 = 0.46
        r3 = 1.08
        return self._eval_radius(r0, r1, r2, r3)

    def _set_proton(self) -> float:
        r0 = 1.03
        r1 = 0.79
        r2 = -1.07
        r3 = 0.82
        return self._eval_radius(r0, r1, r2, r3)

    def __str__(self):
        return f"-- Radii :\n  N : {self.fRn}\n  P : {self.fRp}"


class Diffuseness:
    def __init__(self, p: Particle):
        self.fPart = p
        if self.fPart.N >= 28 or self.fPart.Z >= 28:
            raise ValueError(
                "Check eta parameter in _eval_s. Read the paper for further instructions"
            )
        self.fan = self._set_neutron()
        self.fap = self._set_proton()
        return

    def _eval_s(self, type: str, alpha, s1, s2, s3, s4, s5, s6) -> float:
        a = self.fPart.A
        n = self.fPart.N
        z = self.fPart.Z

        ## Eta parameter
        if type != "n" and type != "p":
            raise ValueError("type for eta infer must be n or p")
        value = n if type == "n" else z
        if value < 29:
            eta = 0
        elif 29 <= value <= 50:
            eta = 1
        elif 51 <= value <= 82:
            eta = 2
        else:
            raise ValueError("Reached limit of implemented eta values")

        return (
            (s1 + s2 * a ** (1 / 3)) * (n / z) ** alpha
            - s3
            + s4 / a ** (1 / 2)
            - eta * s5
            - s6 * z / a ** (1 / 3)
        )

    def _eval_a(self, mass, s) -> float:
        hbarc = 197.32698  # MeV fm
        return hbarc / (2 * (2 * mass * s) ** (1 / 2))

    def _set_neutron(self):
        s1 = 6.29
        s2 = 3.43
        s3 = 5.85
        s4 = 10.59
        s5 = 1.51
        s6 = 0
        alpha = -1
        # in case of odd number of neutrons
        if not (self.fPart.N % 2 == 0):
            s4 *= -1
        sn = self._eval_s("n", alpha, s1, s2, s3, s4, s5, s6)
        mass = physical_constants["neutron mass energy equivalent in MeV"][0]
        return self._eval_a(mass, sn)

    def _set_proton(self):
        s1 = 13.83
        s2 = 0.64
        s3 = 3.56
        s4 = 12.14
        s5 = 1.32
        s6 = 0.98
        alpha = 1
        # in case of odd number of protons
        if not (self.fPart.Z % 2 == 0):
            s4 *= -1
        sp = self._eval_s("p", alpha, s1, s2, s3, s4, s5, s6)
        mass = physical_constants["proton mass energy equivalent in MeV"][0]
        return self._eval_a(mass, sp)

    def __str__(self):
        return f"-- Difuseness :\n   N : {self.fan}\n  P : {self.fap}"


class Bernstein:
    def __init__(
        self,
        p: Particle,
        dnuclear: float | un.UFloat,
        dem: float | un.UFloat,
        bpbn: float = 1,
    ) -> None:
        self.fPart = p
        self.fRadii = Radii(self.fPart)
        self.fDiffu = Diffuseness(self.fPart)
        self.a = 0.7  # fm
        self.fbpbn = bpbn
        self.fDefNuclear = dnuclear
        self.fDefEm = dem
        self.fMnMp = self._do()
        return

    def _do(self) -> float:
        ret = (
            self.fbpbn
            * (self.fDiffu.fan / self.fDiffu.fap)
            * (self.fRadii.fRp / self.fRadii.fRn)
        )
        ret *= (
            self.fDefNuclear  # type: ignore
            / self.fDefEm
            * self.fDiffu.fap
            / self.a
            * (1 + self.fPart.N / self.fPart.Z * 1 / self.fbpbn)
            - 1
        )
        return ret

    def print(self) -> None:
        print("===== Bernstein calculation =====")
        print(self.fRadii)
        print(self.fDiffu)
        print(
            f"-- Deformations :\n  Nuclear : {self.fDefNuclear}\n  EM : {self.fDefEm}"
        )
        print(f"-- Parameters :\n  bp / bn : {self.fbpbn}")
        print(self)
        print("==============================")
        return

    def __str__(self):
        return f"-- Bernstein :\n  Mn / Mp : {self.fMnMp}"


def BE_to_beta(
    be: float | un.UFloat, p: Particle, l: float, isUp: bool = True, lgs: float = 0
) -> float | un.UFloat:
    """
    This function converts a B(EL) to betaL
    """
    r = 1.2 * p.A ** (1.0 / 3)
    if not isUp:
        # Multiply by spin factor, assuming decay is to ground state!
        factor = (2 * l + 1) / (2 * lgs + 1)
        be *= factor  # type: ignore
    beta = 4 * math.pi / (3 * p.fZ * r**l) * umath.sqrt(be)  # type: ignore
    return beta


def simple_bernstein(
    em: float | un.UFloat, nucl: float | un.UFloat, p: Particle, bpbn: float = 1
) -> float | un.UFloat:
    """
    Simple and original Bersntein formula
    """
    ratio = bpbn * (nucl / em * (1 + 1.0 / bpbn * p.fN / p.fZ) - 1)  # type: ignore
    return ratio
