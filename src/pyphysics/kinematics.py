from typing import Tuple
from pyphysics.particle import Particle
import vector as vec
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class KinRet:
    """
    A simple struct to hold results from a Kinematics calculation

    All results are in LAB reference frame
    """

    T3: float = -1
    theta3: float = -1
    phi3: float = -1
    T4: float = -1
    theta4: float = -1
    phi4: float = -1


class Kinematics:
    def __init__(self, reaction: str) -> None:
        self.reactstr = reaction
        # Parse reaction
        p1, p2, p3, T1, Ex = self._parse_reaction(reaction)
        # Init particles
        self._part1 = Particle(p1)
        self._part2 = Particle(p2)
        self._part3 = Particle(p3)
        self._isInverse = self._part1.mass > self._part2.mass
        # Automatically compute 4th particle
        ain = self._part1.A + self._part2.A
        zin = self._part1.Z + self._part2.Z
        self._part4 = Particle.from_numbers(ain - self._part3.A, zin - self._part3.Z)
        # And assign beam energies and Ex
        self.T1: float = T1
        self.Ex: float = Ex
        # Reaction values
        self.Qvalue: float = 0
        self.T1thresh: float = 0
        # Lorentz vectors
        self.beta: float = 0
        self.PIniLab: vec.MomentumObject4D = vec.obj(px=0, py=0, pz=0, e=0)
        self.PIniCM: vec.MomentumObject4D = vec.obj(px=0, py=0, pz=0, e=0)
        # Calculation results
        self.ret: KinRet = KinRet()

        ## Main method
        self._init()
        return

    @staticmethod
    def _parse_reaction(reaction: str) -> tuple:
        A, rest = reaction.split("(", 1)
        B, rest = rest.split(",", 1)
        C, rest = rest.split(")", 1)
        rest = rest.lstrip("@")
        values = rest.split("|")
        value0 = values[0] if values else 0
        value1 = values[1] if len(values) > 1 else 0
        return (A, B, C, float(value0), float(value1))

    def _init(self) -> None:
        ## 1 Q value
        self.Qvalue = (
            self._part1.mass
            + self._part2.mass
            - (self._part3.mass + (self._part4.mass + self.Ex))
        )
        ## 2 T1 threshold
        if self.Qvalue < 0:
            self.T1thresh = (
                -self.Qvalue
                * (
                    self._part1.mass
                    + self._part2.mass
                    + self._part3.mass
                    + (self._part4.mass + self.Ex)
                )
                / (2 * self._part2.mass)
            )
        ## 3 Check energy threshold
        if self.Qvalue < 0:
            if self.T1 < self.T1thresh:
                raise ValueError(
                    f"T1 = {self.T1:.2f} below threshold of {self.T1thresh:.2}"
                )
        ## Build Lorentz vectors
        ## Following ACTAR TPC's reference frame
        ## Boost is taken along beam direction: X
        E1Lab = self.T1 + self._part1.mass
        p1Lab = np.sqrt(E1Lab**2 - self._part1.mass**2)
        P1 = vec.obj(px=p1Lab, py=0, pz=0, E=E1Lab)
        P2 = vec.obj(px=0, py=0, pz=0, E=self._part2.mass)
        self.PIniLab = P1 + P2  # type: ignore
        self.beta = self.PIniLab.beta
        self.PIniCM = self.PIniLab.boostX(-self.beta)

        return

    def get_kin_for(self, thetaCM: float, phiCM: float) -> None:
        # Folow sign criterium for inverse kinematics reactions
        if self._isInverse:
            thetaCM = np.pi - thetaCM
        ECM = self.PIniCM.E
        E3CM = (
            0.5
            * (ECM**2 + self._part3.mass**2 - (self._part4.mass + self.Ex) ** 2)
            / ECM
        )
        p3CM = np.sqrt(E3CM**2 - self._part3.mass**2)
        P3CM = vec.obj(
            px=p3CM * np.cos(thetaCM),
            py=p3CM * np.sin(thetaCM) * np.sin(phiCM),
            pz=p3CM * np.sin(thetaCM) * np.cos(phiCM),
            e=E3CM,
        )
        P4CM = self.PIniCM - P3CM
        # Set the results
        ### 3rd particle
        P3Lab = P3CM.boostX(self.beta)
        self.ret.T3 = P3Lab.E - self._part3.mass
        self.ret.theta3 = self._get_theta(P3Lab)
        self.ret.phi3 = self._get_phi(P3Lab)
        ### 4th particle
        P4Lab = P4CM.boostX(self.beta)  # type: ignore
        self.ret.T4 = P4Lab.E - self._part4.mass
        self.ret.theta4 = self._get_theta(P4Lab)
        self.ret.phi4 = self._get_phi(P4Lab)
        return

    def get_line3(self) -> Tuple[np.ndarray, np.ndarray]:
        thetas = np.arange(0, 180, 0.1)
        x = np.empty_like(thetas)
        y = np.empty_like(thetas)
        for i, cm in enumerate(thetas):
            self.get_kin_for(np.deg2rad(cm), 0)
            x[i] = np.rad2deg(self.theta3)
            y[i] = self.T3
        return (x, y)

    def get_line4(self) -> Tuple[np.ndarray, np.ndarray]:
        thetas = np.arange(0, 180, 0.1)
        x = np.empty_like(thetas)
        y = np.empty_like(thetas)
        for i, cm in enumerate(thetas):
            self.get_kin_for(np.deg2rad(cm), 0)
            x[i] = np.rad2deg(self.theta4)
            y[i] = self.T4
        return (x, y)

    def get_theta3_vs_4(self) -> Tuple[np.ndarray, np.ndarray]:
        thetas = np.arange(0, 180, 0.1)
        x = np.empty_like(thetas)
        y = np.empty_like(thetas)
        for i, cm in enumerate(thetas):
            self.get_kin_for(np.deg2rad(cm), 0)
            x[i] = np.rad2deg(self.theta3)
            y[i] = np.rad2deg(self.theta4)
        return (x, y)

    def get_lab_vs_cm(self) -> Tuple[np.ndarray, np.ndarray]:
        thetas = np.arange(0, 180, 0.1)
        x = np.empty_like(thetas)
        y = np.empty_like(thetas)
        for i, cm in enumerate(thetas):
            self.get_kin_for(np.deg2rad(cm), 0)
            x[i] = cm
            y[i] = np.rad2deg(self.theta3)
        return (x, y)

    def draw(self) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"{self.reactstr}", fontsize=16)
        # 1- Kinematic line 3
        l3x, l3y = self.get_line3()
        ax = axs[0, 0]
        ax.plot(l3x, l3y)
        ax.set_xlabel(r"$\theta_{3}$ [$^{\circ}$]")
        ax.set_ylabel("E$_3$ [MeV]")

        # 2- Kinematic line 4
        l4x, l4y = self.get_line4()
        ax = axs[0, 1]
        ax.plot(l4x, l4y)
        ax.set_xlabel(r"$\theta_{4}$ [$^{\circ}$]")
        ax.set_ylabel("E$_4$ [MeV]")

        # 3- Angle 4 vs 3
        tx, ty = self.get_theta3_vs_4()
        ax = axs[1, 0]
        ax.plot(tx, ty)
        ax.set_xlabel(r"$\theta_{3}$ [$^{\circ}$]")
        ax.set_ylabel(r"$\theta_{4}$ [$^{\circ}$]")

        # 4- ThetaCM vs Lab
        cmx, cmy = self.get_lab_vs_cm()
        ax = axs[1, 1]
        ax.plot(cmx, cmy)
        ax.set_xlabel(r"$\theta_{\text{CM}}$ [$^{\circ}$]")
        ax.set_ylabel(r"$\theta_{\text{Lab, 3}}$ [$^{\circ}$]")

        plt.tight_layout()
        plt.show()
        return

    def _get_theta(self, v: vec.MomentumObject4D) -> float:
        theta = np.acos(v.px / v.mag)
        if not self._isInverse:
            # this is just the opposite conditions as in ActRoot
            ## I think it is related to sign management in python's vector inverse boost transformation
            ## inverting the condition just works
            theta = np.pi - theta
        return theta

    def _get_phi(self, v: vec.MomentumObject4D) -> float:
        phi = np.atan2(v.y, v.z)
        if phi < 0:
            phi += 2 * np.pi
        return phi

    @property
    def T3(self):
        return self.ret.T3

    @property
    def theta3(self):
        return self.ret.theta3

    @property
    def phi3(self) -> float:
        return self.ret.phi3

    @property
    def T4(self) -> float:
        return self.ret.T4

    @property
    def theta4(self) -> float:
        return self.ret.theta4

    @property
    def phi4(self) -> float:
        return self.ret.phi4

    def set_Ex(self, ex: float) -> None:
        self.Ex = ex
        self._init()
        self.ret = KinRet()
        return

    def set_T1(self, t1: float) -> None:
        self.T1 = t1
        self._init()
        self.ret = KinRet()
        return
