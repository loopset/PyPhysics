from .utils import (
    parse_txt,
    create_spline3,
    create_trans_imshow,
    create_interp1d,
    find_root,
)
from .particle import Particle
from .energy_loss import EnergyLoss
from .kinematics import Kinematics

from .theory import QuantumNumbers, ShellModelData, ShellModel, SMDataDict
from .cross_section import Comparator
from .bernstein import Radii, Diffuseness, Bernstein, BE_to_beta, simple_bernstein
from .barager import BaragerRes, Barager

from .fitting import fit_poln

# Let user manually import ROOT dependent submodules
# from .root_interface import parse_tgraph, parse_th1, parse_tcutg
# from .actroot_interface import (
#     DataManInterface,
#     TPCInterface,
#     LineInterface,
#     KinInterface,
#     FitInterface,
#     SFModel,
#     SFInterface,
# )

# Automatically set style
import os
import matplotlib.pyplot as plt

STYLE_PATH = os.path.join(os.path.dirname(__file__), "actroot.mplstyle")
plt.style.use(STYLE_PATH)
