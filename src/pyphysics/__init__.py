from .utils import parse_txt, create_spline3, create_trans_imshow, create_interp1d, find_root
from .particle import Particle
from .cross_section import Comparator
from .bernstein import Radii, Diffuseness, Bernstein, BE_to_beta, simple_bernstein

# Automatically set style
import os
import matplotlib.pyplot as plt
STYLE_PATH = os.path.join(os.path.dirname(__file__), "actroot.mplstyle")
plt.style.use(STYLE_PATH)
