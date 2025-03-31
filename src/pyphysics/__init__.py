from .utils import parse_txt, create_spline3
from .cross_section import Comparator

# Automatically set style
import os
import matplotlib.pyplot as plt
STYLE_PATH = os.path.join(os.path.dirname(__file__), "actroot.mplstyle")
plt.style.use(STYLE_PATH)
