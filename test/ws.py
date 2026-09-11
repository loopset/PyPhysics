import pyphysics as phys
import matplotlib.pyplot as plt
import numpy as np


def parse_overlap(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    # First line contains metadata
    header = lines[1].split()
    n_points = int(header[0])
    rstep = float(header[1])
    rfirst = int(header[2])

    # Accumulate all numbers into a flat list
    all_numbers = []
    for line in lines[2:]:
        all_numbers.extend(float(x) for x in line.split())

    # Split based on n_points
    wf = np.array(all_numbers[:n_points])
    vwf = np.array(all_numbers[n_points:])

    r = np.arange(rfirst, rfirst + rstep * n_points, rstep)

    return r, wf, vwf


# Fresco overlap
fresco = parse_overlap("/media/Data/E748/Fits/12Be_d3He/Inputs/rms_li_1n/fort.26")

# Core: 9Li
# Valence: p
# BE: S2n(11Li)/2
ws = phys.WoodSaxonOverlap(A=9, Z=3, a=1, z=1, n=0, l=1, j=0.5, be=-0.1847)
print(f"gs for defautl V {ws.V:.2f} is :", ws.solve_eigen()[0])
ws.solve_be()

ws.plot()

# plt.gca().plot(fresco[0], fresco[2] / fresco[1], label="Fresco", color="crimson")
plt.show()
