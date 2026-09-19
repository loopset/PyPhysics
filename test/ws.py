import pyphysics as phys
import matplotlib.pyplot as plt
import numpy as np


# Helper function to parse FRESCO nfl overlap output
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
ws = phys.WoodsSaxonOverlap(core="9Li", valence="n", q="0p1/2", be=-0.1847)
ws.solve()
ws.print_config()

norm = np.sum(ws.eigenWF**2 * ws.dr)  # type: ignore
print(f"Norm of the wavefunction: {norm:.4f}")
norm_fresco = np.sum(fresco[1] ** 2 * (fresco[0][1] - fresco[0][0]))
print(f"Norm of the Fresco wavefunction: {norm_fresco:.4f}")

_, axs = ws.plot()

# Plot potential
# axs[0].plot(fresco[0], fresco[2] / fresco[1], label="Fresco", color="crimson")
# Plot wf
axs[-1].plot(fresco[0], fresco[1], label="Fresco", color="crimson")
plt.show()
