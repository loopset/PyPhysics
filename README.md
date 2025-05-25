# PyPhysics

Base classes and utilities to perform nuclear physics analysis using Python:

- Particle databases
- Kinematics calculators
- Spectroscopic factor extracts and model comparators
- Shell model and nuclear structure

## Dependencies
Ensure the following packages are installed before executing `PyPhysics`:

```bash
pip install numpy matplotlib scipy lmfit uncertainties hist vector
```

Optionally, interfacing classes with `ROOT` and `ActRoot` require `pyROOT` to be installed on your system. Follow ROOT installation instruction to be able to use them.

## Installation

After cloning this repository, you must locate the package by modifying the `PYTHONPATH`. For that, append this line to your `.bashrc` or similar file:

```bash
source /home/user/PyPhysics/thisPyPhysics.sh
```

Alternatively, a `pip` installation is accesible via:

```bash
pip install git+https://github.com/loopset/PyPhysics.git
```

However you must keep an eye to the versioning! A simple `git pull` will not update your package: you need to run `pip install` again.