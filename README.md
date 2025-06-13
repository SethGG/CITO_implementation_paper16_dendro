# CITO Implementation paper 16: A novel method for dendrochronology of large historical wooden objects using line trajectory X-ray tomography

This repository contains code for generating synthetic tree-ring phantoms, simulating cone-beam CT scans, and evaluating reconstruction quality based on dendrochronological features.

---

## Contents

* `phantom.py`: Code to generate 3D wood block phantoms with realistic tree-rings.
* `sinogram.py`: Code to simulate a cone-beam CT scan and reconstruct from sinograms using ASTRA Toolbox.
* `experiments.py`: Script to run experiments and evaluate reconstruction quality (SSIM and tree-ring correlation metrics).
* `environment.yml`: Conda environment file for installing required dependencies.

---

## Installation

To set up the environment, use Anaconda:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate cito
```

---

## Usage

### 1. Run All Experiments

Edit the flags at the bottom of `experiments.py` to select which experiments to run:

```python
exp1 = True  # Tree-ring width tilt at cone angle 9°
exp2 = False # Tree-ring width tilt at cone angle 15°
exp3 = False # Tree-ring height tilt at cone angle 9°
exp4 = False # Tree-ring height tilt at cone angle 15°
```

Then run:

```bash
python experiments.py
```

Results will be saved in the `output/` directory:

* `phantoms/` for generated phantom slices.
* `reconstructions/` for reconstructed slices.
* `figures/` for evaluation plots (SSIM, radial correlation, etc).

### 2. View a Phantom Interactively

```bash
python phantom.py
```

This opens a GUI to browse through a 3D phantom slices along different directions.


## Authors

* Daniël Zee (s2063131)
* Martijn Combé (s2599406)

For the course *Computational Imaging and Tomography* (CITO 2024-2025)
