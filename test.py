from simulations_original_paper.plank_phantom import phantom_plank
from phantom import interactive_slice_viewer
import numpy as np
import matplotlib.pyplot as plt

P = phantom_plank((20, 100, 50), p=0.05, angle=np.deg2rad(20)).transpose([0, 2, 1])
interactive_slice_viewer(P)
