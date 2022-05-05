import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import (
    extract_centers, extract_extreme_points, frequency_and_magnitude
)
from visualization.plotting import plot_movements
from io_utils import load_masks


patient = "../Mascaras/P1"
experiment = 'Dedos_enfrentados'
visit = 'Visita_1_OFF'

masks = load_masks(os.path.join(patient, visit, experiment, "masks.pkl"))
extreme_points_raw = extract_centers(masks, move_to_origin=False, normalize=False)
extreme_points_orig = extract_centers(masks, move_to_origin=True, normalize=False)
extreme_points_norm = extract_centers(masks, move_to_origin=False, normalize=True)
extreme_points_both = extract_centers(masks, move_to_origin=True, normalize=True)

'''
results = frequency_and_magnitude(extreme_points, fps=30)

fig, ax = plt.subplots(1,1)
ax.plot(results[0]['v']['freq'], results[0]['v']['mag'])
fig.show()

plot_movements(extreme_points, y_limit=None, x_limit=None, show_figures=True)
'''
