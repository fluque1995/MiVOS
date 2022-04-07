import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import extract_extreme_points, frequency_and_magnitude
from visualization.plotting import plot_movements
from io_utils import load_masks


patient = "../Mascaras/P1"
experiment = 'Suero_izq'
visit = 'Visita_2_ON'

masks = load_masks(os.path.join(patient, visit, experiment, "masks.pkl"))
extreme_points = extract_extreme_points(masks, move_to_origin=True)

results = frequency_and_magnitude(extreme_points, fps=30)

fig, ax = plt.subplots(1,1)
ax.plot(results[0]['x']['freq'], results[0]['x']['mag'])
fig.show()

plot_movements(extreme_points[0], y_limit=None, x_limit=None, show_figures=True)
