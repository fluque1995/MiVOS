import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import extract_extreme_points, frequency_and_magnitude
from visualization.plotting import plot_movements
from io_utils import load_masks


patient = "../Mascaras/P3"
experiment = 'Suero_izq'

results = []

points = []
ys = []

for visit in ['Visita_1_OFF', 'Visita_2_ON', 'Visita_3_ON']:
    masks = load_masks(os.path.join(patient, visit, experiment, "masks.pkl"))
    curr_points = extract_extreme_points(masks, move_to_origin=True)
    curr_extreme = curr_points[:, :, 0, :]

    points.append(curr_extreme)
    results.append(frequency_and_magnitude(curr_extreme, fps=30))


fig, ax = plt.subplots(1,1)
ax.plot(results[1][0]['x']['freq'], results[1][0]['x']['mag'])
fig.show()

plot_movements(points[0], y_limit=None, x_limit=None, show_figures=True)
