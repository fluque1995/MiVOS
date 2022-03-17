import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import extract_centers, frequency_and_magnitude
from visualization.plotting import plot_movements
from io_utils import load_masks


patient = "results/P4"
experiment = 'Dedos_enfrentados'

results = []

centers = []

for visit in ['Visita_1_OFF', 'Visita_2_ON', 'Visita_3_ON']:
    masks = load_masks(os.path.join(patient, visit, experiment, "masks.pkl"))

    curr_centers = extract_centers(masks, normalize=True, move_to_origin=True)
    centers.append(curr_centers)
    results.append(frequency_and_magnitude(curr_centers, fps=30))

plt.plot(results[0][0]['x']['freq'], results[0][0]['x']['mag'])
plt.show()
