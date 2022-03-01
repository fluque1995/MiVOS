import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import extract_centers, frequency_and_magnitude
from visualization.plotting import plot_movements


patient = "results/P1"
experiment = 'Dedos_enfrentados'

results = []

centers = []

for visit in ['Visita_1_OFF', 'Visita_2_ON', 'Visita_3_ON']:
    with open(os.path.join(patient, visit, experiment, "masks.pkl"), "rb") as f:
        masks = pkl.load(f)

    curr_centers = extract_centers(masks, normalize=True, move_to_origin=True)
    centers.append(curr_centers)
    results.append(frequency_and_magnitude(curr_centers, fps=30, temporal_window=60))

print(results)
