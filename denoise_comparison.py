import io_utils
import matplotlib.pyplot as plt

from masks_manipulation import extract_centers, savgol_smoothing
from visualization.plotting import plot_movements

masks_off = io_utils.load_masks("results/P1/Visita_1_OFF/D-N_der/masks.pkl")
masks_on = io_utils.load_masks("results/P1/Visita_3_ON/D-N_der/masks.pkl")

centers_off = extract_centers(masks_off, normalize=True, move_to_origin=True)
smoothed_centers_off = savgol_smoothing(centers_off, 9, 2)

centers_on = extract_centers(masks_on, normalize=True, move_to_origin=True)
smoothed_centers_on = savgol_smoothing(centers_on, 9, 2)

plot_movements(centers_off, x_limit=None, y_limit=None, show_figures=True)
plot_movements(smoothed_centers_off, x_limit=None, y_limit=None, show_figures=True)

plot_movements(centers_on, x_limit=None, y_limit=None, show_figures=True)
plot_movements(smoothed_centers_on, x_limit=None, y_limit=None, show_figures=True)

differences_off = centers_off - smoothed_centers_off
differences_on = centers_on - smoothed_centers_on

with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
    for i, (finger_off, finger_on) in enumerate(zip(differences_off, differences_on)):
        for j, axis in enumerate(finger_off.T):
            ax[0, 2*i+j].hist(axis, label=f"STD: {axis.std():.3f}", edgecolor='black')
            ax[0, 2*i+j].legend()
            ax[0, 2*i+j].set_title(f"Device OFF. Finger: {i+1}, axis: {j+1}")

        for j, axis in enumerate(finger_on.T):
            ax[1, 2*i+j].hist(axis, label=f"STD: {axis.std():.3f}", edgecolor='black')
            ax[1, 2*i+j].legend()
            ax[1, 2*i+j].set_title(f"Device ON. Finger: {i+1}, axis: {j+1}")

plt.show()
