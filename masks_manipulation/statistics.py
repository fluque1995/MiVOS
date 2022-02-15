import numpy as np
import matplotlib.pyplot as plt


def plot_movements(center_positions):
    fig, axs = plt.subplots(2, 1)
    mean_positions = center_positions.mean(axis=(1))
    for ax in axs:
        ax.set_ylim([-20, 20])
    print(mean_positions)
    for i, finger in enumerate(center_positions):
        axs[0].plot(finger[:, 0] - mean_positions[i, 0])
        axs[1].plot(finger[:, 1] - mean_positions[i, 1])

    fig.show()
