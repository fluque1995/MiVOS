import matplotlib.pyplot as plt
import numpy as np


def plot_movements(center_positions):
    """Plot x-axis and y-axis displacement from mean position for center of the
    fingers

    Keyword arguments:
    center_positions -- x-y tuples of positions corresponding to the center of
                        the masks representing the fingers
    """

    fig, axs = plt.subplots(2, 1)
    mean_positions = center_positions.mean(axis=1)

    for i, finger in enumerate(center_positions):
        axs[0].plot(finger[:, 0] - mean_positions[i, 0])
        axs[1].plot(finger[:, 1] - mean_positions[i, 1])

    for ax in axs:
        ax.set_ylim([-20, 20])
        ax.legend([f"Finger {i}" for i in range(len(center_positions))])

    fig.show()


def plot_speeds(center_positions):
    """Plot x-axis and y-axis speed in movement

    Keyword arguments:
    center_positions -- x-y tuples of positions corresponding to the center of
                        the masks representing the fingers
    """

    fig = plt.figure()
    ax = fig.add_subplot()
    speeds = np.diff(center_positions, axis=1)
    for finger in speeds:
        ax.plot(np.sqrt(finger[:, 1]**2 + finger[:, 0]**2))
    fig.show()
