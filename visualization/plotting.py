import matplotlib.pyplot as plt
import numpy as np
import masks_manipulation

def plot_movements(center_positions):
    """Plot x-axis and y-axis displacement from mean position for center of the
    fingers

    Keyword arguments:
    center_positions -- x-y tuples of positions corresponding to the center of
                        the masks representing the fingers
    """

    fig, axs = plt.subplots(2, 1)

    for i, finger in enumerate(center_positions):
        axs[0].plot(finger[:, 0])
        axs[1].plot(finger[:, 1])

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


def plot_finger_heatmaps(masks_list, frames_list, subplots=(1, 1), movement_index=False):
    """Plot heatmaps for the finger masks in the first frame of the video

    Keyword arguments:
    masks          -- Segmentation masks for the fingers in the video
    frame          -- First frame of the video, to use as background
    movement_index -- Calculate movement index of the fingers in the video and
                      show them in the plot (default: False)
    """

    fig, axs = plt.subplots(*subplots)
    for ax, mask, frame in zip(axs.ravel(), masks_list, frames_list):
        heatmap = mask.sum(axis=0)
        ax.imshow(frame)
        ax.matshow(heatmap, alpha=0.9)

        if movement_index:
            moves = masks_manipulation.movement_index(mask)
            for i, (finger, movement) in enumerate(moves.items()):
                ax.text(50, 300 + 20*i,
                        f"Movement index for finger {finger}: {movement}",
                        color='white')

    fig.show()
