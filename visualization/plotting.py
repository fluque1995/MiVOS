import matplotlib.pyplot as plt
import numpy as np
import masks_manipulation


def plot_movements(center_positions, y_limit=20, x_limit=20,
                   show_figures=False, saving_path=None):
    """Plot x-axis and y-axis displacement from mean position for center of the
    fingers

    Keyword arguments:
    center_positions -- x-y tuples of positions corresponding to the center of
                        the masks representing the fingers
    y_limit          -- Min/max value for the vertical movement plot
    x_limit          -- Min/max value for the horizontal movement plot
    show_figures     -- Whether to display the graphs in a window or not
    saving_path      -- Path where we want to save the plot. If None, the image
                        is not saved (default None)
    """

    with plt.style.context(('ggplot')):
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

        for i, finger in enumerate(center_positions):
            axs[0].plot(finger[:, 0])
            axs[1].plot(finger[:, 1], range(len(finger)))

        axs[0].set_ylim([-y_limit, y_limit])
        axs[0].legend([f"Finger {i+1}" for i in range(len(center_positions))])

        axs[1].set_xlim([-x_limit, x_limit])
        axs[1].set_ylim([0, len(finger)])
        axs[1].legend([f"Finger {i+1}" for i in range(len(center_positions))])
        fig.tight_layout()
        if show_figures:
            fig.show()
        if saving_path is not None:
            fig.savefig(saving_path)


def plot_overlapped_movements(centers, y_limit=20, x_limit=20,
                              show_figures=False, saving_path=None):
    """Plot x-axis and y-axis displacement from mean position for center of the
    fingers. Centers is a list, so the centers are plotted overlapped

    Keyword arguments:
    centers      -- List of x-y tuples of positions corresponding to the center of
                    the masks representing the fingers
    y_limit      -- Min/max value for the vertical movement plot
    x_limit      -- Min/max value for the horizontal movement plot
    show_figures -- Whether to display the graphs in a window or not
    saving_path  -- Path where we want to save the plot. If None, the image
                    is not saved (default None)
    """

    overlapping = 0.5

    with plt.style.context(('ggplot')):
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

        for center_positions in centers:
            for i, finger in enumerate(center_positions):
                axs[0].plot(finger[:, 0], alpha=overlapping)
                axs[1].plot(finger[:, 1], range(len(finger)), alpha=overlapping)

            axs[0].set_ylim([-y_limit, y_limit])
            axs[0].legend([f"Finger {i+1}" for i in range(len(center_positions))])

            axs[1].set_xlim([-x_limit, x_limit])
            axs[1].set_ylim([0, len(finger)])
            axs[1].legend([f"Finger {i+1}" for i in range(len(center_positions))])

    fig.tight_layout()
    if show_figures:
        fig.show()

    if saving_path is not None:
        fig.savefig(saving_path)


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


def plot_finger_heatmaps(masks_list, frames_list, subplots=(1, 1), movement_index=False, saving_path=None):
    """Plot heatmaps for the finger masks in the first frame of the video

    Keyword arguments:
    masks          -- Segmentation masks for the fingers in the video
    frame          -- First frame of the video, to use as background
    movement_index -- Calculate movement index of the fingers in the video and
                      show them in the plot (default: False)
    saving_path    -- Path where we want to save the plot. If None, the image
                      is not saved (default None)
    """

    fig, axs = plt.subplots(*subplots, squeeze=False)
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

    #fig.show()
    if saving_path is not None:
        fig.savefig(saving_path)
