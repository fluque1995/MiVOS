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
            vm = axs[0].plot(finger[:, 0])
            hm = axs[1].plot(finger[:, 1], range(len(finger)))

        if y_limit is not None:
            axs[0].set_ylim([-y_limit, y_limit])
        #axs[0].legend([f"Finger {i+1}" for i in range(len(center_positions))])
        axs[0].set_title(f"Fingers' vertical movement", fontsize=10, loc="center")

        if x_limit is not None:
            axs[1].set_xlim([-x_limit, x_limit])

        axs[1].set_ylim([0, len(finger)])
        #axs[1].legend([f"Finger {i+1}" for i in range(len(center_positions))])
        axs[1].set_title(f"Fingers' horizontal movement", fontsize=10, loc="center")
        fig.tight_layout()

        labels = ["Right finger", "Left finger"]
        fig.legend(labels=labels, bbox_to_anchor=(1.2, 0.6))
        if show_figures:
            fig.show()
        if saving_path is not None:
            fig.savefig(saving_path, bbox_inches='tight')


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
            axs[0].legend(["Right finger", "Left finger"])

            axs[1].set_xlim([-x_limit, x_limit])
            axs[1].set_ylim([0, len(finger)])
            axs[1].legend(["Right finger", "Left finger"])

    fig.tight_layout()
    if show_figures:
        fig.show()

    if saving_path is not None:
        fig.savefig(saving_path, bbox_inches='tight')


def plot_finger_heatmaps(masks_list, frames_list, subplots=(1, 1),
                         movement_index=False, show_figures=False,
                         saving_path=None):
    """Plot heatmaps for the finger masks in the first frame of the video

    Keyword arguments:
    masks          -- Segmentation masks for the fingers in the video
    frame          -- First frame of the video, to use as background
    movement_index -- Calculate movement index of the fingers in the video and
                      show them in the plot (default: False)
    show_figures   -- Whether to display the graphs in a window or not
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
            for finger_id, movement in moves.items():
                finger = "right" if finger_id == 1 else "left"
                ax.text(1050, 175 + 25*finger_id,
                        f"Mov. index for {finger} finger: {movement:.2f}",
                        ha='right',
                        color='black')
            ax.set_title("Fingers' heatmaps and movement indexes",
                         fontsize=15, y=1.1)
        else:
            ax.set_title("Fingers' heatmaps",
                         fontsize=15, y=1.1)

    if show_figures:
        fig.show()

    if saving_path is not None:
        fig.savefig(saving_path, bbox_inches='tight')
