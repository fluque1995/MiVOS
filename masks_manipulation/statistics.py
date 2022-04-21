import logging
import matplotlib.pyplot as plt
import numpy as np
import os


def fingers_size(masks, temporal_window=None):
    """Calculate the size of each finger in the video from the masks
    Keyword Arguments:
    masks           -- Matrices with the computed masks
    temporal_window -- Calculate the size of the fingers in windows of n frames.
                       If None, calculate the size for the whole video (default None)
    """
    masks_values = np.unique(masks)
    masks_values = masks_values[masks_values != 0]
    if temporal_window is None:
        finger_sizes = []
        for finger in masks_values:
            curr_finger_indices = (masks == finger)
            frame_sizes = curr_finger_indices.sum(axis=(1, 2))
            total_size = frame_sizes[frame_sizes > 0].mean()
            finger_sizes.append(total_size)
    else:
        finger_sizes = []
        for finger in masks_values:
            curr_finger_sizes = []
            for i in range(0, masks.shape[0], temporal_window):
                curr_masks = masks[i, i + temporal_window]
                curr_finger_indices = curr_masks == finger
                frame_sizes = curr_finger_indices.sum(axis=(1, 2))
                total_size = frame_sizes[frame_sizes > 0].mean()
                curr_finger_sizes.append(total_size)
            finger_sizes.append(curr_finger_sizes)

    return finger_sizes


def frequency_and_magnitude(finger_points,
                            fps=30,
                            temporal_window=None,
                            magnitude_thr=50,
                            graph_path=None):
    """Calculate oscillation frequency and magnitude of finger points using fast
    fourier transform. The oscillation frequency is calculated as the biggest
    frequency whose magnitude is greater than a certain threshold

    Keyword arguments:
    finger_points   -- Sequences of relevant points to use for frequency calculation
    fps             -- Frequency of sampling for data. Default 30, as is the
                       framerate of our test videos
    temporal_window -- Calculate the frequency considering temporal windows of
                       n frames. If None, calculate the frequency using the whole
                       video (default None)
    magnitude_thr   -- Threshold for magnitude to consider the frequency
                       oscillation as significative
    graph_path      -- Path to the file to store the frequency-magnitude graph.
                       If None, the graph is not generated (default None). The
                       graph is only generated when there is no temporal window

    """
    results = {}
    for i, finger in enumerate(finger_points):
        x_points = finger[:, 0]
        y_points = finger[:, 1]

        if np.any(x_points == np.nan) or np.any(y_points == np.nan):
            continue

        if temporal_window is None:
            x_fft = abs(np.fft.fft(x_points))
            x_freqs = np.fft.fftfreq(len(x_fft))
            significative_x_fft = (x_fft > magnitude_thr)
            signif_indices = np.argwhere(significative_x_fft)

            # The biggest frequency with positive value in the frequencies array
            # is placed in the middle of the vector. If no magnitude value is
            # over the threshold, we just return the biggest one
            if len(signif_indices) > 0:
                max_id_x = signif_indices[len(signif_indices) // 2 - 1][0]
            else:
                max_id_x = np.argmax(x_fft)

            y_fft = abs(np.fft.fft(y_points))
            y_freqs = np.fft.fftfreq(len(y_fft))
            significative_y_fft = (y_fft > magnitude_thr)
            signif_indices = np.argwhere(significative_y_fft)
            if len(signif_indices) > 0:
                max_id_y = signif_indices[len(signif_indices) // 2 - 1][0]
            else:
                max_id_y = np.argmax(y_fft)

            results[i] = {
                'x': {
                    'mag': x_fft,
                    'freq': x_freqs,
                    'max_mag': x_fft[max_id_x],
                    'max_freq': abs(fps * x_freqs[max_id_x])
                },
                'y': {
                    'mag': y_fft,
                    'freq': y_freqs,
                    'max_mag': y_fft[max_id_y],
                    'max_freq': abs(fps * y_freqs[max_id_y])
                }
            }

            if graph_path is not None:
                with plt.style.context("ggplot"):
                    color = list(plt.rcParams['axes.prop_cycle'])[i]['color']
                    fig, axs = plt.subplots(nrows=2, ncols=1)
                    axs[0].plot(fps * x_freqs[:len(x_freqs) // 2],
                                x_fft[:len(x_fft) // 2],
                                color=color)
                    axs[0].set_title("Vertical movement frequencies", fontsize=10)
                    axs[1].plot(fps * y_freqs[:len(y_freqs) // 2],
                                y_fft[:len(y_fft) // 2], color=color)
                    axs[1].set_title("Horizontal movement frequencies", fontsize=10)

                    finger_d = {0: 'right', 1: 'left'}
                    fig.suptitle(f"Magnitude-frequency for {finger_d[i]} finger")

                    fig.subplots_adjust(hspace=0.4)

                    graph_prefix = "right_" if i == 0 else "left_"
                    fig.savefig(
                        os.path.split(graph_path)[0] + "/" + graph_prefix +
                        os.path.split(graph_path)[1])

        else:
            x_mags, x_freqs, y_mags, y_freqs = [], [], [], []
            for j in range(0, finger.shape[0], temporal_window):
                curr_x_points = x_points[j:j + temporal_window]
                curr_y_points = y_points[j:j + temporal_window]

                x_fft = abs(np.fft.fft(curr_x_points))
                x_freq = np.fft.fftfreq(len(x_fft))
                max_id_x = np.argmax(x_fft)

                y_fft = abs(np.fft.fft(curr_y_points))
                y_freq = np.fft.fftfreq(len(y_fft))
                max_id_y = np.argmax(y_fft)

                x_mags.append(x_fft[max_id_x])
                x_freqs.append(abs(fps * x_freq[max_id_x]))
                y_mags.append(y_fft[max_id_y])
                y_freqs.append(abs(fps * y_freq[max_id_y]))

            results[i] = {
                'x_mag': x_mags,
                'x_freq': x_freqs,
                'y_mag': y_mags,
                'y_freq': y_freqs
            }

            if graph_path is not None:
                logging.warning((
                    "The method is not suitable for plotting when working on "
                    "temporal windows. Please call the method with no temporal "
                    "window if you want to store the graphs"))

    return results
