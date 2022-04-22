import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import scipy.signal


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
            curr_finger_indices = masks == finger
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


def frequency_and_magnitude(
    finger_points,
    fps=30,
    temporal_window=None,
    percentage_thr=0.2,
    value_thr=30,
    graph_path=None,
):
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
    percentage_thr  -- Percentage threshold of the maximum magnitude value to
                       consider a frequency as relevant
    value_thr       -- Minimum value of magnitude to be considered as relevant
    graph_path      -- Path to the file to store the frequency-magnitude graph.
                       If None, the graph is not generated (default None). The
                       graph is only generated when there is no temporal window

    """
    results = {}
    for i, finger in enumerate(finger_points):
        v_points = finger[:, 0]
        h_points = finger[:, 1]

        if np.any(np.isnan(v_points)) or np.any(np.isnan(h_points)):
            print(f"Skipping finger {i}")
            continue

        if temporal_window is None:
            v_fft = abs(np.fft.fft(v_points))
            v_freqs = np.fft.fftfreq(len(v_fft))

            v_fft = v_fft[: len(v_fft) // 2]
            v_freqs = v_freqs[: len(v_freqs) // 2]

            max_v_fft = np.max(v_fft)
            v_thr = np.max(percentage_thr * max_v_fft, value_thr)

            v_signif_indices = scipy.signal.find_peaks(
                v_fft, height=v_thr, distance=10
            )[0]
            max_id_v = v_signif_indices[-1]

            h_fft = abs(np.fft.fft(h_points))
            h_freqs = np.fft.fftfreq(len(h_fft))

            h_fft = h_fft[: len(h_fft) // 2]
            h_freqs = h_freqs[: len(h_freqs) // 2]

            max_h_fft = np.max(h_fft)
            h_thr = np.max(percentage_thr * max_h_fft, value_thr)

            h_signif_indices = scipy.signal.find_peaks(
                h_fft, height=h_thr, distance=10
            )[0]
            max_id_h = h_signif_indices[-1]

            results[i] = {
                "v": {
                    "mag": v_fft,
                    "freq": v_freqs,
                    "max_mag": v_fft[max_id_v],
                    "max_freq": abs(fps * v_freqs[max_id_v]),
                    "peak_mag": v_fft[v_signif_indices],
                    "peak_freq": fps * v_freqs[v_signif_indices],
                },
                "h": {
                    "mag": h_fft,
                    "freq": h_freqs,
                    "max_mag": h_fft[max_id_h],
                    "max_freq": abs(fps * h_freqs[max_id_h]),
                    "peak_mag": h_fft[h_signif_indices],
                    "peak_freq": fps * h_freqs[h_signif_indices],
                },
            }

            if graph_path is not None:
                with plt.style.context("ggplot"):
                    color = list(plt.rcParams["axes.prop_cycle"])[i]["color"]
                    x_color = list(plt.rcParams["axes.prop_cycle"])[3]["color"]
                    fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True)
                    axs[0].plot(
                        fps * v_freqs,
                        v_fft,
                        color=color,
                    )
                    axs[0].plot(
                        fps * v_freqs[v_signif_indices],
                        v_fft[v_signif_indices],
                        "x",
                        color=x_color,
                    )
                    axs[0].set_title("Vertical movement frequencies", fontsize=10)
                    axs[1].plot(
                        fps * h_freqs,
                        h_fft,
                        color=color,
                    )
                    axs[1].plot(
                        fps * h_freqs[h_signif_indices],
                        h_fft[h_signif_indices],
                        "x",
                        color=x_color,
                    )
                    axs[1].set_title("Horizontal movement frequencies", fontsize=10)

                    finger_d = {0: "right", 1: "left"}
                    fig.suptitle(f"Magnitude-frequency for {finger_d[i]} finger")

                    fig.subplots_adjust(hspace=0.4)

                    graph_prefix = "right_" if i == 0 else "left_"
                    fig.savefig(
                        os.path.split(graph_path)[0]
                        + "/"
                        + graph_prefix
                        + os.path.split(graph_path)[1]
                    )

        else:
            v_mags, v_freqs, h_mags, h_freqs = [], [], [], []
            for j in range(0, finger.shape[0], temporal_window):
                curr_v_points = v_points[j : j + temporal_window]
                curr_h_points = h_points[j : j + temporal_window]

                v_fft = abs(np.fft.fft(curr_v_points))
                v_freq = np.fft.fftfreq(len(v_fft))
                max_id_v = np.argmax(v_fft)

                h_fft = abs(np.fft.fft(curr_h_points))
                h_freq = np.fft.fftfreq(len(h_fft))
                max_id_h = np.argmax(h_fft)

                v_mags.append(v_fft[max_id_v])
                v_freqs.append(abs(fps * v_freq[max_id_v]))
                h_mags.append(h_fft[max_id_h])
                h_freqs.append(abs(fps * h_freq[max_id_h]))

            results[i] = {
                "v_mag": v_mags,
                "v_freq": v_freqs,
                "h_mag": h_mags,
                "h_freq": h_freqs,
            }

            if graph_path is not None:
                logging.warning(
                    (
                        "The method is not suitable for plotting when working on "
                        "temporal windows. Please call the method with no temporal "
                        "window if you want to store the graphs"
                    )
                )

    return results
