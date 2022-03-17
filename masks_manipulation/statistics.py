import numpy as np
import matplotlib.pyplot as plt

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
                curr_masks = masks[i, i+temporal_window]
                curr_finger_indices = curr_masks == finger
                frame_sizes = curr_finger_indices.sum(axis=(1, 2))
                total_size = frame_sizes[frame_sizes > 0].mean()
                curr_finger_sizes.append(total_size)
            finger_sizes.append(curr_finger_sizes)

    return finger_sizes


def frequency_and_magnitude(finger_points, fps=30, temporal_window=None):
    """Calculate oscillation frequency and magnitude of finger points using fast
    fourier transform

    Keyword arguments:
    finger_points   -- Sequences of relevant points to use for frequency calculation
    fps             -- Frequency of sampling for data. Default 30, as is the
                       framerate of our test videos
    temporal_window -- Calculate the frequency considering temporal windows of
                       n frames. If None, calculate the frequency using the whole
                       video (default None)
    """
    results = {}
    for i, finger in enumerate(finger_points):
        x_points = finger[:, 0]
        y_points = finger[:, 1]

        if temporal_window is None:
            x_fft = abs(np.fft.fft(x_points))
            x_freqs = np.fft.fftfreq(len(x_fft))
            max_id_x = np.argmax(x_fft)

            y_fft = abs(np.fft.fft(y_points))
            y_freqs = np.fft.fftfreq(len(y_fft))
            max_id_y = np.argmax(y_fft)

            results[i] = {
                'x_mag': x_fft[max_id_x], 'x_freq': abs(fps*x_freqs[max_id_x]),
                'y_mag': y_fft[max_id_y], 'y_freq': abs(fps*y_freqs[max_id_y])
            }
        else:
            x_mags, x_freqs, y_mags, y_freqs = [], [], [], []
            for j in range(0, finger.shape[0], temporal_window):
                curr_x_points = x_points[j:j+temporal_window]
                curr_y_points = y_points[j:j+temporal_window]

                x_fft = abs(np.fft.fft(curr_x_points))
                x_freq = np.fft.fftfreq(len(x_fft))
                max_id_x = np.argmax(x_fft)

                y_fft = abs(np.fft.fft(curr_y_points))
                y_freq = np.fft.fftfreq(len(y_fft))
                max_id_y = np.argmax(y_fft)

                x_mags.append(x_fft[max_id_x])
                x_freqs.append(abs(fps*x_freq[max_id_x]))
                y_mags.append(y_fft[max_id_y])
                y_freqs.append(abs(fps*y_freq[max_id_y]))

            results[i] = {
                'x_mag': x_mags, 'x_freq': x_freqs,
                'y_mag': y_mags, 'y_freq': y_freqs
            }

    return results
