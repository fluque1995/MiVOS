import numpy as np


def movement_index(masks, temporal_window=-1):
    """Calculate the amount of movement in the fingers, calculated as the ratio of
    the total space visited by each finger during the whole video and the mean
    space occupied by that finger in each frame. This method also allows to perform
    the computation in temporal windows, so slow displacements of the finger during
    the whole video are corrected.

    Keyword arguments:
    masks           -- np.array with the result of segmentation
    temporal_window -- Number of frames to consider in each temporal window. Use
                       -1 to calculate the result for the whole video (default)

    """
    different_values = np.unique(masks)
    different_values = different_values[different_values != 0]
    results = {}
    for value in different_values:
        if temporal_window != -1:
            curr_finger_movements = []
            for i in range(0, masks.shape[0], temporal_window):
                curr_masks = masks[i: i+temporal_window]
                curr_finger_masks = curr_masks == value
                # The total visited space is the sum of the pixel visited at
                # least once
                total_finger_positions = curr_finger_masks.any(axis=0).sum()
                # The mean size of the finger is the mean of the sum of the
                # visited pixels in each frame
                mean_finger_size = curr_finger_masks.sum(axis=(1, 2)).mean()
                curr_finger_movements.append(
                    total_finger_positions / mean_finger_size)

        else:
            curr_finger_masks = masks == value
            total_finger_positions = curr_finger_masks.any(axis=0).sum()
            mean_finger_size = curr_finger_masks.sum(axis=(1, 2)).mean()
            curr_finger_movements = total_finger_positions / mean_finger_size

        results[value] = curr_finger_movements

    return results
