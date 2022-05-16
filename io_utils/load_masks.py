import pickle as pkl
import numpy as np


def load_masks(filepath, noise_thr=0.2):
    """Load the resulting masks from a segmentation, which are stored as pickle
    objects

    Keyword Arguments:
    filepath  -- Path to the pickle file in memory
    noise_thr -- Minimum percentage of frames having a mask value to consider
                 that the mask is not environmental noise. If the mask is
                 considered noise, it is deleted from the masks array
    """

    with open(filepath, "rb") as fin:
        masks = pkl.load(fin)

    unique_masks = np.unique(masks)
    for val in unique_masks:
        if val != 0:
            num_frames = np.unique(np.argwhere(masks == val)[:, 0])
            if len(num_frames) / masks.shape[0] < noise_thr:
                masks[masks == val] = 0

    return masks
