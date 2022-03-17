import pickle as pkl


def load_masks(filepath):
    """Load the resulting masks from a segmentation, which are stored as pickle
    objects

    Keyword Arguments:
    filepath -- Path to the pickle file in memory
    """

    with open(filepath, "rb") as fin:
        masks = pkl.load(fin)

    return masks
