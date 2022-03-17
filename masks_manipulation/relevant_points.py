import numpy as np
import sklearn
import scipy.signal

from .statistics import fingers_size


def extract_centers(masks, normalize=False, move_to_origin=False):
    '''Given the sequence of matrices where masks for fingers are extracted,
    compute the center of the masks for each different one.

    Given the sequence of matrices with the masks for fingers, extract the mean
    point of each finger in each frame. The center of the mask is a good point
    to estimate movement of the finger. In order to make those positions
    comparable, it is possible to normalize the center position by the mean size
    of the finger in the video. Also, it is possible to move the center
    coordinates using the equilibrium position of the point in the whole video

    Keyword arguments:
    masks          -- Matrices with the computed masks
    normalize      -- Choose wether to normalize using mean size of the
                      finger (default False)
    move_to_origin -- Choose wether to move the positions of the fingers to
                          the origin of coordinates (default False)
    '''

    n_masks = masks.max().astype(int)
    centers = np.zeros((n_masks, masks.shape[0], 2))
    for value in range(1, n_masks+1):
        blob_coords = np.argwhere(masks == value)
        for n_frame in range(masks.shape[0]):
            curr_blob = blob_coords[
                blob_coords[:, 0] == n_frame, 1:]
            if curr_blob.size > 0:
                centers[value - 1, n_frame] = curr_blob.mean(axis=0)
            else:
                centers[value - 1, n_frame] = np.nan

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    for i in range(len(centers)):
        nans, x = nan_helper(centers[i, :, 0])
        centers[i, nans, 0] = np.interp(x(nans), x(~nans), centers[i, ~nans, 0])
        nans, x = nan_helper(centers[i, :, 1])
        centers[i, nans, 1] = np.interp(x(nans), x(~nans), centers[i, ~nans, 1])

    if normalize:
        finger_sizes = np.sqrt(fingers_size(masks))
        for i, finger_size in enumerate(finger_sizes):
            centers[i] /= finger_size/10

    if move_to_origin:
        center_means = centers.mean(axis=1, keepdims=True)
        centers = centers - center_means
    return centers


def extract_extreme_points(matrices):
    '''Given the sequence of matrices where masks for fingers are extracted, compute
    the extreme points of each mask in each frame. Extreme points are defined as
    the center of each edge, if the mask is considered to be a rectangle. Using
    PCA, those points can be calculated using the principal components, the
    center of the mask, and the distance between the border of the mask and the
    center.
    '''

    n_masks = matrices.max().astype(int)
    centers = extract_centers(matrices)
    extreme_points = np.zeros((n_masks, matrices.shape[0], 4, 2)).astype(int)
    pca = sklearn.decomposition.PCA(2)

    for idx in range(n_masks):
        finger = idx+1
        finger_centers = centers[idx]
        finger_indices = np.argwhere(matrices == finger)
        for n_frame in range(matrices.shape[0]):
            frame_indices = finger_indices[finger_indices[:, 0] == n_frame, 1:]
            finger_center = finger_centers[n_frame]

            # Calculate direction of maximum variance (X-axis or Y-axis)
            min_x = frame_indices[:, 0].min()
            max_x = frame_indices[:, 0].max()
            min_y = frame_indices[:, 1].min()
            max_y = frame_indices[:, 1].max()
            if max_x - min_x > max_y - min_y:
                max_gt, min_gt = max_x, min_x
                max_lt, min_lt = max_y, min_y
                center_gt, center_lt = finger_center[0], finger_center[1]
            else:
                max_gt, min_gt = max_y, min_y
                max_lt, min_lt = max_x, min_x
                center_gt, center_lt = finger_center[1], finger_center[0]

            # Fit PCA and get max and min variance directions
            pca.fit(frame_indices)
            var_gt = pca.components_[0]
            var_lt = pca.components_[1]

            extreme_points[idx, n_frame] = [
                [finger_center[0] + (min_gt - center_gt)*var_gt[0],
                 finger_center[1] + (min_gt - center_gt)*var_gt[1]],
                [finger_center[0] + (max_gt - center_gt)*var_gt[0],
                 finger_center[1] + (max_gt - center_gt)*var_gt[1]],
                [finger_center[0] + (min_lt - center_lt)*var_lt[0],
                 finger_center[1] + (min_lt - center_lt)*var_lt[1]],
                [finger_center[0] + (max_lt - center_lt)*var_lt[0],
                 finger_center[1] + (max_lt - center_lt)*var_lt[1]],
            ]

    return extreme_points.astype(int)


def savgol_smoothing(finger_centers, window_length=9, polyorder=2):
    """Denoise the signal of finger movements using a Savitzky-Golay filter.
    This filter fits a polynomial in each step of order passed as argument,
    using the number of points specified as window_size for interpolation.

    Keyword Arguments:
    finger_centers -- Numpy array representing the center of the fingers.
                      Expected size: (2, n_frames, 2), corresponding to 2
                      fingers, number of frames in video, and (x,y) coords
    window_length  -- Number of points to consider in each step for windows
                      denoising
    polyorder      -- Order of the polynomial to be fit in each iteration
    """
    smoothed_centers = np.zeros_like(finger_centers)
    for i, finger in enumerate(finger_centers):
        smoothed_centers[i] = scipy.signal.savgol_filter(
            finger, window_length, polyorder, axis=0)

    return smoothed_centers
