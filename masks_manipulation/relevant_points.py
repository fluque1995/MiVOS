import numpy as np
import sklearn

def extract_centers(matrices):
    '''Given the sequence of matrices where masks for fingers are extracted, compute
    the center of the masks for each different one. The center of the mask is a
    good point to estimate movement of the finger
    '''

    n_masks = matrices.max().astype(int)
    centers = np.zeros((n_masks, matrices.shape[0], 2)).astype(int)
    for value in range(1, n_masks+1):
        blob_coords = np.argwhere(matrices == value)
        for n_frame in range(matrices.shape[0]):
            centers[value - 1, n_frame] = blob_coords[
                blob_coords[:, 0] == n_frame, 1:].mean(axis=0)
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
