import cv2
import numpy as np
import sklearn
import sklearn.decomposition
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
        if not nans.all():
            centers[i, nans, 0] = np.interp(x(nans), x(~nans),
                                            centers[i, ~nans, 0])
        nans, x = nan_helper(centers[i, :, 1])
        if not nans.all():
            centers[i, nans, 1] = np.interp(x(nans), x(~nans),
                                            centers[i, ~nans, 1])

    if normalize:
        finger_sizes = np.sqrt(fingers_size(masks))
        for i, finger_size in enumerate(finger_sizes):
            centers[i] /= finger_size/10

    if move_to_origin:
        center_means = centers.mean(axis=1, keepdims=True)
        centers = centers - center_means
    return centers


def extract_extreme_points_old(matrices, normalize=False, move_to_origin=False):
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

            # Fit PCA and get max and min variance directions
            pca.fit(frame_indices)
            transformed_fingers = pca.transform(frame_indices)
            max_gt_id = np.argmax(transformed_fingers[:, 0])
            min_gt_id = np.argmin(transformed_fingers[:, 0])
            max_lt_id = np.argmax(transformed_fingers[:, 1])
            min_lt_id = np.argmin(transformed_fingers[:, 1])
            max_gt = transformed_fingers[max_gt_id]
            min_gt = transformed_fingers[min_gt_id]
            max_lt = transformed_fingers[max_lt_id]
            min_lt = transformed_fingers[min_lt_id]
            max_gt_full, max_lt_full, min_gt_full, min_lt_full = pca.inverse_transform(
                [max_gt, max_lt, min_gt, min_lt])

            # Calculate direction of maximum variance (X-axis or Y-axis)
            min_x = frame_indices[:, 0].min()
            max_x = frame_indices[:, 0].max()
            min_y = frame_indices[:, 1].min()
            max_y = frame_indices[:, 1].max()
            if max_x - min_x > max_y - min_y:
                max_gt, min_gt = max_gt_full[0], min_gt_full[0]
                max_lt, min_lt = max_lt_full[1], min_lt_full[1]
                center_gt, center_lt = finger_center[0], finger_center[1]
            else:
                max_gt, min_gt = max_gt_full[1], min_gt_full[1]
                max_lt, min_lt = max_lt_full[0], min_lt_full[0]
                center_gt, center_lt = finger_center[1], finger_center[0]

            var_gt = pca.components_[0]
            var_lt = pca.components_[1]

            if (var_gt[0] > 0 and var_gt[1] < 0) or (np.all(var_gt < 0)):
                var_gt = -var_gt

            if (var_lt[0] > 0 and var_lt[1] < 0) or (np.all(var_lt < 0)):
                var_lt = -var_lt

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

    if normalize:
        finger_sizes = np.sqrt(fingers_size(matrices))
        for i, finger_size in enumerate(finger_sizes):
            extreme_points[i] /= finger_size/10

    if move_to_origin:
        extreme_means = extreme_points.mean(axis=1, keepdims=True)
        extreme_points = extreme_points - extreme_means

    return extreme_points


def extract_extreme_points(matrices, normalize=False, move_to_origin=False):
    '''Given the sequence of matrices where masks for fingers are extracted, compute
    the extreme points of each mask in each frame. Extreme points are defined as
    the center of each edge, if the mask is considered to be a rectangle. Using
    PCA, those points can be calculated using the principal components, the
    center of the mask, and the distance between the border of the mask and the
    center.
    '''

    def rotate_point(point, center, angle):
        px, py = point
        cx, cy = center
        angle_rad = angle*np.pi/180
        px -= cx
        py -= cy
        sin = np.sin(angle_rad)
        cos = np.cos(angle_rad)
        newx = px * cos - py * sin + cx
        newy = px * sin + py * cos + cy

        return np.array((newx, newy))

    n_masks = matrices.max().astype(int)
    extreme_points = np.zeros((n_masks, matrices.shape[0], 4, 2))
    for idx in range(n_masks):
        finger = idx+1
        finger_indices = np.argwhere(matrices == finger)
        for n_frame in range(matrices.shape[0]):
            frame_indices = finger_indices[finger_indices[:, 0] == n_frame, 1:]

            (cx, cy), (w, h), angle = cv2.minAreaRect(
                frame_indices[:, [1, 0]])

            if h > w:
                h, w = w, h
                angle = angle - 90

            point_right = cx + w/2, cy
            point_left = cx - w/2, cy
            point_top = cx, cy - h/2
            point_bottom = cx, cy + h/2
            point_right = rotate_point(point_right, (cx, cy), angle)
            point_left = rotate_point(point_left, (cx, cy), angle)
            point_top = rotate_point(point_top, (cx, cy), angle)
            point_bottom = rotate_point(point_bottom, (cx, cy), angle)
            extreme_points[idx, n_frame] = [
                point_left[[1, 0]],
                point_right[[1, 0]],
                point_top[[1, 0]],
                point_bottom[[1, 0]]
            ]

    if normalize:
        finger_sizes = np.sqrt(fingers_size(matrices))
        for i, finger_size in enumerate(finger_sizes):
            extreme_points[i] /= finger_size/10
    if move_to_origin:
        extreme_means = extreme_points.mean(axis=1, keepdims=True)
        extreme_points = extreme_points - extreme_means

    return extreme_points


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
