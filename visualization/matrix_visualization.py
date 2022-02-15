import cv2
import numpy as np
import os
import typing


def matrices_to_video(matrices: np.array, vid_path: str, scale_factor=2):
    """Saves a group of matrices representing a sequence of masks as a video

    Keyword arguments:
    matrices -- Sequence of matrices
    vid_path -- Path where the video is going to be stored
    """
    vid_matrices = np.zeros((*matrices.shape, 3)).astype(np.uint8)
    vid_matrices[matrices == 1] = (0, 0, 255)
    vid_matrices[matrices == 2] = (0, 255, 0)
    vid_matrices[matrices == 3] = (0, 255, 255)
    vid_matrices[matrices == 4] = (255, 0, 255)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        vid_path, fourcc, 20.0,
        (scale_factor*vid_matrices.shape[2], scale_factor*vid_matrices.shape[1]))

    for mat in vid_matrices:
        frame = cv2.resize(mat, None, fx=scale_factor, fy=scale_factor)
        out.write(frame)

    out.release()
    return vid_matrices
