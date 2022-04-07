import os
import io_utils

from visualization import matrices_to_video
from masks_manipulation import extract_extreme_points

path = '../Mascaras/P2/Visita_1_OFF/Suero_izq'
output_vid = 'output_der.mp4'

masks = io_utils.load_masks(os.path.join(path, 'masks.pkl'))
extreme_points = extract_extreme_points(masks)

for i, mask in enumerate(masks):
    curr_frame_points = extreme_points[:, i, ...]

    for k, pt in enumerate(curr_frame_points.reshape(8, 2)):
        if pt[0]+1 < mask.shape[0] and pt[1]+1 < mask.shape[1]:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    mask[pt[0].astype(int)+i, pt[1].astype(int)+j] = 3+(k % 4)


matrices_to_video(masks, output_vid)
