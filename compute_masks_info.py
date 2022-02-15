import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import sklearn.decomposition

from visualization import matrices_to_video
from visualization.plotting import plot_movements, plot_speeds
from masks_manipulation import extract_centers, extract_extreme_points

folder = '../MiVOS/Device_OFF'
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pkl.load(f)

# plot_movements(centers)
# plot_speeds(centers)
extreme_points = extract_extreme_points(masks)

plt.imshow(masks[0])
plt.show()

for i in range(masks.shape[0]):
    for finger in extreme_points[:, i, ...]:
        pt1, pt2, pt3, pt4 = finger.reshape((-1, 2))
        masks[i, pt1[0], pt1[1]] = i+3
        masks[i, pt2[0], pt2[1]] = i+3
        masks[i, pt3[0], pt3[1]] = i+3
        masks[i, pt4[0], pt4[1]] = i+3


plt.imshow(masks[0])
plt.show()

print("Saving video")
matrices = matrices_to_video(masks, './trial.mp4')
