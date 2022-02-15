import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

folder = 'Device_ON_2'
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pkl.load(f)

base_img = plt.imread(os.path.join(folder, "first_frame.png"))

onehot_masks = np.zeros((*masks.shape, masks.max() + 1))

it = np.nditer(masks, flags=['multi_index'])
for x in it:
    onehot_masks[(*it.multi_index, x)] = 1

mean_masks = onehot_masks.mean(axis=0)

fingers = mean_masks[..., 1] + mean_masks[..., 2]

left_finger = onehot_masks[..., 1]
right_finger = onehot_masks[..., 2]

mean_finger_left = np.count_nonzero(left_finger, axis=(1, 2)).mean()
total_area_left = np.count_nonzero(left_finger.sum(axis=0))

mean_finger_right = np.count_nonzero(right_finger, axis=(1, 2)).mean()
total_area_right = np.count_nonzero(right_finger.sum(axis=0))

plt.imshow(base_img)
plt.imshow(fingers, alpha=.8)
plt.text(5, 220, f"Movement index for first finger: {total_area_left / mean_finger_left:.4f}", color="white")
plt.text(5, 240, f"Movement index for second finger: {total_area_right / mean_finger_right:.4f}", color="white")
plt.show()
