import matplotlib.pyplot as plt
import os
import pickle as pkl
import numpy.fft

from visualization import matrices_to_video
from visualization.plotting import plot_movements
from masks_manipulation import extract_centers, extract_extreme_points

plt.style.use('ggplot')
folder = 'results/P1/Visita_2_ON/Dedos_enfrentados/'
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pkl.load(f)

centers = extract_centers(masks)
plot_movements(centers)

norm_centers = centers - centers.mean(axis=1, keepdims=True)

finger_1_x = norm_centers[0, :, 0]

spectrum = numpy.fft.fft(finger_1_x)
freq = numpy.fft.fftfreq(len(spectrum))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(freq, abs(spectrum))
fig.show()

# # plot_movements(centers)
# # plot_speeds(centers)
# extreme_points = extract_extreme_points(masks)

# plt.imshow(masks[0])
# plt.show()

# for i in range(masks.shape[0]):
#     for finger in extreme_points[:, i, ...]:
#         pt1, pt2, pt3, pt4 = finger.reshape((-1, 2))
#         masks[i, pt1[0], pt1[1]] = i+3
#         masks[i, pt2[0], pt2[1]] = i+3
#         masks[i, pt3[0], pt3[1]] = i+3
#         masks[i, pt4[0], pt4[1]] = i+3


# plt.imshow(masks[0])
# plt.show()

# print("Saving video")
# matrices = matrices_to_video(masks, './trial.mp4')
