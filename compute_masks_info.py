import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import numpy.fft
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult, STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import savgol_filter

from visualization import matrices_to_video
from visualization.plotting import plot_movements, plot_overlapped_movements
from masks_manipulation import extract_centers, extract_extreme_points, frequency_and_magnitude, savgol_smoothing

folder = '../Resultados_unificados/P1/Visita_1_OFF/D-N_der/'
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks11 = pkl.load(f)

folder2 = '../Resultados_unificados/P1/Visita_3_ON/D-N_der/'
with open(os.path.join(folder2, "masks.pkl"), "rb") as f:
    masks13 = pkl.load(f)

output_path = '../Grid_search/'
os.makedirs(output_path, exist_ok=True)

'''def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n'''

centers11 = extract_centers(masks11, normalize=True, move_to_origin=True)
centers13 = extract_centers(masks13, normalize=True, move_to_origin=True)

plot_movements(centers11, y_limit=30, x_limit=100, saving_path=os.path.join(output_path, 'orig_11.png'))
plot_movements(centers13, y_limit=30, x_limit=100, saving_path=os.path.join(output_path, 'orig_13.png'))

win_len = [3, 5, 7, 9]
pol_or = [1, 2, 3, 4, 5]

for wl in win_len:
    for po in pol_or:
        if po < wl:
            out = os.path.join(output_path, f'wl{wl}-po{po}')
            os.makedirs(out, exist_ok=True)

            smoothed_fingers11 = savgol_smoothing(centers11, wl, po)
            smoothed_fingers13 = savgol_smoothing(centers13, wl, po)

            plot_movements(smoothed_fingers11, y_limit=30, x_limit=100, saving_path=os.path.join(out, '11.png'))
            plot_movements(smoothed_fingers13, y_limit=30, x_limit=100, saving_path=os.path.join(out, '13.png'))

            plot_overlapped_movements([centers11, smoothed_fingers11], y_limit=30, x_limit=100, saving_path=os.path.join(out, 'ov_11.png'))
            plot_overlapped_movements([centers13, smoothed_fingers13], y_limit=30, x_limit=100, saving_path=os.path.join(out, 'ov_13.png'))


'''#sc = moving_average(centers)
d0x = savgol_filter(centers[0, :, 0], window_length=9, polyorder=2)
d0y = savgol_filter(centers[0, :, 1], window_length=9, polyorder=2)
d0 = np.stack((d0x, d0y), axis=1)
d1x = savgol_filter(centers[1, :, 0], window_length=9, polyorder=2)
d1y = savgol_filter(centers[1, :, 1], window_length=9, polyorder=2)
d1 = np.stack((d1x, d1y), axis=1)
fingers = np.stack((d0, d1), axis=0)

plot_movements(fingers)

d0x2 = savgol_filter(centers2[0, :, 0], window_length=9, polyorder=2)
d0y2 = savgol_filter(centers2[0, :, 1], window_length=9, polyorder=2)
d02 = np.stack((d0x2, d0y2), axis=1)
d1x2 = savgol_filter(centers2[1, :, 0], window_length=9, polyorder=2)
d1y2 = savgol_filter(centers2[1, :, 1], window_length=9, polyorder=2)
d12 = np.stack((d1x2, d1y2), axis=1)
fingers2 = np.stack((d02, d12), axis=0)

plot_movements(fingers2)

#TODO:
#   - Interpoleision de centros cuando no hay máscaras
#   - Sacar valor de amplitud interpretable
plt.show()'''
'''for i, finger in enumerate(centers):
    result_y = seasonal_decompose(finger[:, 0], model='additive', period=70)
    #result_y = STL(finger[:, 0], period=80).fit()
    result_y.plot()
    #DecomposeResult(finger[:, 0], result_y[1], result_y[0], result_y[1]).plot()
    result_x = seasonal_decompose(finger[:, 1], model='additive', period=70)
    #result_x = STL(finger[:, 1], period=80).fit()
    result_x.plot()
    #DecomposeResult(finger[:, 1], result_x[1], result_x[0], result_x[1]).plot()

plt.show()'''

'''norm_centers = centers - centers.mean(axis=1, keepdims=True)

finger_1_x = norm_centers[0, :, 0]

spectrum = numpy.fft.fft(finger_1_x)
freq = numpy.fft.fftfreq(len(spectrum))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(freq, abs(spectrum))
fig.show()'''

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
