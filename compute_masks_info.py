import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import sklearn.decomposition

from visualization import matrices_to_video
from visualization.plotting import plot_movements, plot_speeds
from masks_manipulation import extract_centers

folder = '../MiVOS/Device_ON'
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pkl.load(f)

centers = extract_centers(masks)

finger_1_centers = centers[0]
plot_movements(centers)
plot_speeds(centers)
'''

for i, frame in enumerate(masks):
    frame[centers[0,i,0], centers[0,i,1]] = 3
    frame[centers[1,i,0], centers[1,i,1]] = 4

print("Saving video")
matrices = matrices_to_video(masks, './trial.mp4')
