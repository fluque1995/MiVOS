import pickle as pkl
import os
from masks_manipulation import fingers_size

folder = 'results/P1/Visita_1_OFF/Dedos_enfrentados'
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pkl.load(f)

fingers_size(masks)
