import cv2
import pickle as pkl
import visualization.plotting
import masks_manipulation

# Masks visualization and full video movement index calculation
with open("results/P1/Visita_1_OFF/Dedos_enfrentados/masks.pkl", "rb") as f:
    mask1 = pkl.load(f)
img1 = cv2.cvtColor(
    cv2.imread("results/P1/Visita_1_OFF/Dedos_enfrentados/first_frame.png"),
    cv2.COLOR_BGR2RGB)

with open("results/P1/Visita_2_ON/Dedos_enfrentados/masks.pkl", "rb") as f:
    mask2 = pkl.load(f)
img2 = cv2.cvtColor(
    cv2.imread("results/P1/Visita_2_ON/Dedos_enfrentados/first_frame.png"),
    cv2.COLOR_BGR2RGB)


with open("results/P1/Visita_1_OFF/Suero_der/masks.pkl", "rb") as f:
    mask3 = pkl.load(f)
img3 = cv2.cvtColor(
    cv2.imread("results/P1/Visita_1_OFF/Suero_der/first_frame.png"),
    cv2.COLOR_BGR2RGB)


with open("results/P1/Visita_2_ON/Suero_der/masks.pkl", "rb") as f:
    mask4 = pkl.load(f)
img4 = cv2.cvtColor(cv2.imread("results/P1/Visita_2_ON/Suero_der/first_frame.png"),
                    cv2.COLOR_BGR2RGB)

visualization.plotting.plot_finger_heatmaps(
    [mask1, mask2, mask3, mask4],
    [img1, img2, img3, img4],
    (2, 2),
    movement_index=True)

with open("results/P3/Visita_1_OFF/Extension/masks.pkl", "rb") as f:
    mask5 = pkl.load(f)
img5 = cv2.cvtColor(
    cv2.imread("results/P3/Visita_1_OFF/Extension/first_frame.png"),
    cv2.COLOR_BGR2RGB)

with open("results/P3/Visita_2_ON/Extension/masks.pkl", "rb") as f:
    mask6 = pkl.load(f)
img6 = cv2.cvtColor(
    cv2.imread("results/P3/Visita_2_ON/Extension/first_frame.png"),
    cv2.COLOR_BGR2RGB)

visualization.plotting.plot_finger_heatmaps(
    [mask5, mask6],
    [img5, img6],
    (2, 1),
    movement_index=True)

movement_index_full = masks_manipulation.movement_index(mask5)
movement_index_windows = masks_manipulation.movement_index(mask5, temporal_window=30)

for finger, values in movement_index_full.items():
    print(f"Dedo {finger} - valor: {values}")

print()

for finger, values in movement_index_windows.items():
    print(f"Dedo {finger} - valores")
    for val in values:
        print(val)
