import csv
import cv2
import io_utils
import numpy as np
import os

from masks_manipulation import (
    extract_centers,
    extract_extreme_points,
    frequency_and_magnitude,
    movement_index,
)
from visualization.plotting import plot_movements, plot_finger_heatmaps

masks_dir = "../Mascaras/"
output_dir = "../Resultados/"

graph_file_extensions = ".pdf"

os.makedirs(output_dir, exist_ok=True)

statistics_csv = open(os.path.join(output_dir, "statistics.csv"), "w", newline="")
writer = csv.writer(statistics_csv)

dataset_header = [
    "Paciente",
    "Visita",
    "Experimento",
    "Frecuencia_v_dedo_1_completo",
    "Magnitud_v_dedo_1_completo",
    "Frecuencia_h_dedo_1_completo",
    "Magnitud_h_dedo_1_completo",
    "Frecuencia_v_dedo_2_completo",
    "Magnitud_v_dedo_2_completo",
    "Frecuencia_h_dedo_2_completo",
    "Magnitud_h_dedo_2_completo",
    "Movimiento_dedo_1_completo",
    "Movimiento_dedo_2_completo",
    "Frecuencia_v_dedo_1_60",
    "Magnitud_v_dedo_1_60",
    "Frecuencia_h_dedo_1_60",
    "Magnitud_h_dedo_1_60",
    "Frecuencia_v_dedo_2_60",
    "Magnitud_v_dedo_2_60",
    "Frecuencia_h_dedo_2_60",
    "Magnitud_h_dedo_2_60",
    "Movimiento_dedo_1_60",
    "Movimiento_dedo_2_60",
]

writer.writerow(dataset_header)

for patient in sorted(os.listdir(masks_dir)):
    for visit in sorted(os.listdir(os.path.join(masks_dir, patient))):
        for experiment in sorted(os.listdir(os.path.join(masks_dir, patient, visit))):
            # Do not process nose-finger experiment with these methods
            if experiment[0:3] == "D-N":
                continue

            curr_path = os.path.join(masks_dir, patient, visit, experiment)
            masks_path = os.path.join(curr_path, "masks.pkl")
            first_frame_path = os.path.join(curr_path, "first_frame.png")
            output_path = os.path.join(output_dir, patient, visit, experiment)

            if os.path.exists(masks_path):
                os.makedirs(output_path, exist_ok=True)
                masks = io_utils.load_masks(masks_path)
                first_frame = cv2.cvtColor(
                    cv2.imread(first_frame_path), cv2.COLOR_BGR2RGB
                )
                print(f"{patient}-{visit}-{experiment} loaded.")

                if experiment[0:5] == "Suero":
                    extremes = extract_extreme_points(
                        masks, normalize=True, move_to_origin=True
                    )
                    centers = extract_centers(
                        masks, normalize=True, move_to_origin=True
                    )
                    if experiment == "Suero_der":
                        bottle = extremes[0, :, 1, :]
                        finger = centers[1]
                        curr_points = np.stack([bottle, finger])
                    elif experiment == "Suero_izq":
                        finger = centers[0]
                        bottle = extremes[1, :, 0, :]
                        curr_points = np.stack([finger, bottle])
                    else:
                        print("This experiment is not supported")
                        continue

                else:
                    print("Extracting centers...")
                    curr_points = extract_centers(
                        masks, normalize=True, move_to_origin=True
                    )

                print("Computing frequency for the whole video...")
                freq_and_mags = frequency_and_magnitude(
                    curr_points,
                    fps=30,
                    graph_path=os.path.join(
                        output_path, "fourier" + graph_file_extensions
                    ),
                )

                try:
                    f1 = freq_and_mags[0]
                except:
                    f1 = []
                try:
                    f2 = freq_and_mags[1]
                except:
                    f2 = []
                f1_v_whole = f1["v"]["max_freq"] if len(f1) > 0 else "-"
                m1_v_whole = f1["v"]["max_mag"] if len(f1) > 0 else "-"
                f1_h_whole = f1["h"]["max_freq"] if len(f1) > 0 else "-"
                m1_h_whole = f1["h"]["max_mag"] if len(f1) > 0 else "-"
                f2_v_whole = f2["v"]["max_freq"] if len(f2) > 0 else "-"
                m2_v_whole = f2["v"]["max_mag"] if len(f2) > 0 else "-"
                f2_h_whole = f2["h"]["max_freq"] if len(f2) > 0 else "-"
                m2_h_whole = f2["h"]["max_mag"] if len(f2) > 0 else "-"

                print("Computing frequency with temporal_window=60...")
                freq_and_mags_60 = frequency_and_magnitude(
                    curr_points, fps=30, temporal_window=60,
                    graph_path=os.path.join(
                        output_path, "fourier" + graph_file_extensions
                    )
                )
                try:
                    f1_60 = freq_and_mags_60[0]
                except:
                    f1_60 = []
                try:
                    f2_60 = freq_and_mags_60[1]
                except:
                    f2_60 = []

                f1_v_60 = np.mean(f1_60["v_freq"]) if len(f1_60) > 0 else "-"
                m1_v_60 = np.mean(f1_60["v_mag"]) if len(f1_60) > 0 else "-"
                f1_h_60 = np.mean(f1_60["h_freq"]) if len(f1_60) > 0 else "-"
                m1_h_60 = np.mean(f1_60["h_mag"]) if len(f1_60) > 0 else "-"
                f2_v_60 = np.mean(f2_60["v_freq"]) if len(f2_60) > 0 else "-"
                m2_v_60 = np.mean(f2_60["v_mag"]) if len(f2_60) > 0 else "-"
                f2_h_60 = np.mean(f2_60["h_freq"]) if len(f2_60) > 0 else "-"
                m2_h_60 = np.mean(f2_60["h_mag"]) if len(f2_60) > 0 else "-"

                print("Plotting movements...")
                if experiment[:3] == "D-N":
                    x_limit = 50
                    y_limit = 150
                elif experiment[:5] == "Suero":
                    x_limit = None
                    y_limit = None
                else:
                    x_limit = 20
                    y_limit = 20

                plot_movements(
                    curr_points,
                    x_limit=x_limit,
                    y_limit=y_limit,
                    saving_path=os.path.join(
                        output_path, "movement" + graph_file_extensions
                    ),
                )
                print("Computing movement index for the whole video...")
                mov_indices = movement_index(masks)
                mv1_whole = mov_indices[list(mov_indices)[0]]
                mv2_whole = "-"

                if len(mov_indices) > 1:
                    mv2_whole = mov_indices[list(mov_indices)[1]]

                print("Computing movement index with temporal_window=60...")
                mov_indices_60 = movement_index(masks, temporal_window=60)
                mv1_60 = np.mean(mov_indices_60[list(mov_indices_60)[0]])
                mv2_60 = "-"

                if len(mov_indices_60) > 1:
                    mv2_60 = np.mean(mov_indices_60[list(mov_indices_60)[1]])

                print("Plotting finger heatmaps...")
                plot_finger_heatmaps(
                    [masks],
                    [first_frame],
                    movement_index=True,
                    saving_path=os.path.join(
                        output_path, "heatmap" + graph_file_extensions
                    ),
                )
                print("Writing results...")
                writer.writerow(
                    [
                        patient,
                        visit,
                        experiment,
                        f1_v_whole,
                        m1_v_whole,
                        f1_h_whole,
                        m1_h_whole,
                        f2_v_whole,
                        m2_v_whole,
                        f2_h_whole,
                        m2_h_whole,
                        mv1_whole,
                        mv2_whole,
                        f1_v_60,
                        m1_v_60,
                        f1_h_60,
                        m1_h_60,
                        f2_v_60,
                        m2_v_60,
                        f2_h_60,
                        m2_h_60,
                        mv1_60,
                        mv2_60,
                    ]
                )
            else:
                print(f"{patient}-{visit}-{experiment} masks not found. Skipping.")

statistics_csv.close()
