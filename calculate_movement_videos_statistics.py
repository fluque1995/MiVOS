import csv
import io_utils
import logging
import matplotlib.pyplot as plt
import os

from itertools import product
from masks_manipulation import extract_centers, savgol_smoothing
from visualization.plotting import plot_movements


masks_folder = '../Mascaras_efficientnet'
output_folder = '../Resultados_efficientnet'

combined_paths = product(
    [f"P{i+1}" for i in range(6)],  # PATIENT
    ["Visita_1_OFF", "Visita_2_ON", "Visita_3_ON"],  # VISIT
    ["D-N_izq", "D-N_der"],  # VIDEO NAME
)

stats_csv = open(os.path.join(output_folder, "statistics_movement.csv"), "w", newline="")

writer = csv.writer(stats_csv)
dataset_header = [
    "Paciente",
    "Visita",
    "Experimento",
    "STD_v_dedo_1",
    "STD_h_dedo_1",
    "STD_v_dedo_2",
    "STD_h_dedo_2",
]
writer.writerow(dataset_header)

for patient, visit, experiment in combined_paths:
    curr_row = [patient, visit, experiment]
    print(f"Working on {patient} - {visit} - {experiment}")
    masks_file = os.path.join(masks_folder, patient, visit, experiment, "masks.pkl")
    graphs_folder = os.path.join(output_folder, patient, visit, experiment)
    try:
        masks = io_utils.load_masks(masks_file)
    except:
        logging.warning(
            f"No masks found for {patient} - {visit} - {experiment}. Skipping..."
        )
        continue

    os.makedirs(graphs_folder, exist_ok=True)
    centers = extract_centers(masks, normalize=True, move_to_origin=True)
    smoothed_centers = savgol_smoothing(centers, 9, 2)

    plot_movements(
        centers,
        x_limit=None,
        y_limit=None,
        saving_path=os.path.join(graphs_folder, "movement_original.png"),
    )
    plot_movements(
        smoothed_centers,
        x_limit=None,
        y_limit=None,
        saving_path=os.path.join(graphs_folder, "movement_smoothed.png"),
    )

    differences = centers - smoothed_centers


    csv_row = [patient, visit, experiment]
    with plt.style.context(('ggplot')):

        fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True)
        fig.suptitle(f"{patient} - {visit} - {experiment}")
        plt.subplots_adjust(hspace=0.3)

        ax = ax.ravel()
        curr_data = differences[0, :, 0]
        csv_row.append(curr_data.std())
        ax[0].hist(curr_data, bins=50, range=[-10, 10],
                   label=f"STD: {curr_data.std():.3f}", edgecolor='black')
        ax[0].legend(prop={'size': 6})
        ax[0].set_title("Right finger, vertical movement", fontsize='small')

        curr_data = differences[0, :, 1]
        csv_row.append(curr_data.std())
        ax[1].hist(curr_data, bins=50, range=[-10, 10],
                   label=f"STD: {curr_data.std():.3f}", edgecolor='black')
        ax[1].legend(prop={'size': 6})
        ax[1].set_title("Right finger, horizontal movement", fontsize='small')

        curr_data = differences[1, :, 0]
        csv_row.append(curr_data.std())
        ax[2].hist(curr_data, bins=50, range=[-10, 10], color='#348ABD',
                   label=f"STD: {curr_data.std():.3f}", edgecolor='black')
        ax[2].legend(prop={'size': 6})
        ax[2].set_title("Left finger, vertical movement", fontsize='small')

        curr_data = differences[1, :, 1]
        csv_row.append(curr_data.std())
        ax[3].hist(curr_data, bins=50, range=[-10, 10], color='#348ABD',
                   label=f"STD: {curr_data.std():.3f}", edgecolor='black')
        ax[3].legend(prop={'size': 6})
        ax[3].set_title("Left finger, horizontal movement", fontsize='small')

    fig.savefig(os.path.join(graphs_folder, "stds.png"), bbox_inches='tight')
    writer.writerow(csv_row)

stats_csv.close()
