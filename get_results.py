import csv
import cv2
import io_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import (
    extract_centers, extract_extreme_points,
    frequency_and_magnitude, movement_index
)
from visualization.plotting import plot_movements, plot_finger_heatmaps

masks_dir = '../Mascaras/'
output_dir = '../Results_pre_savgol/'

os.makedirs(output_dir, exist_ok=True)

statistics_csv = open(os.path.join(output_dir, 'statistics.csv'), 'w', newline='')
writer = csv.writer(statistics_csv)

header = ['Paciente', 'Visita', 'Experimento',
          'Frecuencia_x_dedo_1_completo', 'Magnitud_x_dedo_1_completo',
          'Frecuencia_y_dedo_1_completo', 'Magnitud_y_dedo_1_completo',
          'Frecuencia_x_dedo_2_completo', 'Magnitud_x_dedo_2_completo',
          'Frecuencia_y_dedo_2_completo', 'Magnitud_y_dedo_2_completo',
          'Frecuencia_x_dedo_1_30', 'Magnitud_x_dedo_1_30',
          'Frecuencia_y_dedo_1_30', 'Magnitud_y_dedo_1_30',
          'Frecuencia_x_dedo_2_30', 'Magnitud_x_dedo_2_30',
          'Frecuencia_y_dedo_2_30', 'Magnitud_y_dedo_2_30',
          'Frecuencia_x_dedo_1_60', 'Magnitud_x_dedo_1_60',
          'Frecuencia_y_dedo_1_60', 'Magnitud_y_dedo_1_60',
          'Frecuencia_x_dedo_2_60', 'Magnitud_x_dedo_2_60',
          'Frecuencia_y_dedo_2_60', 'Magnitud_y_dedo_2_60',
          'Movimiento_dedo_1_completo', 'Movimiento_dedo_2_completo',
          'Movimiento_dedo_1_30', 'Movimiento_dedo_2_30',
          'Movimiento_dedo_1_60', 'Movimiento_dedo_2_60']

writer.writerow(header)

for patient in sorted(os.listdir(masks_dir)):
    for visit in sorted(os.listdir(os.path.join(masks_dir, patient))):
        for experiment in sorted(os.listdir(os.path.join(masks_dir, patient, visit))):
            # Do not process nose-finger experiment with these methods
            if experiment[0:3] == 'D-N':
                continue

            curr_path = os.path.join(masks_dir, patient, visit, experiment)
            masks_path = os.path.join(curr_path, 'masks.pkl')
            first_frame_path = os.path.join(curr_path, 'first_frame.png')
            output_path = os.path.join(output_dir, patient, visit, experiment)

            if os.path.exists(masks_path):
                os.makedirs(output_path, exist_ok=True)
                masks = io_utils.load_masks(masks_path)
                first_frame = cv2.cvtColor(
                    cv2.imread(first_frame_path), cv2.COLOR_BGR2RGB)
                print(f'{patient}-{visit}-{experiment} loaded.')

                if experiment[0:5] == 'Suero':
                    curr_points = extract_extreme_points(masks, normalize=False,
                                                         move_to_origin=True)
                    if experiment == 'Suero_der':
                        curr_points = curr_points[:, :, 1, :]
                    elif experiment == 'Suero_izq':
                        curr_points = curr_points[:, :, 0, :]
                    else:
                        print("This experiment is not supported")
                        continue

                else:
                    print('Extracting centers...')
                    curr_points = extract_centers(masks, normalize=True,
                                                  move_to_origin=True)

                print('Computing frequency for the whole video...')
                freq_and_mags = frequency_and_magnitude(curr_points, fps=30)
                f1 = freq_and_mags[list(freq_and_mags)[0]] if not np.isnan(freq_and_mags[list(freq_and_mags)[0]]['x']['max_mag']) else {}
                f2 = freq_and_mags[list(freq_and_mags)[1]] if len(freq_and_mags) > 1 else {}

                f1_x_whole = f1['x']['max_freq'] if len(f1) > 0 else '-'
                m1_x_whole = f1['x']['max_mag'] if len(f1) > 0 else '-'
                f1_y_whole = f1['y']['max_freq'] if len(f1) > 0 else '-'
                m1_y_whole = f1['y']['max_mag'] if len(f1) > 0 else '-'
                f2_x_whole = f2['y']['max_freq'] if len(f2) > 0 else '-'
                m2_x_whole = f2['x']['max_mag'] if len(f2) > 0 else '-'
                f2_y_whole = f2['y']['max_freq'] if len(f2) > 0 else '-'
                m2_y_whole = f2['y']['max_mag'] if len(f2) > 0 else '-'

                print('Computing frequency with temporal_window=30...')
                freq_and_mags_30 = frequency_and_magnitude(curr_points, fps=30, temporal_window=30)
                f1_30 = freq_and_mags_30[list(freq_and_mags_30)[0]] if not np.isnan(freq_and_mags_30[list(freq_and_mags_30)[0]]['x_mag']).all() else {}
                f2_30 = freq_and_mags_30[list(freq_and_mags_30)[1]] if len(freq_and_mags_30) > 1 else {}

                f1_x_30 = f1_30['x_freq'] if len(f1_30) > 0 else '-'
                m1_x_30 = f1_30['x_mag'] if len(f1_30) > 0 else '-'
                f1_y_30 = f1_30['y_freq'] if len(f1_30) > 0 else '-'
                m1_y_30 = f1_30['y_mag'] if len(f1_30) > 0 else '-'
                f2_x_30 = f2_30['x_freq'] if len(f2_30) > 0 else '-'
                m2_x_30 = f2_30['x_mag'] if len(f2_30) > 0 else '-'
                f2_y_30 = f2_30['y_freq'] if len(f2_30) > 0 else '-'
                m2_y_30 = f2_30['y_mag'] if len(f2_30) > 0 else '-'

                print('Computing frequency with temporal_window=60...')
                freq_and_mags_60 = frequency_and_magnitude(curr_points, fps=30, temporal_window=60)
                f1_60 = freq_and_mags_60[list(freq_and_mags_60)[0]] if not np.isnan(freq_and_mags_60[list(freq_and_mags_60)[0]]['x_mag']).all() else {}
                f2_60 = freq_and_mags_60[list(freq_and_mags_60)[1]] if len(freq_and_mags_60) > 1 else {}

                f1_x_60 = f1_60['x_freq'] if len(f1_60) > 0 else '-'
                m1_x_60 = f1_60['x_mag'] if len(f1_60) > 0 else '-'
                f1_y_60 = f1_60['y_freq'] if len(f1_60) > 0 else '-'
                m1_y_60 = f1_60['y_mag'] if len(f1_60) > 0 else '-'
                f2_x_60 = f2_60['x_freq'] if len(f2_60) > 0 else '-'
                m2_x_60 = f2_60['x_mag'] if len(f2_60) > 0 else '-'
                f2_y_60 = f2_60['y_freq'] if len(f2_60) > 0 else '-'
                m2_y_60 = f2_60['y_mag'] if len(f2_60) > 0 else '-'

                print('Plotting movements...')
                if experiment[:3] == 'D-N':
                    x_limit = 50
                    y_limit = 150
                elif experiment[:5] == 'Suero':
                    x_limit = None
                    y_limit = None
                else:
                    x_limit = 20
                    y_limit = 20

                plot_movements(curr_points, x_limit=x_limit, y_limit=y_limit,
                               saving_path=os.path.join(output_path, 'movement.png'))
                print('Computing movement index for the whole video...')
                mov_indices = movement_index(masks)
                mv1_whole = mov_indices[list(mov_indices)[0]]
                mv2_whole = '-'

                if len(mov_indices) > 1:
                    mv2_whole = mov_indices[list(mov_indices)[1]]

                print('Computing movement index with temporal_window=30...')
                mov_indices_30 = movement_index(masks, temporal_window=30)
                mv1_30 = mov_indices_30[list(mov_indices_30)[0]]
                mv2_30 = '-'

                if len(mov_indices_30) > 1:
                    mv2_30 = mov_indices_30[list(mov_indices_30)[1]]

                print('Computing movement index with temporal_window=60...')
                mov_indices_60 = movement_index(masks, temporal_window=60)
                mv1_60 = mov_indices_60[list(mov_indices_60)[0]]
                mv2_60 = '-'

                if len(mov_indices_60) > 1:
                    mv2_60 = mov_indices_60[list(mov_indices_60)[1]]

                print('Plotting finger heatmaps...')
                plot_finger_heatmaps([masks], [first_frame], movement_index=True, saving_path=os.path.join(output_path, 'heatmap.png'))
                print('Writing results...')
                writer.writerow([patient, visit, experiment,
                                 f1_x_whole, m1_x_whole,
                                 f1_y_whole, m1_y_whole,
                                 f2_x_whole, m2_x_whole,
                                 f2_y_whole, m2_y_whole,
                                 f1_x_30, m1_x_30,
                                 f1_y_30, m1_y_30,
                                 f2_x_30, m2_x_30,
                                 f2_y_30, m2_y_30,
                                 f1_x_60, m1_x_60,
                                 f1_y_60, m1_y_60,
                                 f2_x_60, m2_x_60,
                                 f2_y_60, m2_y_60,
                                 mv1_whole, mv2_whole,
                                 mv1_30, mv2_30,
                                 mv1_60, mv2_60])
            else:
                print(f'{patient}-{visit}-{experiment} masks not found. Skipping.')

statistics_csv.close()
