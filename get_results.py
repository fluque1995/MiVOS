import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from masks_manipulation import extract_centers, frequency_and_magnitude, movement_index
from visualization.plotting import plot_movements, plot_finger_heatmaps

masks_dir = '../Resultados/'
output_dir = '../Results/'

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

for patient in os.listdir(masks_dir):
    for visit in os.listdir(os.path.join(masks_dir, patient)):
        for experiment in os.listdir(os.path.join(masks_dir, patient, visit)):

            curr_path = os.path.join(masks_dir, patient, visit, experiment)
            masks_path = os.path.join(curr_path, 'masks.pkl')
            first_frame_path = os.path.join(curr_path, 'first_frame.png')
            output_path = os.path.join(output_dir, patient, visit, experiment)

            if os.path.exists(masks_path):
                os.makedirs(output_path, exist_ok=True)
                with open(masks_path, "rb") as f:
                    masks = pkl.load(f)
                    first_frame = cv2.cvtColor(
                        cv2.imread(first_frame_path), cv2.COLOR_BGR2RGB)
                    print(f'{patient}-{visit}-{experiment} loaded.')

                    print('Extracting centers...')
                    curr_centers = extract_centers(masks, normalize=True, move_to_origin=True)

                    print('Computing frequency for the whole video...')
                    freq_and_mags = frequency_and_magnitude(curr_centers, fps=30)
                    f1_x_whole = freq_and_mags[list(freq_and_mags)[0]]['x_freq']
                    m1_x_whole = freq_and_mags[(list(freq_and_mags)[0])]['x_mag']
                    f1_y_whole = freq_and_mags[(list(freq_and_mags)[0])]['y_freq']
                    m1_y_whole = freq_and_mags[(list(freq_and_mags)[0])]['y_mag']
                    f2_x_whole = '-'
                    m2_x_whole = '-'
                    f2_y_whole = '-'
                    m2_y_whole = '-'

                    if len(freq_and_mags) > 1:
                        f2_x_whole = freq_and_mags[(list(freq_and_mags)[1])]['x_freq']
                        m2_x_whole = freq_and_mags[(list(freq_and_mags)[1])]['x_mag']
                        f2_y_whole = freq_and_mags[(list(freq_and_mags)[1])]['y_freq']
                        m2_y_whole = freq_and_mags[(list(freq_and_mags)[1])]['y_mag']

                    print('Computing frequency with temporal_window=30...')
                    freq_and_mags_30 = frequency_and_magnitude(curr_centers, fps=30, temporal_window=30)
                    f1_x_30 = freq_and_mags_30[list(freq_and_mags_30)[0]]['x_freq']
                    m1_x_30 = freq_and_mags_30[list(freq_and_mags_30)[0]]['x_mag']
                    f1_y_30 = freq_and_mags_30[list(freq_and_mags_30)[0]]['y_freq']
                    m1_y_30 = freq_and_mags_30[list(freq_and_mags_30)[0]]['y_mag']
                    f2_x_30 = '-'
                    m2_x_30 = '-'
                    f2_y_30 = '-'
                    m2_y_30 = '-'

                    if len(freq_and_mags_30) > 1:
                        f2_x_30 = freq_and_mags_30[list(freq_and_mags_30)[1]]['x_freq']
                        m2_x_30 = freq_and_mags_30[list(freq_and_mags_30)[1]]['x_mag']
                        f2_y_30 = freq_and_mags_30[list(freq_and_mags_30)[1]]['y_freq']
                        m2_y_30 = freq_and_mags_30[list(freq_and_mags_30)[1]]['y_mag']

                    print('Computing frequency with temporal_window=60...')
                    freq_and_mags_60 = frequency_and_magnitude(curr_centers, fps=30, temporal_window=60)
                    f1_x_60 = freq_and_mags_60[list(freq_and_mags_60)[0]]['x_freq']
                    m1_x_60 = freq_and_mags_60[list(freq_and_mags_60)[0]]['x_mag']
                    f1_y_60 = freq_and_mags_60[list(freq_and_mags_60)[0]]['y_freq']
                    m1_y_60 = freq_and_mags_60[list(freq_and_mags_60)[0]]['y_mag']
                    f2_x_60 = '-'
                    m2_x_60 = '-'
                    f2_y_60 = '-'
                    m2_y_60 = '-'

                    if len(freq_and_mags_60) > 1:
                        f2_x_60 = freq_and_mags_60[list(freq_and_mags_60)[1]]['x_freq']
                        m2_x_60 = freq_and_mags_60[list(freq_and_mags_60)[1]]['x_mag']
                        f2_y_60 = freq_and_mags_60[list(freq_and_mags_60)[1]]['y_freq']
                        m2_y_60 = freq_and_mags_60[list(freq_and_mags_60)[1]]['y_mag']

                    print('Plotting movements...')
                    plot_movements(curr_centers, saving_path=os.path.join(output_path, 'movement.png'))

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
