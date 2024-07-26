"""
Peaks_finding.py

This file finds a corresponding local maxima in the brow signal when a maxima is found in a blink signal.

The parameters chosen in this script are NOT EXPECTED TO BE CHANGED, since they were found by experimentation. The only
one that can be changed is the channel, since any of the signals coming from each can be analyzed.

Last edition: 26/07/2024
Author: Javier Hernández Rubia
"""


import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.signal import find_peaks
import argparse


def low_pass_filter(data, window_size=3):
    filtered_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return filtered_data

def find_local_maxima(signal):
    local_maxima_indices = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            local_maxima_indices.append(i)
    return local_maxima_indices


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--eye', required=True)
parser.add_argument('-b', '--brow', required=True)
args = parser.parse_args()


eye = args.eye
brow = args.brow

# getting the information of the third point of the eye.
def parkinson_blink (eyesignal, browsignal, channel):
    with open(eyesignal, 'r') as f:
        eye = json.load(f)

    with open(browsignal, 'r') as f:
        brow = json.load(f)
    val1 = np.array(list(eye.values()))[:,channel]
    val1 = low_pass_filter(val1)

    mean1 = np.mean(val1, axis=0)

    val2 = np.array(list(brow.values()))[:,channel]
    val2 = low_pass_filter(val2)



    # Encontrar los máximos locales en la señal de referencia
    maxima_signal1 = find_peaks(val1, distance=10, height=mean1, prominence=mean1/6)
    pos1 =[]

    # Verificar si en la otra señal también hay máximos locales en las mismas posiciones
    maxima_signal2 = find_local_maxima(val2)

    print(maxima_signal2)

    for i in maxima_signal1[0]:
        pos1.append(val1[i])


    common_maxima_neighbors = []
    # Coordenada específica
    for specific_coord in maxima_signal1[0]:
        # Verificar si hay un máximo local en la coordenada específica o en sus 5 vecinos en ambas señales
        neighbors = set(range(specific_coord - 5, specific_coord + 6))
        # common_maxima_indices = set(maxima_signal1[0]).intersection(maxima_signal2)
        common_maxima_neighbors.append(list(set(maxima_signal1[0]).intersection(neighbors))[0])

    print('+++++++++++++++++++++++++++++++++++++++', '\n', common_maxima_neighbors, '\n', '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    pos2 = []
    for i in common_maxima_neighbors:
        pos2.append(val2[i])

    return common_maxima_neighbors, pos1, pos2

common_maxima_neighbors, pos1, pos2 = parkinson_blink(eye, brow, 3)

with open(eye, 'r') as f:
    eye = json.load(f)

with open(brow, 'r') as f:
    brow = json.load(f)

val1 = np.array(list(eye.values()))[:, 3]
val1 = low_pass_filter(val1)

mean1 = np.mean(val1, axis=0)

val2 = np.array(list(brow.values()))[:, 3]
val2 = low_pass_filter(val2)

plt.figure(figsize=(10, 5))
plt.plot(val1)
plt.plot(val2)
plt.scatter(list(common_maxima_neighbors), pos1, color='red')
plt.scatter(list(common_maxima_neighbors), pos2, color='green')

plt.xlabel('Frame')
plt.ylabel('Distances')
plt.legend()
plt.grid(True)
plt.show()

