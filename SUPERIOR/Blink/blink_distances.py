"""
blink_distances.py

Python file intended to obtain the euclidian distance of keypoints of both eyes to measure the amplitude and movement
of the eye during a video.
This code is generalized for any case where the json file that is input follows the same stucture as the one proposed
in this script.

Last edition: 26/07/2024
Author: Javier Hernández Rubia
"""

import json
import math
import argparse

# Function to calculate the Euclidean distance between two 3D points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2 +
                     (point1[2] - point2[2]) ** 2)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()


# Cargar el diccionario desde un archivo JSON
input_file_path = args.input[:-5]
output_file_path = args.output[:-5]

with open(f'{input_file_path}_left.json', 'r') as json_file:
    data = json.load(json_file)

# New dictionary to store the distances
distances_left = {}

for frame, points in data.items():
    upper0_points = points['Upper0']
    upper1_points = points['Upper1']
    frame_distances = []

    for p0, p1 in zip(upper0_points, upper1_points):
        distance = euclidean_distance(p0, p1)
        frame_distances.append(distance)

    distances_left[frame] = frame_distances

with open(f'{output_file_path}_left.json', 'w') as json_file:
    json.dump(distances_left, json_file, indent=4)

print(f'Distances saved in {output_file_path}')

with open(f'{input_file_path}_right.json', 'r') as json_file:
    data = json.load(json_file)

# New dictionary to store the distances
distances_right = {}

for frame, points in data.items():
    upper0_points = points['Upper0']
    upper1_points = points['Upper1']
    frame_distances = []

    for p0, p1 in zip(upper0_points, upper1_points):
        distance = euclidean_distance(p0, p1)
        frame_distances.append(distance)

    distances_right[frame] = frame_distances

with open(f'{output_file_path}_right.json', 'w') as json_file:
    json.dump(distances_right, json_file, indent=4)

print(f'Distances saved in {output_file_path}')