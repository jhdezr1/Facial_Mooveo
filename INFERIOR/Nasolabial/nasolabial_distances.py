"""
nasolabial_distances.py

Python file intended to obtain the euclidian distance of keypoints of the chin to measure the amplitude and movement
of the mouth during a video.
This code is generalized for any case where the json file that is input follows the same stucture as the one proposed
in this script.

Last edition: 24/07/2024
Author: Javier Hernández Rubia
"""

import json
import math
import argparse

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2 +
                     (point1[2] - point2[2]) ** 2)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()


# Load the dictionary from a json file
input_file_path = f'{args.input}'
output_file_path = f'{args.output}'

with open(input_file_path, 'r') as json_file:
    data = json.load(json_file)

# Create a dictionary to store the distances
distances = {}

# Calculate the distance between points corresponding to the reference point and the chin keypoints
for frame, points in data.items():
    left_points = points['left']
    upper1_points = points['right']
    frame_distances = []

    for p0, p1 in zip(upper0_points, upper1_points):
        distance = euclidean_distance(p0, p1)
        frame_distances.append(distance)

    distances[frame] = frame_distances

# Save the distances in a json file
with open(output_file_path, 'w') as json_file:
    json.dump(distances, json_file, indent=4)

print(f'Distances saved in {output_file_path}')