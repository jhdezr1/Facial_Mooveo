"""
Facial_entropy.py

This file calculates the entropy in different areas of the input face.

The parameters chosen in this script are NOT EXPECTED TO BE CHANGED, since they were found by experimentation. The only
one that can be changed is the channel, since any of the signals coming from each can be analyzed.

Last edition: 26/07/2024
Author: Javier Hernández Rubia
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import entropy
import argparse
import torch
from ultralytics import YOLO


# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
yolo_model = '/home/jhernandez2020/Escritorio/Facial_Mooveo/yolov8n-face.pt'

# Function to extract keypoints from a frame
def get_keypoints(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None

    keypoints = []
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y, landmark.z))
    return np.array(keypoints)


# Define forehead keypoint indices (Mediapipe uses 468 keypoints; indices for the forehead need to be identified)
forehead_indices = [103, 67, 109, 10, 104, 69, 108, 151, 105, 66, 107, 9, 297, 332, 337, 299, 333, 336, 296, 334, 338]
rightcheek_indices = [101, 50, 36, 147, 205, 203, 207, 206]
leftcheek_indices = [330, 266, 280, 423, 425, 411, 426, 427]
cheeks_indices = [101, 50, 36, 147, 205, 203, 207, 206, 330, 266, 280, 423, 425, 411, 426, 427]
chin_indices = [201, 200, 421, 208, 199, 428, 32, 262, 171, 175, 396]
  # Example indices, adjust based on the actual forehead region

device = torch.device("cpu")

# Function to extract forehead keypoints
def extract_forehead_keypoints(keypoints):
    return keypoints[forehead_indices], keypoints[rightcheek_indices], keypoints[leftcheek_indices], keypoints[cheeks_indices], keypoints[chin_indices]

def import_yolo_detection(yolo_model):
    model = YOLO(yolo_model).to(device)
    return model
def face_roi_yolo(model_face, conf_thresh, frame, offset):
    # Function to obtain area of image where face is located
    try:
        prediction = model_face.track(frame, persist=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Switching to CPU...")
            model_face.to("cpu")
            prediction = model_face.track(frame, persist=True)
        else:
            raise e

    bboxes = prediction[0].boxes
    face_roi = []

    if bboxes.shape[0] > 0:
        for i in range(bboxes.shape[0]):

            if bboxes[i].conf.numpy()[0] > conf_thresh:
                # print(bboxes.id.numpy().astype(int))
                bbox_pred = bboxes[i].xyxy.numpy()
                bbox_pred = bbox_pred.astype(int)
                bbox_pred[0][:2] -= offset
                bbox_pred[0][2:] += offset
                bbox_pred[0][::2] = bbox_pred[0][::2].clip(min=0, max=frame.shape[1])
                bbox_pred[0][1::2] = bbox_pred[0][1::2].clip(min=0, max=frame.shape[0])
                face_roi.append(bbox_pred[0])
    else:
        face_roi = [None]

    return face_roi

face_detector = import_yolo_detection(yolo_model)

# Function to process video and extract forehead keypoints
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    forehead_movements = []
    leftcheek_movements = []
    rightcheek_movements = []
    cheeks_movements = []
    chin_movements = []
    bboxes_frames = []
    colors_bbox = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_roi = face_roi_yolo(face_detector, 0.5, frame, 20)
        for i in range(len(face_roi)):
            if face_roi[i] is not None:
                xmin, ymin, xmax, ymax = face_roi[i]
                bboxes_frames.append(face_roi[i])
            else:
                xmin, ymin, xmax, ymax = bboxes_frames[-1]

            frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors_bbox[i], 4)

            face_cropped = frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)]
            face_cropped = np.array(face_cropped, dtype=np.uint8)
        keypoints = get_keypoints(face_cropped)
        if keypoints is not None:
            forehead_keypoints, rightcheek_keypoints, leftcheek_keypoints, cheeks_keypoints, chin_keypoints = extract_forehead_keypoints(keypoints)
            forehead_movements.append(forehead_keypoints)
            rightcheek_movements.append(rightcheek_keypoints)
            leftcheek_movements.append(leftcheek_keypoints)
            cheeks_movements.append(cheeks_keypoints)
            chin_movements.append(chin_keypoints)
        frame = cv2.cvtColor(face_cropped, cv2.COLOR_RGB2BGR)
        cv2.imshow('Resultados', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # in case the user press 'q' character we will quit the process
            break
    cap.release()
    return np.array(forehead_movements), np.array(rightcheek_movements), np.array(leftcheek_movements), np.array(cheeks_movements), np.array(chin_movements)




# Function to calculate displacements
def calculate_displacements(movements):
    displacements = np.linalg.norm(np.diff(movements, axis=0), axis=2)
    return displacements


# Function to calculate entropy
def calculate_entropy(displacements):
    flat_displacements = displacements.flatten()
    histogram, bin_edges = np.histogram(flat_displacements, bins=30, density=True)
    return entropy(histogram)


# Path to your video file
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True)
args = parser.parse_args()
video_path = args.video
# Process video to get forehead movements
forehead_movements, rightcheek_movements, leftcheek_movements, cheeks_movements, chin_movements = process_video(video_path)

# Calculate displacements
displacements_forehead = calculate_displacements(forehead_movements)
displacements_leftcheek = calculate_displacements(leftcheek_movements)
displacements_rightcheek = calculate_displacements(rightcheek_movements)
displacements_cheeks = calculate_displacements(cheeks_movements)
displacements_chin = calculate_displacements(chin_movements)

# Calculate movement entropy
entropy_forehead = calculate_entropy(displacements_forehead)
entropy_leftcheek = calculate_entropy(displacements_leftcheek)
entropy_rightcheek = calculate_entropy(displacements_rightcheek)
entropy_cheeks = calculate_entropy(displacements_cheeks)
entropy_chin = calculate_entropy(displacements_chin)

print(f"Movement Entropy for: \n \t * Forehead -->{entropy_forehead} \n \t * Left Cheek -->{entropy_leftcheek} \n \t * Right Cheek -->{entropy_rightcheek} \n \t * Cheeks -->{entropy_cheeks} \n \t * Chin -->{entropy_chin}")
